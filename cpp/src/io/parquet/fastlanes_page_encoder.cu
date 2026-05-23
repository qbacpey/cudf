/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Note (FL-P3-R1 follow-up): this source file is intentionally source-included from
// `page_enc.cu` and therefore inherits the enclosing `cudf::io::parquet::detail::` namespace
// plus the local templated kernels defined there (e.g. `gpuGatherSinglePageTyped`,
// `gpuEncodePageLevels`, `gpuEncodeCpuPages`).
//
// FL-P3-R4 removed the test-only A/B encode-path selector that previously lived here, so this
// file no longer pulls in the slim path-selector header.

namespace fastlanes_encode_stage {

namespace parquet_fastlanes = cudf::io::parquet::detail::fastlanes;

// =============================================================================
// Page categorization (single-pass host-side classifier)
// =============================================================================

/**
 * @brief Lightweight per-page category record built during the single-pass classification.
 *
 * Carries everything the encode loop needs to dispatch to the right encoder without re-reading
 * the host page metadata. `page_idx` is the original page slot in `pages` and is the canonical
 * key for upload-buffer placement, so per-category iteration cannot perturb downstream layout.
 */
struct fastlanes_page_category {
  size_t page_idx;
  uint32_t num_values;
  encode_kernel_mask kernel_mask;
  Encoding encoding;
};

/**
 * @brief Pages grouped by FastLanes encoding mode, preserving original page_idx order.
 *
 * Each per-mode vector is appended to in ascending `page_idx`, so the categorized encode
 * loop places per-page payloads in the same upload-buffer slots that the pre-refactor
 * straight-line loop did. (Verified by the FL-P3-R2 byte-identity A/B harness before
 * removal in FL-P3-R4.)
 */
struct fastlanes_categorized_pages {
  std::vector<fastlanes_page_category> raw32_pages;
  std::vector<fastlanes_page_category> split64_pages;
  std::vector<fastlanes_page_category> native64_pages;

  [[nodiscard]] size_t total_categorized() const
  {
    return raw32_pages.size() + split64_pages.size() + native64_pages.size();
  }
};

/**
 * @brief Single-pass host-side classifier of FastLanes-eligible pages.
 *
 * Reads only `kernel_mask` and `num_leaf_values` (already on host after the copy-down) and
 * sorts each page into the appropriate per-encoding bucket. Non-FastLanes pages and empty
 * pages are filtered out here so the encode pass can skip them implicitly.
 */
fastlanes_categorized_pages categorize_fastlanes_pages(std::vector<EncPage> const& host_pages)
{
  fastlanes_categorized_pages result;

  for (size_t page_idx = 0; page_idx < host_pages.size(); ++page_idx) {
    auto const& page = host_pages[page_idx];
    if (not is_fastlanes_mask(page.kernel_mask)) { continue; }
    if (page.num_leaf_values == 0) { continue; }

    auto const encoding = fastlanes_encoding_for_mask(page.kernel_mask);
    fastlanes_page_category const cat{
      page_idx, page.num_leaf_values, page.kernel_mask, encoding};

    switch (encoding) {
      case Encoding::FASTLANE_BITPACK_RAW: result.raw32_pages.push_back(cat); break;
      case Encoding::FASTLANE_BITPACK_SPLIT64: result.split64_pages.push_back(cat); break;
      case Encoding::FASTLANES_DELTA_BINARY: result.native64_pages.push_back(cat); break;
      default: break;
    }
  }

  return result;
}

// =============================================================================
// Upload-side staging and downstream kernel launch
// =============================================================================

struct fastlanes_cpu_upload_buffers {
  explicit fastlanes_cpu_upload_buffers(size_t num_pages)
    : host_upload_ptrs(num_pages, nullptr), host_upload_sizes(num_pages, 0)
  {
    encoded_buffers.reserve(num_pages);
  }

  std::vector<rmm::device_buffer> encoded_buffers;
  std::vector<uint8_t*> host_upload_ptrs;
  std::vector<uint32_t> host_upload_sizes;
};

std::vector<EncPage> copy_fastlanes_pages_to_host(device_span<EncPage> pages,
                                                  rmm::cuda_stream_view stream)
{
  std::vector<EncPage> host_pages(pages.size());
  cudaMemcpyAsync(host_pages.data(),
                  pages.data(),
                  pages.size_bytes(),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return host_pages;
}

template <typename HeaderT>
void validate_fastlanes_pre_delta_policy(encode_kernel_mask kernel_mask, HeaderT const& hdr)
{
  if (!is_fastlanes_mask(kernel_mask)) { return; }

  auto const encoding = fastlanes_encoding_for_mask(kernel_mask);
  auto const is_valid = [&]() {
    switch (encoding) {
      case Encoding::FASTLANE_BITPACK_RAW:
        return ::fastlanes::is_pre_delta_valid_for_raw32(hdr.pre_delta);
      case Encoding::FASTLANE_BITPACK_SPLIT64:
        return ::fastlanes::is_pre_delta_valid_for_split64(hdr.pre_delta);
      case Encoding::FASTLANES_DELTA_BINARY:
        return ::fastlanes::is_pre_delta_valid_for_native64(hdr.pre_delta);
      default: return false;
    }
  }();

  if (!is_valid) {
    throw std::invalid_argument("FastLanes: invalid PRE_DELTA policy for selected page encoding mode");
  }
}

void ensure_fastlanes_page_fits_reserved_size(size_t page_idx,
                                              EncPage const& page,
                                              uint32_t encoded_blob_size)
{
  if (encoded_blob_size > page.max_data_size) {
    throw std::runtime_error("FastLanes encoded page exceeds reserved page size: page=" +
                             std::to_string(page_idx) +
                             " encoded=" + std::to_string(encoded_blob_size) +
                             " reserved=" + std::to_string(page.max_data_size) +
                             " num_leaf=" + std::to_string(page.num_leaf_values));
  }
}

void upload_fastlanes_results_and_launch(device_span<EncPage> pages,
                                         bool write_v2_headers,
                                         device_span<device_span<uint8_t const>> comp_in,
                                         device_span<device_span<uint8_t>> comp_out,
                                         device_span<codec_exec_result> comp_results,
                                         fastlanes_cpu_upload_buffers const& upload_buffers,
                                         uint32_t fastlanes_kernel_mask_bits,
                                         rmm::cuda_stream_view stream)
{
  auto const num_pages = pages.size();
  rmm::device_uvector<uint8_t*> d_upload_ptrs(num_pages, stream);
  rmm::device_uvector<uint32_t> d_upload_sizes(num_pages, stream);
  cudaMemcpyAsync(d_upload_ptrs.data(),
                  upload_buffers.host_upload_ptrs.data(),
                  num_pages * sizeof(uint8_t*),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaMemcpyAsync(d_upload_sizes.data(),
                  upload_buffers.host_upload_sizes.data(),
                  num_pages * sizeof(uint32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, stream.value()>>>(
    pages,
    write_v2_headers,
    static_cast<encode_kernel_mask>(fastlanes_kernel_mask_bits));

  gpuEncodeCpuPages<encode_block_size><<<num_pages, encode_block_size, 0, stream.value()>>>(
    pages,
    comp_in,
    comp_out,
    comp_results,
    d_upload_ptrs.data(),
    d_upload_sizes.data(),
    write_v2_headers);
}

// =============================================================================
// Batched per-encoding encode path
// =============================================================================
//
// Within each FastLanes encoding mode (RAW32, SPLIT64, NATIVE64):
//   1. Allocate per-page gather buffers and launch ALL gather kernels on the same stream
//      with NO per-page sync.
//   2. Hand the host-side array of device gather pointers to the encoder's `encode_pages`
//      batch API in a single call. The encoder is responsible for any internal D->H/H->D
//      shuffling needed to produce the encoded device blobs.
//   3. Batch-read the page headers (first `header_probe_bytes` of each encoded blob) D->H
//      with one `cudaStreamSynchronize` for the whole category, then run the PRE_DELTA
//      header-validation check on each.
//   4. Splice the per-page encoded device buffers / pointers / sizes into `upload_buffers`
//      keyed by the original `page_idx`, so the downstream `gpuEncodePageLevels` and
//      `gpuEncodeCpuPages` launches consume identical slot placement.
//
// Sync audit: this path emits, from our code, exactly ONE `cudaStreamSynchronize` per
// non-empty encoding category (inside `validate_category_headers_batched`). Combined with
// the single entry-time sync inside `copy_fastlanes_pages_to_host`, total sync count is at
// most 1 + 3 = 4 per `run_fastlanes_cpu_encode` call, independent of page count. The encoder
// may emit additional internal syncs; those are not in this TU's scope.

/**
 * @brief Lazy holder for the three per-encoding encoder instances.
 *
 * Encoders are constructed on first use so workloads that touch only one encoding pay no
 * setup cost for the other two.
 */
struct fastlanes_encoder_lazy_pool {
  std::unique_ptr<parquet_fastlanes::FastLanesInt32Encoder> encoder_i32;
  std::unique_ptr<parquet_fastlanes::FastLanesInt64Split32Encoder> encoder_i64_split32;
  std::unique_ptr<parquet_fastlanes::FastLanesInt64NativeEncoder> encoder_i64_native;

  parquet_fastlanes::FastLanesInt32Encoder& get_i32()
  {
    if (!encoder_i32) {
      encoder_i32 = std::make_unique<parquet_fastlanes::FastLanesInt32Encoder>();
    }
    return *encoder_i32;
  }
  parquet_fastlanes::FastLanesInt64Split32Encoder& get_i64_split32()
  {
    if (!encoder_i64_split32) {
      encoder_i64_split32 =
        std::make_unique<parquet_fastlanes::FastLanesInt64Split32Encoder>();
    }
    return *encoder_i64_split32;
  }
  parquet_fastlanes::FastLanesInt64NativeEncoder& get_i64_native()
  {
    if (!encoder_i64_native) {
      encoder_i64_native = std::make_unique<parquet_fastlanes::FastLanesInt64NativeEncoder>();
    }
    return *encoder_i64_native;
  }
};

/**
 * @brief Single host-visible probe size used for batched header validation.
 *
 * The PageHeader fields that participate in validation live in the first 24 bytes
 * (OFFSET_MIN_VALUE_HIGH_BITS = 20, + sizeof(uint32_t)). Rounded up to 32 for alignment so
 * a single tight D->H copy per page covers everything `PageHeader::deserialize` reads. The
 * full on-disk header is larger but the trailing bytes do not affect PRE_DELTA validation.
 */
constexpr size_t header_probe_bytes = 32;

/**
 * @brief Batched header validator.
 *
 * Issues one async D->H copy per page from the encoded device blob's leading bytes onto a
 * single host scratch buffer, then performs a SINGLE `cudaStreamSynchronize` for the entire
 * category before running per-page `validate_fastlanes_pre_delta_policy`.
 */
void validate_category_headers_batched(
  std::vector<fastlanes_page_category> const& category,
  std::vector<uint8_t*> const& enc_ptrs,
  std::vector<uint32_t> const& enc_sizes,
  rmm::cuda_stream_view stream)
{
  size_t const n = category.size();
  if (n == 0) { return; }

  std::vector<uint8_t> header_scratch(n * header_probe_bytes, uint8_t{0});

  for (size_t i = 0; i < n; ++i) {
    if (enc_ptrs[i] == nullptr || enc_sizes[i] == 0) { continue; }
    auto const probe = std::min(static_cast<size_t>(enc_sizes[i]), header_probe_bytes);
    cudaMemcpyAsync(header_scratch.data() + i * header_probe_bytes,
                    enc_ptrs[i],
                    probe,
                    cudaMemcpyDeviceToHost,
                    stream.value());
  }
  cudaStreamSynchronize(stream.value());

  for (size_t i = 0; i < n; ++i) {
    if (enc_ptrs[i] == nullptr || enc_sizes[i] == 0) { continue; }
    auto const hdr =
      ::fastlanes::PageHeader::deserialize(header_scratch.data() + i * header_probe_bytes);
    validate_fastlanes_pre_delta_policy(category[i].kernel_mask, hdr);
  }
}

/**
 * @brief Templated batched-encode driver for one encoding category.
 *
 * `ValueT` is the signed integer type the encoder consumes (`int32_t` for RAW32, `int64_t`
 * for SPLIT64/NATIVE64). The gather kernel produces unsigned buffers of the matching width.
 *
 * Lifetimes: `gather_buffers` owns the RMM allocations for the gather staging area and must
 * outlive the encoder's `encode_pages` call. The encoder reads from these device pointers
 * during its own D->H staging; once it returns, the gather buffers can be released.
 */
template <typename ValueT, typename EncoderT>
void encode_category_batched(device_span<EncPage> pages,
                             std::vector<EncPage> const& host_pages,
                             std::vector<fastlanes_page_category> const& category,
                             EncoderT& encoder,
                             fastlanes_cpu_upload_buffers& upload_buffers,
                             rmm::cuda_stream_view stream)
{
  size_t const n = category.size();
  if (n == 0) { return; }

  using UnsignedT = std::make_unsigned_t<ValueT>;
  static_assert(sizeof(UnsignedT) == sizeof(ValueT),
                "Gather buffer unsigned type must match encoder value-type width");

  // Phase 1: pre-reserve gather staging and launch ALL gather kernels with NO per-page sync.
  // `gather_buffers` is reserved upfront so emplace_back never reallocates and invalidates
  // the device pointers we stash in `gather_ptrs`.
  std::vector<rmm::device_uvector<UnsignedT>> gather_buffers;
  gather_buffers.reserve(n);
  std::vector<ValueT*> gather_ptrs(n, nullptr);
  std::vector<uint32_t> gather_counts(n, 0);

  for (size_t i = 0; i < n; ++i) {
    auto const& cat = category[i];
    gather_buffers.emplace_back(cat.num_values, stream);
    gpuGatherSinglePageTyped<UnsignedT, encode_block_size>
      <<<1, encode_block_size, 0, stream.value()>>>(
        pages, cat.page_idx, gather_buffers.back().data(), nullptr);
    gather_ptrs[i]   = reinterpret_cast<ValueT*>(gather_buffers.back().data());
    gather_counts[i] = cat.num_values;
  }

  // Phase 2: single batched encode. Any internal sync the encoder needs to read the gather
  // buffers will naturally drain the queued gather kernels above (same stream).
  auto encoded = encoder.encode_pages(gather_ptrs, gather_counts, stream);
  auto& enc_buffers = std::get<0>(encoded);
  auto& enc_ptrs    = std::get<1>(encoded);
  auto& enc_sizes   = std::get<2>(encoded);

  // Phase 3: batched header validation -- one sync covers the whole category.
  validate_category_headers_batched(category, enc_ptrs, enc_sizes, stream);

  // Phase 4: register per-page uploads. Slot placement is keyed on `cat.page_idx` so the
  // downstream kernels see the same per-page layout regardless of category iteration order.
  for (size_t i = 0; i < n; ++i) {
    auto const& cat = category[i];
    auto const sz   = enc_sizes[i];

    ensure_fastlanes_page_fits_reserved_size(cat.page_idx, host_pages[cat.page_idx], sz);

    upload_buffers.encoded_buffers.push_back(std::move(enc_buffers[i]));
    upload_buffers.host_upload_ptrs[cat.page_idx]  = enc_ptrs[i];
    upload_buffers.host_upload_sizes[cat.page_idx] = sz;
  }
}

// =============================================================================
// Entry point
// =============================================================================

void run_fastlanes_cpu_encode(device_span<EncPage> pages,
                              bool write_v2_headers,
                              device_span<device_span<uint8_t const>> comp_in,
                              device_span<device_span<uint8_t>> comp_out,
                              device_span<codec_exec_result> comp_results,
                              uint32_t fastlanes_kernel_mask_bits,
                              rmm::cuda_stream_view stream)
{
  auto host_pages = copy_fastlanes_pages_to_host(pages, stream);
  fastlanes_cpu_upload_buffers upload_buffers(pages.size());

  auto const categorized = categorize_fastlanes_pages(host_pages);

  fastlanes_encoder_lazy_pool encoders;

  if (!categorized.raw32_pages.empty()) {
    encode_category_batched<int32_t>(pages,
                                     host_pages,
                                     categorized.raw32_pages,
                                     encoders.get_i32(),
                                     upload_buffers,
                                     stream);
  }
  if (!categorized.split64_pages.empty()) {
    encode_category_batched<int64_t>(pages,
                                     host_pages,
                                     categorized.split64_pages,
                                     encoders.get_i64_split32(),
                                     upload_buffers,
                                     stream);
  }
  if (!categorized.native64_pages.empty()) {
    encode_category_batched<int64_t>(pages,
                                     host_pages,
                                     categorized.native64_pages,
                                     encoders.get_i64_native(),
                                     upload_buffers,
                                     stream);
  }

  upload_fastlanes_results_and_launch(pages,
                                      write_v2_headers,
                                      comp_in,
                                      comp_out,
                                      comp_results,
                                      upload_buffers,
                                      fastlanes_kernel_mask_bits,
                                      stream);
}

}  // namespace fastlanes_encode_stage
