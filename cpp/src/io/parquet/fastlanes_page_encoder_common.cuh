/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Note (FL-P4-R2): this header is intentionally source-included from `fastlanes_page_encoder.cu`,
// which is itself source-included from `page_enc.cu`. It therefore inherits the enclosing
// `cudf::io::parquet::detail::` namespace plus the local templated kernels defined in
// `page_enc.cu` (e.g. `gpuGatherSinglePageTyped`, `gpuEncodePageLevels`, `gpuEncodeCpuPages`).
//
// All helpers below are marked `inline` so the source-include model continues to work even if
// this header is later included from more than one translation unit; no new exported libcudf
// symbols are introduced.

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
 *
 * FL-P4-R2 dropped the `split64_pages` member: SPLIT64 is hard-refused upstream in
 * writer_impl.cu (see FL-P4-R1), so it can no longer reach the encoder.
 */
struct fastlanes_categorized_pages {
  std::vector<fastlanes_page_category> raw32_pages;
  std::vector<fastlanes_page_category> native64_pages;

  [[nodiscard]] size_t total_categorized() const
  {
    return raw32_pages.size() + native64_pages.size();
  }
};

/**
 * @brief Single-pass host-side classifier of FastLanes-eligible pages.
 *
 * Reads only `kernel_mask` and `num_leaf_values` (already on host after the copy-down) and
 * sorts each page into the appropriate per-encoding bucket. Non-FastLanes pages and empty
 * pages are filtered out here so the encode pass can skip them implicitly.
 */
inline fastlanes_categorized_pages categorize_fastlanes_pages(
  std::vector<EncPage> const& host_pages)
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

inline std::vector<EncPage> copy_fastlanes_pages_to_host(device_span<EncPage> pages,
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
inline void validate_fastlanes_pre_delta_policy(encode_kernel_mask kernel_mask, HeaderT const& hdr)
{
  if (!is_fastlanes_mask(kernel_mask)) { return; }

  auto const encoding = fastlanes_encoding_for_mask(kernel_mask);
  auto const is_valid = [&]() {
    switch (encoding) {
      case Encoding::FASTLANE_BITPACK_RAW:
        return ::fastlanes::is_pre_delta_valid_for_raw32(hdr.pre_delta);
      case Encoding::FASTLANES_DELTA_BINARY:
        return ::fastlanes::is_pre_delta_valid_for_native64(hdr.pre_delta);
      default: return false;
    }
  }();

  if (!is_valid) {
    throw std::invalid_argument("FastLanes: invalid PRE_DELTA policy for selected page encoding mode");
  }
}

inline void ensure_fastlanes_page_fits_reserved_size(size_t page_idx,
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

inline void upload_fastlanes_results_and_launch(
  device_span<EncPage> pages,
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

  auto const kernel_mask = static_cast<encode_kernel_mask>(fastlanes_kernel_mask_bits);
  gpuEncodePageLevels<encode_block_size><<<num_pages, encode_block_size, 0, stream.value()>>>(
    pages, write_v2_headers, kernel_mask);

  gpuEncodeCpuPages<encode_block_size><<<num_pages, encode_block_size, 0, stream.value()>>>(
    pages,
    comp_in,
    comp_out,
    comp_results,
    d_upload_ptrs.data(),
    d_upload_sizes.data(),
    write_v2_headers,
    kernel_mask);
}

// =============================================================================
// Batched per-encoding encode path
// =============================================================================
//
// Within each FastLanes encoding mode (RAW32, NATIVE64):
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
// most 1 + 2 = 3 per `run_fastlanes_cpu_encode` call, independent of page count (FL-P4-R2
// reduced the 1 + 3 bound from FL-P3-R3 by removing the SPLIT64 category). The encoder may
// emit additional internal syncs; those are not in this TU's scope.

/**
 * @brief Lazy holder for the two per-encoding encoder instances.
 *
 * Encoders are constructed on first use so workloads that touch only one encoding pay no
 * setup cost for the other one. FL-P4-R2 dropped the SPLIT64 encoder member; the
 * underlying `FastLanesInt64Split32Encoder` symbol still exists in the FastLanes encoder
 * library but is production-unreachable.
 */
struct fastlanes_encoder_lazy_pool {
  std::unique_ptr<parquet_fastlanes::FastLanesInt32Encoder> encoder_i32;
  std::unique_ptr<parquet_fastlanes::FastLanesInt64NativeEncoder> encoder_i64_native;

  parquet_fastlanes::FastLanesInt32Encoder& get_i32()
  {
    if (!encoder_i32) {
      encoder_i32 = std::make_unique<parquet_fastlanes::FastLanesInt32Encoder>();
    }
    return *encoder_i32;
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
inline void validate_category_headers_batched(
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

// FL-P4-R3 split the previous `encode_category_batched<ValueT, EncoderT>` template into two
// dedicated non-template variants (`encode_raw32_category_batched` and
// `encode_native64_category_batched`) that live in `fastlanes_page_encoder_raw32.cuh` and
// `fastlanes_page_encoder_native64.cuh` respectively. The shared shape (gather, encode,
// validate-headers, register-upload) is preserved verbatim across the two; only the encoder
// type and gather value width differ.

}  // namespace fastlanes_encode_stage
