/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace fastlanes_encode_stage {

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

fastlanes_page_type_info gather_fastlanes_page_type_info(device_span<EncPage> pages,
                                                         size_t page_idx,
                                                         rmm::cuda_stream_view stream)
{
  rmm::device_scalar<fastlanes_page_type_info> d_type_info(fastlanes_page_type_info{}, stream);
  gpuGatherSinglePageTyped<int32_t, encode_block_size>
    <<<1, encode_block_size, 0, stream.value()>>>(pages, page_idx, nullptr, d_type_info.data());
  return d_type_info.value(stream);
}

void maybe_log_fastlanes_host_page_meta(EncPage const& page, size_t page_idx, uint32_t chunk_page_index)
{
  if (!fastlanes::debug::is_header_enabled()) { return; }
  std::cout << "[FL HOST META ] page=" << page_idx << " chunk=" << page.chunk_id
            << " chunk_page=" << chunk_page_index << " start_row=" << page.start_row
            << " num_rows=" << page.num_rows << " num_leaf=" << page.num_leaf_values
            << " max_data_size=" << page.max_data_size << "\n";
}

void maybe_log_fastlanes_int32_vector_boundaries(rmm::device_uvector<uint32_t> const& gather_buffer,
                                                 uint32_t num_values,
                                                 size_t page_idx,
                                                 rmm::cuda_stream_view stream)
{
  if (!fastlanes::debug::is_vector_boundary_enabled()) { return; }

  auto print_u32_at = [&](uint32_t idx, char const* tag) {
    if (idx >= num_values) return;
    uint32_t v{};
    cudaMemcpyAsync(
      &v, gather_buffer.data() + idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value());
    cudaStreamSynchronize(stream.value());
    std::cout << "[FL GATHER   ] page=" << page_idx << " " << tag << " idx=" << idx
              << " val=0x" << std::hex << v << std::dec << "\n";
  };

  print_u32_at(0, "head");
  print_u32_at(1, "head");
  if (num_values > 1023) { print_u32_at(1023, "boundary"); }
  if (num_values > 1024) { print_u32_at(1024, "boundary"); }
}

template <typename HeaderT>
void validate_fastlanes_pre_delta_policy(encode_kernel_mask kernel_mask, HeaderT const& hdr)
{
  if (!is_fastlanes_mask(kernel_mask)) { return; }

  auto const encoding = fastlanes_encoding_for_mask(kernel_mask);
  auto const is_valid = [&]() {
    switch (encoding) {
      case Encoding::FASTLANE_BITPACK_RAW:
        return fastlanes::is_pre_delta_valid_for_raw32(hdr.pre_delta);
      case Encoding::FASTLANE_BITPACK_SPLIT64:
        return fastlanes::is_pre_delta_valid_for_split64(hdr.pre_delta);
      case Encoding::FASTLANES_DELTA_BINARY:
        return fastlanes::is_pre_delta_valid_for_native64(hdr.pre_delta);
      default: return false;
    }
  }();

  if (!is_valid) {
    throw std::invalid_argument("FastLanes: invalid PRE_DELTA policy for selected page encoding mode");
  }
}

constexpr char const* fastlanes_mode_name(Encoding page_encoding)
{
  switch (page_encoding) {
    case Encoding::FASTLANE_BITPACK_RAW: return "RAW32";
    case Encoding::FASTLANE_BITPACK_SPLIT64: return "SPLIT64";
    case Encoding::FASTLANES_DELTA_BINARY: return "NATIVE64";
    default: return "UNKNOWN";
  }
}

template <typename HeaderT>
void maybe_log_fastlanes_encoded_page(size_t page_idx,
                                      uint32_t chunk_id,
                                      uint32_t chunk_page_index,
                                      Encoding page_encoding,
                                      HeaderT const& hdr,
                                      fastlanes_cudf::EncodedPageResult const& result)
{
  if (!fastlanes::debug::is_workload_enabled()) { return; }

  auto const* payload_u32 = reinterpret_cast<uint32_t const*>(
    fastlanes::PageHeader::payload_ptr(result.host_blob.data()));
  auto const page_mode_name = fastlanes_mode_name(page_encoding);
  auto const is_split64     = page_encoding == Encoding::FASTLANE_BITPACK_SPLIT64;

  std::cout << "[FL ENCODED  ] page=" << page_idx << " chunk=" << chunk_id
            << " chunk_page=" << chunk_page_index << " mode=" << page_mode_name
            << " pre_delta=" << (hdr.pre_delta ? "true" : "false")
            << " cast=" << fastlanes::debug::cast_mode_str(result.cast_mode)
            << " hdr=" << fastlanes::PageHeader::header_size();
  if (is_split64) {
    std::cout << " bw_lo=" << static_cast<int>(hdr.component_bitwidth_low)
              << " bw_hi=" << static_cast<int>(hdr.component_bitwidth_high)
              << " min_lo=0x" << std::hex << hdr.min_value_low_bits
              << " min_hi=0x" << hdr.min_value_high_bits << std::dec;
  } else {
    std::cout << " bw=" << static_cast<int>(hdr.component_bitwidth_low)
              << " min=0x" << std::hex << hdr.min_value_bits() << std::dec;
  }
  std::cout
            << " orig=" << hdr.original_count << " pad=" << hdr.padded_count
            << " body=" << hdr.body_size << " blob=" << result.total_size() << "\n";
  if (hdr.body_size >= sizeof(uint32_t)) {
    std::cout << "[FL PAYLOAD  ] page=" << page_idx << " payload[0]=0x" << std::hex
              << payload_u32[0] << std::dec << "\n";
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

uint32_t register_fastlanes_upload(size_t page_idx,
                                   fastlanes_cudf::EncodedPageResult&& result,
                                   fastlanes_cpu_upload_buffers& upload_buffers)
{
  auto const encoded_blob_size = static_cast<uint32_t>(result.total_size());
  upload_buffers.encoded_buffers.push_back(std::move(result.device_blob));
  upload_buffers.host_upload_ptrs[page_idx] =
    static_cast<uint8_t*>(upload_buffers.encoded_buffers.back().data());
  upload_buffers.host_upload_sizes[page_idx] = encoded_blob_size;
  return encoded_blob_size;
}

fastlanes_cudf::EncodedPageResult encode_fastlanes_int32_page(
  device_span<EncPage> pages,
  size_t page_idx,
  uint32_t num_values,
  fastlanes_cudf::FastLanesInt32Encoder& encoder,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint32_t> gather_buffer(num_values, stream);
  gpuGatherSinglePageTyped<uint32_t, encode_block_size>
    <<<1, encode_block_size, 0, stream.value()>>>(pages, page_idx, gather_buffer.data(), nullptr);
  cudaStreamSynchronize(stream.value());

  maybe_log_fastlanes_int32_vector_boundaries(gather_buffer, num_values, page_idx, stream);
  return encoder.encode_page(reinterpret_cast<int32_t const*>(gather_buffer.data()),
                             num_values,
                             stream);
}

fastlanes_cudf::EncodedPageResult encode_fastlanes_int64_page(
  device_span<EncPage> pages,
  size_t page_idx,
  uint32_t num_values,
  fastlanes_cudf::FastLanesInt64Split32Encoder& encoder,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint64_t> gather_buffer(num_values, stream);
  gpuGatherSinglePageTyped<uint64_t, encode_block_size>
    <<<1, encode_block_size, 0, stream.value()>>>(pages, page_idx, gather_buffer.data(), nullptr);
  cudaStreamSynchronize(stream.value());

  return encoder.encode_page(reinterpret_cast<int64_t const*>(gather_buffer.data()),
                             num_values,
                             stream);
}

fastlanes_cudf::EncodedPageResult encode_fastlanes_int64_native_page(
  device_span<EncPage> pages,
  size_t page_idx,
  uint32_t num_values,
  fastlanes_cudf::FastLanesInt64NativeEncoder& encoder,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint64_t> gather_buffer(num_values, stream);
  gpuGatherSinglePageTyped<uint64_t, encode_block_size>
    <<<1, encode_block_size, 0, stream.value()>>>(pages, page_idx, gather_buffer.data(), nullptr);
  cudaStreamSynchronize(stream.value());

  return encoder.encode_page(reinterpret_cast<int64_t const*>(gather_buffer.data()),
                             num_values,
                             stream);
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

  std::unique_ptr<fastlanes_cudf::FastLanesInt32Encoder> encoder_i32;
  std::unique_ptr<fastlanes_cudf::FastLanesInt64Split32Encoder> encoder_i64_split32;
  std::unique_ptr<fastlanes_cudf::FastLanesInt64NativeEncoder> encoder_i64_native;

  auto get_encoder_i32 = [&]() -> fastlanes_cudf::FastLanesInt32Encoder& {
    if (!encoder_i32) { encoder_i32 = std::make_unique<fastlanes_cudf::FastLanesInt32Encoder>(false); }
    return *encoder_i32;
  };

  auto get_encoder_i64_split32 = [&]() -> fastlanes_cudf::FastLanesInt64Split32Encoder& {
    if (!encoder_i64_split32) {
      encoder_i64_split32 = std::make_unique<fastlanes_cudf::FastLanesInt64Split32Encoder>(false);
    }
    return *encoder_i64_split32;
  };

  auto get_encoder_i64_native = [&]() -> fastlanes_cudf::FastLanesInt64NativeEncoder& {
    if (!encoder_i64_native) {
      encoder_i64_native = std::make_unique<fastlanes_cudf::FastLanesInt64NativeEncoder>(false);
    }
    return *encoder_i64_native;
  };

  std::unordered_map<uint32_t, uint32_t> chunk_page_counters;
  for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
    if (not is_fastlanes_mask(host_pages[page_idx].kernel_mask)) { continue; }

    auto const chunk_id         = host_pages[page_idx].chunk_id;
    auto const chunk_page_index = chunk_page_counters[chunk_id]++;
    uint32_t const num_values   = host_pages[page_idx].num_leaf_values;
    if (num_values == 0) { continue; }

    maybe_log_fastlanes_host_page_meta(host_pages[page_idx], page_idx, chunk_page_index);

    auto const type_info = gather_fastlanes_page_type_info(pages, page_idx, stream);

    if (type_info.physical_type == Type::INT32) {
      auto result = encode_fastlanes_int32_page(
        pages, page_idx, num_values, get_encoder_i32(), stream);

      auto const hdr = fastlanes::PageHeader::deserialize(result.host_blob.data());
      validate_fastlanes_pre_delta_policy(host_pages[page_idx].kernel_mask, hdr);

      auto const page_encoding = fastlanes_encoding_for_mask(host_pages[page_idx].kernel_mask);
      maybe_log_fastlanes_encoded_page(
        page_idx, chunk_id, chunk_page_index, page_encoding, hdr, result);

      auto const encoded_blob_size = static_cast<uint32_t>(result.total_size());
      ensure_fastlanes_page_fits_reserved_size(page_idx, host_pages[page_idx], encoded_blob_size);

      register_fastlanes_upload(page_idx, std::move(result), upload_buffers);

      if (fastlanes::debug::is_workload_enabled()) {
        std::cout << "[FL UPLOAD   ] page=" << page_idx
                  << " ptr=" << static_cast<void*>(upload_buffers.host_upload_ptrs[page_idx])
                  << " size=" << upload_buffers.host_upload_sizes[page_idx] << "\n";
      }
    } else if (type_info.physical_type == Type::INT64 &&
               (type_info.logical_type == cudf::type_id::INT64 ||
                type_info.logical_type == cudf::type_id::UINT64)) {
      auto const encoding = fastlanes_encoding_for_mask(host_pages[page_idx].kernel_mask);

      auto result = [&]() {
        if (encoding == Encoding::FASTLANE_BITPACK_SPLIT64) {
          return encode_fastlanes_int64_page(
            pages, page_idx, num_values, get_encoder_i64_split32(), stream);
        }
        if (encoding == Encoding::FASTLANES_DELTA_BINARY) {
          return encode_fastlanes_int64_native_page(
            pages, page_idx, num_values, get_encoder_i64_native(), stream);
        }
        throw std::invalid_argument(
          "FastLanes INT64 path only supports SPLIT64 and FASTLANES_DELTA_BINARY encodings");
      }();

      auto const hdr = fastlanes::PageHeader::deserialize(result.host_blob.data());
      validate_fastlanes_pre_delta_policy(host_pages[page_idx].kernel_mask, hdr);

      maybe_log_fastlanes_encoded_page(
        page_idx, chunk_id, chunk_page_index, encoding, hdr, result);

      auto const encoded_blob_size = static_cast<uint32_t>(result.total_size());
      ensure_fastlanes_page_fits_reserved_size(page_idx, host_pages[page_idx], encoded_blob_size);

      register_fastlanes_upload(page_idx, std::move(result), upload_buffers);

      if (fastlanes::debug::is_workload_enabled()) {
        std::cout << "[FL UPLOAD   ] page=" << page_idx
                  << " ptr=" << static_cast<void*>(upload_buffers.host_upload_ptrs[page_idx])
                  << " size=" << upload_buffers.host_upload_sizes[page_idx] << "\n";
      }
    } else {
      throw std::invalid_argument(
        "FastLanes encoding supports selected INT32 and INT64 logical types in flat pages");
    }
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