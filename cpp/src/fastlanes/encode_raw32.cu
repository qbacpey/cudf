#include "encode_common.hpp"

#include <cstddef>

namespace cudf::io::parquet::detail::fastlanes {

namespace {

EncodedPageResult encode_scalar32_page(std::vector<int32_t> const& host_input,
                                       uint32_t count,
                                       uint64_t padded,
                                       ::fastlanes::TypeCastMode cast_mode,
                                       cuda::stream_ref stream)
{
  EncodedPageResult result;

  auto normalized = detail::normalize_page_data<int32_t>(
    host_input.data(), count, static_cast<uint32_t>(padded));
  auto const bitwidth = normalized.bitwidth;

  size_t const encoded_bytes = ::fastlanes::encoded_size_bytes(padded, bitwidth);
  if (encoded_bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("FastLanesEncoder: encoded body size exceeds 32-bit header limit");
  }
  auto const encoded_bytes_u32 = static_cast<uint32_t>(encoded_bytes);
  std::vector<detail::unsigned_t<int32_t>> encoded_body(
    encoded_bytes / sizeof(detail::unsigned_t<int32_t>));

  detail::encode_vectors<int32_t>(normalized.values.data(), padded, bitwidth, encoded_body.data());

  result.host_blob = ::fastlanes::PageHeader::serialize_scalar32(
    bitwidth,
    count,
    static_cast<uint32_t>(padded),
    static_cast<uint32_t>(normalized.min_value),
    reinterpret_cast<uint8_t const*>(encoded_body.data()),
    encoded_bytes_u32,
    ::fastlanes::default_pre_delta_for_raw32());
  detail::upload_encoded_blob(result, stream);

  result.bitwidth       = bitwidth;
  result.cast_mode      = cast_mode;
  result.original_count = count;
  result.padded_count   = static_cast<uint32_t>(padded);
  result.body_size      = encoded_bytes_u32;
  result.min_value      = normalized.min_value;

  return result;
}

EncodedPageResult encode_scalar32_page_helper(int32_t const* d_input,
                                              uint32_t count,
                                              cuda::stream_ref stream)
{
  if (count == 0) { return detail::create_empty_scalar32_result(stream); }

  if (d_input == nullptr) {
    throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
  }

  uint64_t const padded = ::fastlanes::padded_count(count);
  std::vector<int32_t> host_input(padded, int32_t{0});

  detail::cuda_check(
    cudaMemcpyAsync(
      host_input.data(), d_input, count * sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()),
    "download");
  detail::cuda_check(cudaStreamSynchronize(stream.get()), "stream synchronize after download");

  auto const cast_mode = detail::analyze_data(host_input.data(), count).first;
  return encode_scalar32_page(host_input, count, padded, cast_mode, stream);
}

std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
encode_scalar32_pages_helper(std::vector<int32_t*> const& h_gather_ptrs,
                             std::vector<uint32_t> const& h_gather_counts,
                             cuda::stream_ref stream)
{
  size_t const num_pages = h_gather_ptrs.size();

  std::vector<rmm::device_buffer> encoded_buffers;
  std::vector<uint8_t*> h_upload_ptrs(num_pages, nullptr);
  std::vector<uint32_t> h_upload_sizes(num_pages, 0);
  encoded_buffers.reserve(num_pages);

  for (size_t i = 0; i < num_pages; ++i) {
    if (h_gather_ptrs[i] == nullptr) { continue; }

    uint32_t const count = h_gather_counts[i];
    if (count == 0) { continue; }

    auto page = encode_scalar32_page_helper(h_gather_ptrs[i], count, stream);

    encoded_buffers.push_back(std::move(page.device_blob));
    h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
    h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
  }

  return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
}

}  // namespace

EncodedPageResult FastLanesInt32Encoder::encode_page(int32_t const* d_input,
                                                     uint32_t count,
                                                     cuda::stream_ref stream)
{
  return encode_scalar32_page_helper(d_input, count, stream);
}

std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
FastLanesInt32Encoder::encode_pages(std::vector<int32_t*> const& h_gather_ptrs,
                                    std::vector<uint32_t> const& h_gather_counts,
                                    cuda::stream_ref stream)
{
  return encode_scalar32_pages_helper(h_gather_ptrs, h_gather_counts, stream);
}

}  // namespace cudf::io::parquet::detail::fastlanes
