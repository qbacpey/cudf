#include "encode_common.hpp"

#include <cstring>

namespace cudf::io::parquet::detail::fastlanes {

namespace {

EncodedPageResult encode_split32_page(std::vector<int64_t> const& host_input,
                                      uint32_t count,
                                      uint64_t padded,
                                      ::fastlanes::TypeCastMode cast_mode,
                                      rmm::cuda_stream_view stream)
{
  EncodedPageResult result;
  auto split = detail::normalize_split32_page_data(
    host_input.data(), count, static_cast<uint32_t>(padded));

  size_t const encoded_bytes_low  = ::fastlanes::encoded_size_bytes(padded, split.bitwidth_low);
  size_t const encoded_bytes_high = ::fastlanes::encoded_size_bytes(padded, split.bitwidth_high);
  size_t const encoded_body_bytes = encoded_bytes_low + encoded_bytes_high;
  if (encoded_body_bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("FastLanesEncoder: split32 body size exceeds 32-bit header limit");
  }

  std::vector<uint32_t> encoded_low(encoded_bytes_low / sizeof(uint32_t));
  std::vector<uint32_t> encoded_high(encoded_bytes_high / sizeof(uint32_t));

  detail::encode_vectors<uint32_t>(
    split.low_deltas.data(), padded, split.bitwidth_low, encoded_low.data());
  detail::encode_vectors<uint32_t>(
    split.high_deltas.data(), padded, split.bitwidth_high, encoded_high.data());

  std::vector<uint8_t> encoded_body(encoded_body_bytes);
  if (encoded_bytes_low > 0) {
    std::memcpy(encoded_body.data(), encoded_low.data(), encoded_bytes_low);
  }
  if (encoded_bytes_high > 0) {
    std::memcpy(encoded_body.data() + encoded_bytes_low, encoded_high.data(), encoded_bytes_high);
  }

  auto const encoded_body_size_u32 = static_cast<uint32_t>(encoded_body.size());

  result.host_blob = ::fastlanes::PageHeader::serialize_split32(
    split.bitwidth_low,
    split.bitwidth_high,
    count,
    static_cast<uint32_t>(padded),
    split.min_low_bits,
    split.min_high_bits,
    encoded_body.data(),
    encoded_body_size_u32,
    ::fastlanes::default_pre_delta_for_split64());
  detail::upload_encoded_blob(result, stream);

  result.bitwidth =
    split.bitwidth_low > split.bitwidth_high ? split.bitwidth_low : split.bitwidth_high;
  result.cast_mode      = cast_mode;
  result.original_count = count;
  result.padded_count   = static_cast<uint32_t>(padded);
  result.body_size      = encoded_body_size_u32;
  result.min_value =
    (static_cast<uint64_t>(split.min_high_bits) << 32) | static_cast<uint64_t>(split.min_low_bits);

  return result;
}

EncodedPageResult encode_split32_page_helper(int64_t const* d_input,
                                             uint32_t count,
                                             rmm::cuda_stream_view stream)
{
  if (count == 0) { return detail::create_empty_split32_result(stream); }

  if (d_input == nullptr) {
    throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
  }

  uint64_t const padded = ::fastlanes::padded_count(count);
  std::vector<int64_t> host_input(padded, int64_t{0});

  detail::cuda_check(
    cudaMemcpyAsync(
      host_input.data(), d_input, count * sizeof(int64_t), cudaMemcpyDeviceToHost, stream.value()),
    "download");
  detail::cuda_check(cudaStreamSynchronize(stream.value()), "stream synchronize after download");

  auto const cast_mode = detail::analyze_data(host_input.data(), count).first;
  return encode_split32_page(host_input, count, padded, cast_mode, stream);
}

std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
encode_split32_pages_helper(std::vector<int64_t*> const& h_gather_ptrs,
                            std::vector<uint32_t> const& h_gather_counts,
                            rmm::cuda_stream_view stream)
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

    auto page = encode_split32_page_helper(h_gather_ptrs[i], count, stream);

    encoded_buffers.push_back(std::move(page.device_blob));
    h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
    h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
  }

  return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
}

}  // namespace

EncodedPageResult FastLanesInt64Split32Encoder::encode_page(int64_t const* d_input,
                                                            uint32_t count,
                                                            rmm::cuda_stream_view stream)
{
  return encode_split32_page_helper(d_input, count, stream);
}

std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
FastLanesInt64Split32Encoder::encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
                                           std::vector<uint32_t> const& h_gather_counts,
                                           rmm::cuda_stream_view stream)
{
  return encode_split32_pages_helper(h_gather_ptrs, h_gather_counts, stream);
}

}  // namespace cudf::io::parquet::detail::fastlanes
