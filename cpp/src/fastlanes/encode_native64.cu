#include "encode_common.hpp"

#include <cudf/fastlanes/native64_host.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstring>
#include <limits>

namespace cudf::io::parquet::detail::fastlanes {

namespace {

namespace native64_encoder = cudf::io::parquet::detail::fastlanes::native64;

EncodedPageResult encode_native64_page_helper(int64_t const* d_input,
                                              uint32_t count,
                                              rmm::cuda_stream_view stream)
{
  if (count == 0) { return detail::create_empty_native64_result(stream); }

  if (d_input == nullptr) {
    throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
  }

  auto const* d_input_u64 = reinterpret_cast<uint64_t const*>(d_input);
  auto const min_value_bits =
    native64_encoder::derive_min_base_bits(d_input_u64, count, stream.value());

  auto const idx_begin = thrust::make_counting_iterator<uint32_t>(0);
  auto const idx_end   = idx_begin + count;
  auto const max_delta = thrust::transform_reduce(
    rmm::exec_policy(stream),
    idx_begin,
    idx_end,
    [d_input_u64, min_value_bits] __device__(uint32_t idx) -> uint64_t {
      return d_input_u64[idx] - min_value_bits;
    },
    uint64_t{0},
    thrust::maximum<uint64_t>{});

  auto const has_negative = thrust::transform_reduce(
    rmm::exec_policy(stream),
    idx_begin,
    idx_end,
    [d_input_u64] __device__(uint32_t idx) -> bool {
      return (d_input_u64[idx] & (uint64_t{1} << 63)) != 0;
    },
    false,
    thrust::logical_or<bool>{});
  auto const cast_mode = has_negative ? ::fastlanes::TypeCastMode::SIGNED_REINTERPRET
                                      : ::fastlanes::TypeCastMode::SIGNED_SAFE;

  auto const bitwidth = detail::compute_bitwidth_u64(max_delta);

  auto const padded_count      = static_cast<uint32_t>(::fastlanes::padded_count(count));
  auto const encoded_body_size = ::fastlanes::encoded_size_bytes(padded_count, bitwidth);
  if (encoded_body_size > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("FastLanesEncoder: native64 body size exceeds 32-bit header limit");
  }

  rmm::device_uvector<uint64_t> padded_input(padded_count, stream);
  thrust::fill(rmm::exec_policy(stream), padded_input.begin(), padded_input.end(), min_value_bits);
  detail::cuda_check(cudaMemcpyAsync(padded_input.data(),
                                     d_input_u64,
                                     count * sizeof(uint64_t),
                                     cudaMemcpyDeviceToDevice,
                                     stream.value()),
                     "copy native64 input");

  EncodedPageResult result;
  auto const serialized_layout = ::fastlanes::PageHeader::serialize_native64(
    bitwidth,
    count,
    padded_count,
    min_value_bits,
    nullptr,
    encoded_body_size,
    ::fastlanes::default_pre_delta_for_native64());

  auto const header_size = ::fastlanes::PageHeader::header_size();
  result.host_blob.assign(serialized_layout.size(), uint8_t{0});
  std::memcpy(result.host_blob.data(), serialized_layout.data(), header_size);

  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  detail::cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                                     serialized_layout.data(),
                                     header_size,
                                     cudaMemcpyHostToDevice,
                                     stream.value()),
                     "upload native64 header");

  auto* payload_device_ptr = reinterpret_cast<uint64_t*>(
    ::fastlanes::PageHeader::payload_ptr(static_cast<uint8_t*>(result.device_blob.data())));
  native64_encoder::launch_native64_encode(
    bitwidth, padded_input.data(), payload_device_ptr, min_value_bits, count, stream.value());

  result.bitwidth       = bitwidth;
  result.cast_mode      = cast_mode;
  result.original_count = count;
  result.padded_count   = padded_count;
  result.body_size      = encoded_body_size;
  result.min_value      = min_value_bits;

  return result;
}

std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
encode_native64_pages_helper(std::vector<int64_t*> const& h_gather_ptrs,
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

    auto page = encode_native64_page_helper(h_gather_ptrs[i], count, stream);

    encoded_buffers.push_back(std::move(page.device_blob));
    h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
    h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
  }

  return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
}

}  // namespace

EncodedPageResult FastLanesInt64NativeEncoder::encode_page(int64_t const* d_input,
                                                           uint32_t count,
                                                           rmm::cuda_stream_view stream)
{
  return encode_native64_page_helper(d_input, count, stream);
}

std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
FastLanesInt64NativeEncoder::encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
                                          std::vector<uint32_t> const& h_gather_counts,
                                          rmm::cuda_stream_view stream)
{
  return encode_native64_pages_helper(h_gather_ptrs, h_gather_counts, stream);
}

}  // namespace cudf::io::parquet::detail::fastlanes
