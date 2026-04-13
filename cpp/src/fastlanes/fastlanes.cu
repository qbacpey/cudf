#include <cudf/fastlanes/debug.hpp>
#include <cudf/fastlanes/fastlanes_encode.cuh>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>
#include <cudf/fastlanes/native64_cuda.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf::io::parquet::detail::fastlanes_cudf {

namespace {

template <typename T>
using unsigned_t = std::make_unsigned_t<T>;

template <typename T>
inline constexpr uint8_t type_bits_v = sizeof(T) * 8;

template <typename T>
struct normalized_page_data {
  std::vector<unsigned_t<T>> values;
  uint8_t bitwidth;
  uint64_t min_value;
};

struct split32_page_data {
  std::vector<uint32_t> low_deltas;
  std::vector<uint32_t> high_deltas;
  uint32_t min_low_bits;
  uint32_t min_high_bits;
  uint8_t bitwidth_low;
  uint8_t bitwidth_high;
};

inline void cuda_check(cudaError_t err, char const* operation)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("FastLanesEncoder: CUDA ") + operation +
                             " failed: " + cudaGetErrorString(err));
  }
}

inline void upload_encoded_blob(EncodedPageResult& result, rmm::cuda_stream_view stream)
{
  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             result.host_blob.size(),
                             cudaMemcpyHostToDevice,
                             stream.value()),
             "upload");
}

template <typename T>
std::pair<fastlanes::TypeCastMode, bool> analyze_data(T const* data, uint32_t count)
{
  bool has_negative = false;
  for (uint32_t i = 0; i < count; ++i) {
    if (data[i] < 0) {
      has_negative = true;
      break;
    }
  }
  return {has_negative ? fastlanes::TypeCastMode::SIGNED_REINTERPRET
                       : fastlanes::TypeCastMode::SIGNED_SAFE,
          has_negative};
}

template <typename T>
uint8_t compute_bitwidth(unsigned_t<T> const* data, uint32_t count)
{
  // TODO: Consider merge all compute_bitwidth variants into one that takes max_val as argument to
  // reduce code duplication For the computation of max_val, in the CPU side just use
  // std::max_element for simplicity since it's a one-time linear scan and likely negligible in
  // latency compared to the rest of the encoding process. For GPU side, if we want to compute
  // max_val on GPU, we can use thrust::reduce with thrust::maximum to find the max value in a
  // stream-aware manner.
  unsigned_t<T> max_val = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (data[i] > max_val) { max_val = data[i]; }
  }

  if (max_val == 0) { return 1; }

  uint8_t bits = 0;
  while (max_val > 0) {
    max_val >>= 1;
    bits++;
  }
  return bits;
}

uint8_t compute_bitwidth_u32(uint32_t const* data, uint32_t count)
{
  uint32_t max_val = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (data[i] > max_val) { max_val = data[i]; }
  }

  if (max_val == 0) { return 1; }

  uint8_t bits = 0;
  while (max_val > 0) {
    max_val >>= 1;
    bits++;
  }
  return bits;
}

uint8_t compute_bitwidth_u64(uint64_t max_val)
{
  if (max_val == 0) { return 1; }

  uint8_t bits = 0;
  while (max_val > 0) {
    max_val >>= 1;
    bits++;
  }
  return bits;
}

template <typename T>
normalized_page_data<T> normalize_page_data(T const* data, uint32_t count, uint32_t padded_count)
{
  normalized_page_data<T> page{};
  page.values.assign(padded_count, unsigned_t<T>{0});

  auto min_val = std::numeric_limits<T>::max();
  for (uint32_t i = 0; i < count; ++i) {
    min_val = std::min(min_val, data[i]);
  }
  page.min_value = fastlanes::int64_to_u64_bits(static_cast<int64_t>(min_val));

  auto const min_bits = static_cast<unsigned_t<T>>(min_val);
  for (uint32_t i = 0; i < count; ++i) {
    auto const value_bits = static_cast<unsigned_t<T>>(data[i]);
    page.values[i]        = value_bits - min_bits;
  }

  page.bitwidth = compute_bitwidth<T>(page.values.data(), count);

  if (!fastlanes::is_valid_bitwidth(page.bitwidth)) {
    throw std::invalid_argument("FastLanesEncoder: normalized page produced invalid bitwidth");
  }

  return page;
}

split32_page_data normalize_split32_page_data(int64_t const* data,
                                              uint32_t count,
                                              uint32_t padded_count)
{
  split32_page_data page{};
  page.low_deltas.assign(padded_count, uint32_t{0});
  page.high_deltas.assign(padded_count, uint32_t{0});

  if (count == 0) {
    page.min_low_bits  = 0;
    page.min_high_bits = 0;
    page.bitwidth_low  = 1;
    page.bitwidth_high = 1;
    return page;
  }

  uint32_t min_low  = std::numeric_limits<uint32_t>::max();
  uint32_t min_high = std::numeric_limits<uint32_t>::max();

  for (uint32_t i = 0; i < count; ++i) {
    auto const bits = fastlanes::int64_to_u64_bits(data[i]);
    auto const low  = static_cast<uint32_t>(bits);
    auto const high = static_cast<uint32_t>(bits >> 32);
    min_low         = std::min(min_low, low);
    min_high        = std::min(min_high, high);
  }

  page.min_low_bits  = min_low;
  page.min_high_bits = min_high;

  for (uint32_t i = 0; i < count; ++i) {
    auto const bits     = fastlanes::int64_to_u64_bits(data[i]);
    auto const low      = static_cast<uint32_t>(bits);
    auto const high     = static_cast<uint32_t>(bits >> 32);
    page.low_deltas[i]  = low - min_low;
    page.high_deltas[i] = high - min_high;
  }

  page.bitwidth_low  = compute_bitwidth_u32(page.low_deltas.data(), count);
  page.bitwidth_high = compute_bitwidth_u32(page.high_deltas.data(), count);

  if (!fastlanes::is_valid_bitwidth(page.bitwidth_low) ||
      !fastlanes::is_valid_bitwidth(page.bitwidth_high)) {
    throw std::invalid_argument(
      "FastLanesEncoder<int64_t>: split32 normalization produced invalid bitwidth");
  }

  return page;
}

template <typename T>
void encode_vectors(unsigned_t<T> const* input,
                    uint64_t padded,
                    uint8_t bitwidth,
                    unsigned_t<T>* output)
{
  // TODO: Consider merge encode_vectors and encode_vectors_u32 into one function to reduce code
  // duplication. Since now we can assume we would only call
  // `generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bitwidth);` for only uint32_t type.
  // Or even just take this piece of code into the caller since it's just one line and the rest is
  // vector loop scaffolding.
  uint64_t const n_vectors = fastlanes::num_vectors(padded);
  auto const* in_ptr       = input;
  auto* out_ptr            = output;

  size_t const out_elements_per_vector = (fastlanes::VECTOR_SIZE * bitwidth) / type_bits_v<T>;

  for (uint64_t v = 0; v < n_vectors; ++v) {
    generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bitwidth);
    in_ptr += fastlanes::VECTOR_SIZE;
    out_ptr += out_elements_per_vector;
  }
}

void encode_vectors_u32(uint32_t const* input, uint64_t padded, uint8_t bitwidth, uint32_t* output)
{
  uint64_t const n_vectors = fastlanes::num_vectors(padded);
  auto const* in_ptr       = input;
  auto* out_ptr            = output;

  size_t const out_elements_per_vector = (fastlanes::VECTOR_SIZE * bitwidth) / 32;

  for (uint64_t v = 0; v < n_vectors; ++v) {
    generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bitwidth);
    in_ptr += fastlanes::VECTOR_SIZE;
    out_ptr += out_elements_per_vector;
  }
}

template <typename T>
EncodedPageResult encode_scalar32_page(std::vector<T> const& host_input,
                                       uint32_t count,
                                       uint64_t padded,
                                       fastlanes::TypeCastMode cast_mode,
                                       rmm::cuda_stream_view stream,
                                       bool debug_print)
{
  EncodedPageResult result;

  auto normalized = normalize_page_data<T>(host_input.data(), count, static_cast<uint32_t>(padded));
  auto const bitwidth = normalized.bitwidth;

  size_t const encoded_bytes = fastlanes::encoded_size_bytes(padded, bitwidth);
  if (encoded_bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("FastLanesEncoder: encoded body size exceeds 32-bit header limit");
  }
  auto const encoded_bytes_u32 = static_cast<uint32_t>(encoded_bytes);
  std::vector<unsigned_t<T>> encoded_body(encoded_bytes / sizeof(unsigned_t<T>));

  encode_vectors<T>(normalized.values.data(), padded, bitwidth, encoded_body.data());

  if (debug_print) {
    using namespace fastlanes::debug;
    PageDebugInfo info = make_debug_info(bitwidth,
                                         cast_mode,
                                         count,
                                         static_cast<uint32_t>(padded),
                                         encoded_bytes_u32,
                                         normalized.min_value,
                                         encoded_body.data(),
                                         encoded_body.size());
    print_page_debug(std::cout, info);
  }

  result.host_blob =
    fastlanes::PageHeader::serialize_scalar32(bitwidth,
                                              count,
                                              static_cast<uint32_t>(padded),
                                              static_cast<uint32_t>(normalized.min_value),
                                              reinterpret_cast<uint8_t const*>(encoded_body.data()),
                                              encoded_bytes_u32,
                                              fastlanes::default_pre_delta_for_mode(false));
  upload_encoded_blob(result, stream);

  result.bitwidth       = bitwidth;
  result.cast_mode      = cast_mode;
  result.original_count = count;
  result.padded_count   = static_cast<uint32_t>(padded);
  result.body_size      = encoded_bytes_u32;
  result.min_value      = normalized.min_value;

  return result;
}

EncodedPageResult encode_split32_page(std::vector<int64_t> const& host_input,
                                      uint32_t count,
                                      uint64_t padded,
                                      fastlanes::TypeCastMode cast_mode,
                                      rmm::cuda_stream_view stream)
{
  EncodedPageResult result;
  auto split = normalize_split32_page_data(host_input.data(), count, static_cast<uint32_t>(padded));

  size_t const encoded_bytes_low  = fastlanes::encoded_size_bytes(padded, split.bitwidth_low);
  size_t const encoded_bytes_high = fastlanes::encoded_size_bytes(padded, split.bitwidth_high);
  size_t const encoded_body_bytes = encoded_bytes_low + encoded_bytes_high;
  if (encoded_body_bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("FastLanesEncoder: split32 body size exceeds 32-bit header limit");
  }

  std::vector<uint32_t> encoded_low(encoded_bytes_low / sizeof(uint32_t));
  std::vector<uint32_t> encoded_high(encoded_bytes_high / sizeof(uint32_t));

  encode_vectors_u32(split.low_deltas.data(), padded, split.bitwidth_low, encoded_low.data());
  encode_vectors_u32(split.high_deltas.data(), padded, split.bitwidth_high, encoded_high.data());

  std::vector<uint8_t> encoded_body(encoded_body_bytes);
  if (encoded_bytes_low > 0) {
    std::memcpy(encoded_body.data(), encoded_low.data(), encoded_bytes_low);
  }
  if (encoded_bytes_high > 0) {
    std::memcpy(encoded_body.data() + encoded_bytes_low, encoded_high.data(), encoded_bytes_high);
  }

  auto const encoded_body_size_u32 = static_cast<uint32_t>(encoded_body.size());

  result.host_blob =
    fastlanes::PageHeader::serialize_split32(split.bitwidth_low,
                                             split.bitwidth_high,
                                             count,
                                             static_cast<uint32_t>(padded),
                                             split.min_low_bits,
                                             split.min_high_bits,
                                             encoded_body.data(),
                                             encoded_body_size_u32,
                                             fastlanes::default_pre_delta_for_mode(true));
  upload_encoded_blob(result, stream);

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

/**
 * @brief Build an empty RAW32-encoded page result.
 */
EncodedPageResult create_empty_scalar32_result(rmm::cuda_stream_view stream)
{
  // TODO: Consider merge all the create_empty_*_result functions into one that takes an encoding
  // type enum to reduce code duplication.
  EncodedPageResult result;
  result.bitwidth       = 1;
  result.cast_mode      = fastlanes::TypeCastMode::SIGNED_SAFE;
  result.original_count = 0;
  result.padded_count   = 0;
  result.body_size      = 0;
  result.min_value      = 0;

  result.host_blob = fastlanes::PageHeader::serialize_scalar32(
    1, 0, 0, 0, nullptr, 0, fastlanes::default_pre_delta_for_mode(false));

  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             result.host_blob.size(),
                             cudaMemcpyHostToDevice,
                             stream.value()),
             "upload empty blob");

  return result;
}

/**
 * @brief Build an empty SPLIT64-encoded page result.
 */
EncodedPageResult create_empty_split32_result(rmm::cuda_stream_view stream)
{
  EncodedPageResult result;
  result.bitwidth       = 1;
  result.cast_mode      = fastlanes::TypeCastMode::SIGNED_SAFE;
  result.original_count = 0;
  result.padded_count   = 0;
  result.body_size      = 0;
  result.min_value      = 0;

  result.host_blob = fastlanes::PageHeader::serialize_split32(
    1, 1, 0, 0, 0, 0, nullptr, 0, fastlanes::default_pre_delta_for_mode(true));

  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             result.host_blob.size(),
                             cudaMemcpyHostToDevice,
                             stream.value()),
             "upload empty blob");

  return result;
}

/**
 * @brief Build an empty NATIVE64-encoded page result.
 */
EncodedPageResult create_empty_native64_result(rmm::cuda_stream_view stream)
{
  EncodedPageResult result;
  result.bitwidth       = 1;
  result.cast_mode      = fastlanes::TypeCastMode::SIGNED_SAFE;
  result.original_count = 0;
  result.padded_count   = 0;
  result.body_size      = 0;
  result.min_value      = 0;

  result.host_blob = fastlanes::PageHeader::serialize_native64(
    1, 0, 0, 0, nullptr, 0, fastlanes::default_pre_delta_for_mode(false));

  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             result.host_blob.size(),
                             cudaMemcpyHostToDevice,
                             stream.value()),
             "upload empty blob");

  return result;
}

/**
 * @brief Encode one INT32 page via the explicit RAW32 helper path.
 */
EncodedPageResult encode_scalar32_page_helper(int32_t const* d_input,
                                              uint32_t count,
                                              rmm::cuda_stream_view stream,
                                              bool debug_print)
{
  if (count == 0) { return create_empty_scalar32_result(stream); }

  if (d_input == nullptr) {
    throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
  }

  uint64_t const padded = fastlanes::padded_count(count);
  std::vector<int32_t> host_input(padded, int32_t{0});

  cuda_check(
    cudaMemcpyAsync(
      host_input.data(), d_input, count * sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()),
    "download");
  cuda_check(cudaStreamSynchronize(stream.value()), "stream synchronize after download");

  auto const cast_mode = analyze_data(host_input.data(), count).first;
  return encode_scalar32_page(host_input, count, padded, cast_mode, stream, debug_print);
}

/**
 * @brief Encode one INT64 page via the explicit SPLIT64 helper path.
 */
EncodedPageResult encode_split32_page_helper(int64_t const* d_input,
                                             uint32_t count,
                                             rmm::cuda_stream_view stream,
                                             bool debug_print)
{
  static_cast<void>(debug_print);

  if (count == 0) { return create_empty_split32_result(stream); }

  if (d_input == nullptr) {
    throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
  }

  uint64_t const padded = fastlanes::padded_count(count);
  std::vector<int64_t> host_input(padded, int64_t{0});

  cuda_check(
    cudaMemcpyAsync(
      host_input.data(), d_input, count * sizeof(int64_t), cudaMemcpyDeviceToHost, stream.value()),
    "download");
  cuda_check(cudaStreamSynchronize(stream.value()), "stream synchronize after download");

  auto const cast_mode = analyze_data(host_input.data(), count).first;
  return encode_split32_page(host_input, count, padded, cast_mode, stream);
}

/**
 * @brief Encode one INT64 page via the explicit Native64 helper path.
 */
EncodedPageResult encode_native64_page_helper(int64_t const* d_input,
                                              uint32_t count,
                                              rmm::cuda_stream_view stream,
                                              bool debug_print)
{
  static_cast<void>(debug_print);

  if (count == 0) { return create_empty_native64_result(stream); }

  if (d_input == nullptr) {
    throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
  }

  auto const* d_input_u64 = reinterpret_cast<uint64_t const*>(d_input);
  auto const min_value_bits =
    native64_generated::derive_min_base_bits(d_input_u64, count, stream.value());

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

  auto const bitwidth = compute_bitwidth_u64(max_delta);

  auto const padded_count      = static_cast<uint32_t>(fastlanes::padded_count(count));
  auto const encoded_body_size = fastlanes::encoded_size_bytes(padded_count, bitwidth);
  if (encoded_body_size > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("FastLanesEncoder: native64 body size exceeds 32-bit header limit");
  }

  rmm::device_uvector<uint64_t> padded_input(padded_count, stream);
  // Filling with min_value_bits to ensure that padding does not introduce larger deltas that could
  // affect bitwidth calculation in encode_by_bw_gpu_device_ptrs.
  thrust::fill(rmm::exec_policy(stream), padded_input.begin(), padded_input.end(), min_value_bits);
  cuda_check(cudaMemcpyAsync(padded_input.data(),
                             d_input_u64,
                             count * sizeof(uint64_t),
                             cudaMemcpyDeviceToDevice,
                             stream.value()),
             "copy native64 input");

  EncodedPageResult result;
  result.host_blob =
    fastlanes::PageHeader::serialize_native64(bitwidth,
                                              count,
                                              padded_count,
                                              min_value_bits,
                                              nullptr,
                                              encoded_body_size,
                                              fastlanes::default_pre_delta_for_mode(false));
  // TODO: In this place using host_blob as staging for header bytes is a bit awkward since the body
  // encoding is GPU-only. Consider refactor into host_header_block & host_body_block for clearer
  // separation of host vs device responsibilities.
  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);

  auto const header_size = fastlanes::PageHeader::header_size();
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             header_size,
                             cudaMemcpyHostToDevice,
                             stream.value()),
             "upload native64 header");

  auto* payload_device_ptr = reinterpret_cast<uint64_t*>(
    fastlanes::PageHeader::payload_ptr(static_cast<uint8_t*>(result.device_blob.data())));
  native64_generated::encode_by_bw_gpu_device_ptrs(
    bitwidth, padded_input.data(), payload_device_ptr, min_value_bits, count, stream.value());

  result.bitwidth       = bitwidth;
  result.cast_mode      = fastlanes::TypeCastMode::SIGNED_SAFE;
  result.original_count = count;
  result.padded_count   = padded_count;
  result.body_size      = encoded_body_size;
  result.min_value      = min_value_bits;

  return result;
}

/**
 * @brief Encode a batch of INT32 pages via the explicit RAW32 helper path.
 */
std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
encode_scalar32_pages_helper(std::vector<int32_t*> const& h_gather_ptrs,
                             std::vector<uint32_t> const& h_gather_counts,
                             rmm::cuda_stream_view stream,
                             bool debug_print)
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

    auto page = encode_scalar32_page_helper(h_gather_ptrs[i], count, stream, debug_print);

    encoded_buffers.push_back(std::move(page.device_blob));
    h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
    h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
  }

  return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
}

/**
 * @brief Encode a batch of INT64 pages via the explicit SPLIT64 helper path.
 */
std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
encode_split32_pages_helper(std::vector<int64_t*> const& h_gather_ptrs,
                            std::vector<uint32_t> const& h_gather_counts,
                            rmm::cuda_stream_view stream,
                            bool debug_print)
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

    auto page = encode_split32_page_helper(h_gather_ptrs[i], count, stream, debug_print);

    encoded_buffers.push_back(std::move(page.device_blob));
    h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
    h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
  }

  return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
}

/**
 * @brief Encode a batch of INT64 pages via the explicit Native64 helper path.
 */
std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
encode_native64_pages_helper(std::vector<int64_t*> const& h_gather_ptrs,
                             std::vector<uint32_t> const& h_gather_counts,
                             rmm::cuda_stream_view stream,
                             bool debug_print)
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

    auto page = encode_native64_page_helper(h_gather_ptrs[i], count, stream, debug_print);

    encoded_buffers.push_back(std::move(page.device_blob));
    h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
    h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
  }

  return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
}

}  // namespace

/**
 * @brief Construct the explicit INT32 RAW32 FastLanes encoder.
 */
FastLanesInt32Encoder::FastLanesInt32Encoder(bool debug_print) : debug_print_(debug_print) {}

/**
 * @brief Encode a single INT32 page via the explicit RAW32 helper path.
 */
EncodedPageResult FastLanesInt32Encoder::encode_page(int32_t const* d_input,
                                                     uint32_t count,
                                                     rmm::cuda_stream_view stream)
{ return encode_scalar32_page_helper(d_input, count, stream, debug_print_); }

/**
 * @brief Encode batched INT32 pages via the explicit RAW32 helper path.
 */
std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
FastLanesInt32Encoder::encode_pages(std::vector<int32_t*> const& h_gather_ptrs,
                                    std::vector<uint32_t> const& h_gather_counts,
                                    rmm::cuda_stream_view stream)
{ return encode_scalar32_pages_helper(h_gather_ptrs, h_gather_counts, stream, debug_print_); }

/**
 * @brief Construct the explicit INT64 SPLIT64 FastLanes encoder.
 */
FastLanesInt64Split32Encoder::FastLanesInt64Split32Encoder(bool debug_print)
  : debug_print_(debug_print)
{
}

/**
 * @brief Encode a single INT64 page via the explicit SPLIT64 helper path.
 */
EncodedPageResult FastLanesInt64Split32Encoder::encode_page(int64_t const* d_input,
                                                            uint32_t count,
                                                            rmm::cuda_stream_view stream)
{ return encode_split32_page_helper(d_input, count, stream, debug_print_); }

/**
 * @brief Encode batched INT64 pages via the explicit SPLIT64 helper path.
 */
std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
FastLanesInt64Split32Encoder::encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
                                           std::vector<uint32_t> const& h_gather_counts,
                                           rmm::cuda_stream_view stream)
{ return encode_split32_pages_helper(h_gather_ptrs, h_gather_counts, stream, debug_print_); }

/**
 * @brief Construct the explicit INT64 Native64 FastLanes encoder.
 */
FastLanesInt64NativeEncoder::FastLanesInt64NativeEncoder(bool debug_print)
  : debug_print_(debug_print)
{
}

/**
 * @brief Encode a single INT64 page via the explicit Native64 helper path.
 */
EncodedPageResult FastLanesInt64NativeEncoder::encode_page(int64_t const* d_input,
                                                           uint32_t count,
                                                           rmm::cuda_stream_view stream)
{ return encode_native64_page_helper(d_input, count, stream, debug_print_); }

/**
 * @brief Encode batched INT64 pages via the explicit Native64 helper path.
 */
std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
FastLanesInt64NativeEncoder::encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
                                          std::vector<uint32_t> const& h_gather_counts,
                                          rmm::cuda_stream_view stream)
{ return encode_native64_pages_helper(h_gather_ptrs, h_gather_counts, stream, debug_print_); }

}  // namespace cudf::io::parquet::detail::fastlanes_cudf
