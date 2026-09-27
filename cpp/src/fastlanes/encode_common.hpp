#pragma once

#include <cudf/fastlanes/fastlanes_encode.cuh>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>
#include <cudf/io/parquet_schema.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf::io::parquet::detail::fastlanes::detail {

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

inline void upload_encoded_blob(EncodedPageResult& result, cuda::stream_ref stream)
{
  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             result.host_blob.size(),
                             cudaMemcpyHostToDevice,
                             stream.get()),
             "upload");
}

template <typename T>
std::pair<::fastlanes::TypeCastMode, bool> analyze_data(T const* data, uint32_t count)
{
  bool has_negative = false;
  for (uint32_t i = 0; i < count; ++i) {
    if (data[i] < 0) {
      has_negative = true;
      break;
    }
  }
  return {has_negative ? ::fastlanes::TypeCastMode::SIGNED_REINTERPRET
                       : ::fastlanes::TypeCastMode::SIGNED_SAFE,
          has_negative};
}

template <typename T>
uint8_t compute_bitwidth(unsigned_t<T> const* data, uint32_t count)
{
  auto max_val = unsigned_t<T>{0};
  for (uint32_t i = 0; i < count; ++i) {
    if (data[i] > max_val) { max_val = data[i]; }
  }

  uint8_t bits = 0;
  do {
    ++bits;
    max_val >>= 1;
  } while (max_val != 0);
  return bits;
}

inline uint8_t compute_bitwidth_u64(uint64_t max_val)
{
  uint8_t bits = 0;
  do {
    ++bits;
    max_val >>= 1;
  } while (max_val != 0);
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
  page.min_value = ::fastlanes::int64_to_u64_bits(static_cast<int64_t>(min_val));

  auto const min_bits = static_cast<unsigned_t<T>>(min_val);
  for (uint32_t i = 0; i < count; ++i) {
    auto const value_bits = static_cast<unsigned_t<T>>(data[i]);
    page.values[i]        = value_bits - min_bits;
  }

  page.bitwidth = compute_bitwidth<T>(page.values.data(), count);

  if (!::fastlanes::is_valid_bitwidth(page.bitwidth)) {
    throw std::invalid_argument("FastLanesEncoder: normalized page produced invalid bitwidth");
  }

  return page;
}

inline split32_page_data normalize_split32_page_data(int64_t const* data,
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
    auto const bits = ::fastlanes::int64_to_u64_bits(data[i]);
    auto const low  = static_cast<uint32_t>(bits);
    auto const high = static_cast<uint32_t>(bits >> 32);
    min_low         = std::min(min_low, low);
    min_high        = std::min(min_high, high);
  }

  page.min_low_bits  = min_low;
  page.min_high_bits = min_high;

  for (uint32_t i = 0; i < count; ++i) {
    auto const bits     = ::fastlanes::int64_to_u64_bits(data[i]);
    auto const low      = static_cast<uint32_t>(bits);
    auto const high     = static_cast<uint32_t>(bits >> 32);
    page.low_deltas[i]  = low - min_low;
    page.high_deltas[i] = high - min_high;
  }

  page.bitwidth_low  = compute_bitwidth<uint32_t>(page.low_deltas.data(), count);
  page.bitwidth_high = compute_bitwidth<uint32_t>(page.high_deltas.data(), count);

  if (!::fastlanes::is_valid_bitwidth(page.bitwidth_low) ||
      !::fastlanes::is_valid_bitwidth(page.bitwidth_high)) {
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
  uint64_t const n_vectors = ::fastlanes::num_vectors(padded);
  auto const* in_ptr       = input;
  auto* out_ptr            = output;

  size_t const out_elements_per_vector = (::fastlanes::VECTOR_SIZE * bitwidth) / type_bits_v<T>;

  for (uint64_t v = 0; v < n_vectors; ++v) {
    generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bitwidth);
    in_ptr += ::fastlanes::VECTOR_SIZE;
    out_ptr += out_elements_per_vector;
  }
}

EncodedPageResult create_empty_result(Encoding encoding, cuda::stream_ref stream);
EncodedPageResult create_empty_scalar32_result(cuda::stream_ref stream);
EncodedPageResult create_empty_split32_result(cuda::stream_ref stream);
EncodedPageResult create_empty_native64_result(cuda::stream_ref stream);

}  // namespace cudf::io::parquet::detail::fastlanes::detail
