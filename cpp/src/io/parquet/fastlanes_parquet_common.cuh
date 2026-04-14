/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/fastlanes/common.cuh>

namespace cudf::io::parquet::detail {

CUDF_HOST_DEVICE constexpr bool is_fastlanes_bitpack_raw_supported_logical(cudf::type_id logical_type)
{
  switch (logical_type) {
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8:
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DURATION_MILLISECONDS: return true;
    default: return false;
  }
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_bitpack_split64_supported_logical(
  cudf::type_id logical_type)
{
  return logical_type == cudf::type_id::INT64 || logical_type == cudf::type_id::UINT64;
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_flat_column(int32_t max_rep_level)
{
  return max_rep_level == 0;
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_bitpack_raw_runtime_supported(
  Type physical_type,
  cudf::type_id logical_type,
  int32_t max_rep_level)
{
  return is_fastlanes_flat_column(max_rep_level) &&
         physical_type == Type::INT32 &&
         is_fastlanes_bitpack_raw_supported_logical(logical_type);
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_bitpack_split64_runtime_supported(
  Type physical_type,
  cudf::type_id logical_type,
  int32_t max_rep_level)
{
  return is_fastlanes_flat_column(max_rep_level) &&
         physical_type == Type::INT64 &&
         is_fastlanes_bitpack_split64_supported_logical(logical_type);
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_delta_binary_runtime_supported(
  Type physical_type,
  cudf::type_id logical_type,
  int32_t max_rep_level)
{
  return is_fastlanes_flat_column(max_rep_level) &&
         physical_type == Type::INT64 &&
         is_fastlanes_bitpack_split64_supported_logical(logical_type);
}

/**
 * @brief Returns true when FASTLANES_DELTA_BINARY encode routing is activated.
 */
CUDF_HOST_DEVICE constexpr bool is_fastlanes_delta_binary_encode_enabled() { return true; }

CUDF_HOST_DEVICE constexpr uint32_t fastlanes_bitpack_kernel_masks()
{
  return BitOr(
    encode_kernel_mask::FASTLANE_BITPACK_RAW, encode_kernel_mask::FASTLANE_BITPACK_SPLIT64);
}

CUDF_HOST_DEVICE constexpr uint32_t fastlanes_kernel_masks()
{
  return BitOr(fastlanes_bitpack_kernel_masks(), encode_kernel_mask::FASTLANES_DELTA_BINARY);
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_bitpack_mask(encode_kernel_mask kernel_mask)
{
  return BitAnd(kernel_mask, fastlanes_bitpack_kernel_masks()) != 0;
}

CUDF_HOST_DEVICE constexpr bool is_fastlanes_mask(encode_kernel_mask kernel_mask)
{
  return BitAnd(kernel_mask, fastlanes_kernel_masks()) != 0;
}

CUDF_HOST_DEVICE constexpr Encoding fastlanes_encoding_for_mask(encode_kernel_mask kernel_mask)
{
  if (BitAnd(kernel_mask, encode_kernel_mask::FASTLANES_DELTA_BINARY) != 0) {
    return Encoding::FASTLANES_DELTA_BINARY;
  }
  if (BitAnd(kernel_mask, encode_kernel_mask::FASTLANE_BITPACK_SPLIT64) != 0) {
    return Encoding::FASTLANE_BITPACK_SPLIT64;
  }
  return Encoding::FASTLANE_BITPACK_RAW;
}

CUDF_HOST_DEVICE constexpr size_t fastlanes_component_streams_for_mask(
  encode_kernel_mask kernel_mask,
  Type physical_type)
{
  if (kernel_mask == encode_kernel_mask::FASTLANE_BITPACK_SPLIT64) { return size_t{2}; }

  if (kernel_mask == encode_kernel_mask::FASTLANE_BITPACK_RAW) {
    // Defensive fallback: avoid under-allocation if an unexpected INT64 raw request slips through.
    return physical_type == Type::INT64 ? size_t{2} : size_t{1};
  }

  if (kernel_mask == encode_kernel_mask::FASTLANES_DELTA_BINARY) {
    // Native64 payloads are single-stream 64-bit component data.
    return size_t{1};
  }

  return size_t{0};
}

__device__ inline bool fastlanes_set_decode_error(int lane,
                                                  kernel_error::pointer error_code,
                                                  decode_error code)
{
  if (lane == 0) { set_error(static_cast<kernel_error::value_type>(code), error_code); }
  return false;
}

__device__ inline bool fastlanes_is_supported_dtype_len(uint32_t dtype_len)
{
  return dtype_len == 1 || dtype_len == 2 || dtype_len == 4 || dtype_len == 8;
}

__device__ __host__ inline constexpr bool fastlanes_is_valid_header_for_encoding(
  fastlanes::PageHeader const& header,
  Encoding encoding)
{
  switch (encoding) {
    case Encoding::FASTLANE_BITPACK_RAW: return fastlanes::is_valid_raw32_header(header);
    case Encoding::FASTLANE_BITPACK_SPLIT64: return fastlanes::is_valid_split64_header(header);
    case Encoding::FASTLANES_DELTA_BINARY: return fastlanes::is_valid_native64_header(header);
    default: return false;
  }
}

}  // namespace cudf::io::parquet::detail
