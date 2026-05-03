/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cudf::io::parquet::detail::fastlanes::native64 {
/**
 * @brief Derive the minimum base bits required to encode the given values with the FastLanes Native64 encoding strategy.
 * @param values Pointer to the input values on the GPU
 * @param total_count Total number of values
 * @param stream CUDA stream to use for the operation
 * @return Minimum base bits required for encoding
 */
uint64_t derive_min_base_bits(uint64_t const* values,
                              uint32_t total_count,
                              cudaStream_t stream);
/**
 * @brief Encode the given values using the FastLanes Native64 encoding strategy on the GPU.
 * @param bw Bit width to use for encoding
 * @param values Pointer to the input values on the GPU
 * @param packed Pointer to the output buffer for the packed values on the GPU
 * @param base_bits Base bits to use for encoding
 * @param total_count Total number of values
 * @param stream CUDA stream to use for the operation
 */
void launch_native64_encode(uint8_t bw,
                            uint64_t const* values,
                            uint64_t* packed,
                            uint64_t base_bits,
                            uint32_t total_count,
                            cudaStream_t stream);

/**
 * @brief Decode the given packed values using the FastLanes Native64 encoding strategy on the GPU.
 * @param bw Bit width used for encoding
 * @param packed Pointer to the input buffer of packed values on the GPU
 * @param decoded Pointer to the output buffer for the decoded values on the GPU
 * @param base_bits Base bits used for encoding
 * @param total_count Total number of values
 * @param stream CUDA stream to use for the operation
 */
void launch_native64_decode(uint8_t bw,
                            uint64_t const* packed,
                            uint64_t* decoded,
                            uint64_t base_bits,
                            uint32_t total_count,
                            cudaStream_t stream);

}  // namespace cudf::io::parquet::detail::fastlanes::native64
