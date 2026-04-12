/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace native64_generated {

uint64_t derive_min_base_bits(uint64_t const* values,
                              uint32_t total_count,
                              cudaStream_t stream);

void encode_by_bw_gpu_device_ptrs(uint8_t bw,
                                  uint64_t const* values,
                                  uint64_t* packed,
                                  uint64_t base_bits,
                                  uint32_t total_count,
                                  cudaStream_t stream);

void decode_by_bw_gpu_device_ptrs(uint8_t bw,
                                  uint64_t const* packed,
                                  uint64_t* decoded,
                                  uint64_t base_bits,
                                  uint32_t total_count,
                                  cudaStream_t stream);

}  // namespace native64_generated
