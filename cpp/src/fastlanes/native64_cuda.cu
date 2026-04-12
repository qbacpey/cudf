/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source metadata helper: cpp/examples/parquet_io/tools/tests/run_parity_matrix.py
 */

#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/native64_cuda.cuh>

#include <cuda_runtime_api.h>
#include <rmm/exec_policy.hpp>
#include <thrust/reduce.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace native64_generated {

[[nodiscard]] std::string cuda_error(cudaError_t status, char const* operation);

}  // namespace native64_generated

#include <cudf/fastlanes/native64_cuda_kernels.inl>

namespace native64_generated {

[[nodiscard]] std::string cuda_error(cudaError_t status, char const* operation)
{
  std::ostringstream oss;
  oss << operation << " failed: " << cudaGetErrorString(status);
  return oss.str();
}

uint64_t derive_min_base_bits(uint64_t const* values, uint32_t total_count, cudaStream_t stream)
{
  if (total_count == 0) { return uint64_t{0}; }
  if (values == nullptr) {
    throw std::invalid_argument("derive_min_base_bits requires non-null values when total_count > 0");
  }

  return thrust::reduce(rmm::exec_policy(stream),
                        values,
                        values + total_count,
                        std::numeric_limits<uint64_t>::max(),
                        thrust::minimum<uint64_t>());
}

void encode_by_bw_gpu_device_ptrs(uint8_t bw,
                                  uint64_t const* values,
                                  uint64_t* packed,
                                  uint64_t base_bits,
                                  uint32_t total_count,
                                  cudaStream_t stream)
{
  if (bw > 64) {
    throw std::invalid_argument("encode_by_bw_gpu_device_ptrs requires bw in [0,64]");
  }
  kEncodeDevicePtrDispatch[bw](values, packed, base_bits, total_count, stream);
}

void decode_by_bw_gpu_device_ptrs(uint8_t bw,
                                  uint64_t const* packed,
                                  uint64_t* decoded,
                                  uint64_t base_bits,
                                  uint32_t total_count,
                                  cudaStream_t stream)
{
  if (bw > 64) {
    throw std::invalid_argument("decode_by_bw_gpu_device_ptrs requires bw in [0,64]");
  }
  kDecodeDevicePtrDispatch[bw](packed, decoded, base_bits, total_count, stream);
}

}  // namespace native64_generated
