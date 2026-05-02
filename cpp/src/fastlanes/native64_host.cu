/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source metadata helper: cpp/examples/parquet_io/tools/tests/run_parity_matrix.py
 */

#include <cudf/fastlanes/native64_host.hpp>
#include <cudf/fastlanes/native64_device.cuh>

#include <cuda_runtime_api.h>
#include <rmm/exec_policy.hpp>
#include <thrust/reduce.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

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

void launch_native64_encode(uint8_t bw,
                            uint64_t const* values,
                            uint64_t* packed,
                            uint64_t base_bits,
                            uint32_t total_count,
                            cudaStream_t stream)
{
  if (bw > 64) {
    throw std::invalid_argument("launch_native64_encode requires bw in [0,64]");
  }
  kEncodeDevicePtrDispatch[bw](values, packed, base_bits, total_count, stream);
}

void launch_native64_decode(uint8_t bw,
                            uint64_t const* packed,
                            uint64_t* decoded,
                            uint64_t base_bits,
                            uint32_t total_count,
                            cudaStream_t stream)
{
  if (bw > 64) {
    throw std::invalid_argument("launch_native64_decode requires bw in [0,64]");
  }
  kDecodeDevicePtrDispatch[bw](packed, decoded, base_bits, total_count, stream);
}

}  // namespace native64_generated
