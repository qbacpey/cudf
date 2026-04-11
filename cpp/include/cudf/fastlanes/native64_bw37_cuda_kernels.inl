/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA helpers for FL64 bw37 parity harness.
 * This file is intentionally included by parquet_fastlanes_native64_bw37_test.cu
 * inside its anonymous namespace and expects constants plus cuda_error() from
 * the including translation unit.
 */

__device__ __forceinline__ void encode_lane_native64_bw37(uint64_t const* __restrict vector_in,
                                                           uint64_t* __restrict vector_out,
                                                           uint32_t lane,
                                                           uint64_t base_bits)
{
  uint64_t words[kWordsPerLane37] = {};

  for (uint32_t i = 0; i < kValuesPerLane; ++i) {
    auto const delta =
      (vector_in[static_cast<size_t>(i) * kLanesPerVector + lane] - base_bits) & kBw37Mask;
    auto const bit_index = i * kBw37;
    auto const word_idx  = bit_index >> 6;
    auto const bit_off   = bit_index & 63;

    words[word_idx] |= delta << bit_off;
    if (bit_off > (64 - kBw37)) {
      words[word_idx + 1] |= delta >> (64 - bit_off);
    }
  }

  for (uint32_t word_idx = 0; word_idx < kWordsPerLane37; ++word_idx) {
    vector_out[static_cast<size_t>(word_idx) * kLanesPerVector + lane] = words[word_idx];
  }
}

__device__ __forceinline__ void decode_lane_native64_bw37(uint64_t const* __restrict vector_in,
                                                           uint64_t* __restrict vector_out,
                                                           uint32_t lane,
                                                           uint64_t base_bits)
{
  for (uint32_t i = 0; i < kValuesPerLane; ++i) {
    auto const out_idx    = i * kLanesPerVector + lane;
    auto const bit_index  = i * kBw37;
    auto const word_idx   = bit_index >> 6;
    auto const bit_off    = bit_index & 63;
    auto const lo_word    = vector_in[static_cast<size_t>(word_idx) * kLanesPerVector + lane];
    auto delta            = lo_word >> bit_off;

    if (bit_off > (64 - kBw37)) {
      auto const hi_word = vector_in[(static_cast<size_t>(word_idx) + 1) * kLanesPerVector + lane];
      delta |= hi_word << (64 - bit_off);
    }

    vector_out[out_idx] = base_bits + (delta & kBw37Mask);
  }
}

static_assert(kLanesPerVector == 16,
              "bw37 helper kernels require a 16-lane striped bitstream contract");
constexpr uint32_t kThreadsPerBlockBw37 = 32;
static_assert(kThreadsPerBlockBw37 >= 16, "kThreadsPerBlockBw37 must be at least 16");
static_assert((kThreadsPerBlockBw37 % 16) == 0,
              "kThreadsPerBlockBw37 must be a multiple of 16");
constexpr uint32_t kVectorsPerBlockBw37 = kThreadsPerBlockBw37 / 16;

__global__ void encode_native64_bw37_kernel(uint64_t const* __restrict values,
                                            uint64_t* __restrict packed,
                                            uint64_t base_bits,
                                            uint32_t padded_count)
{
  auto const tid       = static_cast<uint32_t>(threadIdx.x);
  auto const subvec    = tid / kLanesPerVector;
  auto const lane      = tid % kLanesPerVector;
  auto const vector_id = static_cast<uint32_t>(blockIdx.x) * kVectorsPerBlockBw37 + subvec;

  auto const vector_start = vector_id * kVectorSize;
  if (vector_start >= padded_count) { return; }

  auto const* vector_in = values + vector_start;
  auto* vector_out      = packed + static_cast<size_t>(vector_id) * kWordsPerVector37;

  encode_lane_native64_bw37(vector_in, vector_out, lane, base_bits);
}

__global__ void decode_native64_bw37_kernel(uint64_t const* __restrict packed,
                                            uint64_t* __restrict decoded,
                                            uint64_t base_bits,
                                            uint32_t padded_count)
{
  auto const tid       = static_cast<uint32_t>(threadIdx.x);
  auto const subvec    = tid / kLanesPerVector;
  auto const lane      = tid % kLanesPerVector;
  auto const vector_id = static_cast<uint32_t>(blockIdx.x) * kVectorsPerBlockBw37 + subvec;

  auto const vector_start = vector_id * kVectorSize;
  if (vector_start >= padded_count) { return; }

  auto const* vector_in = packed + static_cast<size_t>(vector_id) * kWordsPerVector37;
  auto* vector_out      = decoded + vector_start;

  decode_lane_native64_bw37(vector_in, vector_out, lane, base_bits);
}

inline void encode_bw37_gpu(uint64_t const* values,
                            uint64_t* packed,
                            uint64_t base_bits,
                            uint32_t total_count,
                            cudaStream_t stream)
{
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));
  if (num_vectors == 0) { return; }

  auto const num_blocks = (num_vectors + kVectorsPerBlockBw37 - 1U) / kVectorsPerBlockBw37;
  encode_native64_bw37_kernel<<<num_blocks, kThreadsPerBlockBw37, 0, stream>>>(
    values, packed, base_bits, padded_count);

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(launch_status, "bw37 encode kernel launch"));
  }
}

inline void decode_bw37_gpu(uint64_t const* packed,
                            uint64_t* decoded,
                            uint64_t base_bits,
                            uint32_t total_count,
                            cudaStream_t stream)
{
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));
  if (num_vectors == 0) { return; }

  auto const num_blocks = (num_vectors + kVectorsPerBlockBw37 - 1U) / kVectorsPerBlockBw37;
  decode_native64_bw37_kernel<<<num_blocks, kThreadsPerBlockBw37, 0, stream>>>(
    packed, decoded, base_bits, padded_count);

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(launch_status, "bw37 decode kernel launch"));
  }
}
