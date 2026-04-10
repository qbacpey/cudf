/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test-only CUDA helpers for FL64 bw37 parity.
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

__global__ void encode_native64_bw37_test_kernel(uint64_t const* __restrict values,
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

__global__ void decode_native64_bw37_test_kernel(uint64_t const* __restrict packed,
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

[[nodiscard]] std::vector<uint64_t> encode_bw37_on_gpu(std::vector<uint64_t> const& values,
                                                        uint64_t base_bits,
                                                        uint32_t total_count)
{
  auto const stream      = cudf::get_default_stream();
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));
  auto const num_blocks =
    (num_vectors + kVectorsPerBlockBw37 - 1U) / kVectorsPerBlockBw37;

  if (values.size() != padded_count) {
    throw std::invalid_argument("encode_bw37_on_gpu values size mismatch");
  }

  std::vector<uint64_t> packed(static_cast<size_t>(num_vectors) * kWordsPerVector37, 0ULL);

  uint64_t* d_values = nullptr;
  uint64_t* d_packed = nullptr;

  auto const values_bytes = values.size() * sizeof(uint64_t);
  auto const packed_bytes = packed.size() * sizeof(uint64_t);

  auto const malloc_values_status = cudaMalloc(reinterpret_cast<void**>(&d_values), values_bytes);
  if (malloc_values_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(malloc_values_status, "cudaMalloc(d_values)"));
  }

  auto const malloc_packed_status = cudaMalloc(reinterpret_cast<void**>(&d_packed), packed_bytes);
  if (malloc_packed_status != cudaSuccess) {
    cudaFree(d_values);
    throw std::runtime_error(cuda_error(malloc_packed_status, "cudaMalloc(d_packed)"));
  }

  auto cleanup = [&]() {
    cudaFree(d_values);
    cudaFree(d_packed);
  };

  auto const h2d_status = cudaMemcpyAsync(
    d_values, values.data(), values_bytes, cudaMemcpyHostToDevice, stream.value());
  if (h2d_status != cudaSuccess) {
    auto err_msg = cuda_error(h2d_status, "cudaMemcpy H2D values");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const memset_status = cudaMemsetAsync(d_packed, 0, packed_bytes, stream.value());
  if (memset_status != cudaSuccess) {
    auto err_msg = cuda_error(memset_status, "cudaMemset packed");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  encode_native64_bw37_test_kernel<<<num_blocks, kThreadsPerBlockBw37, 0, stream.value()>>>(
    d_values, d_packed, base_bits, padded_count);

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    auto err_msg = cuda_error(launch_status, "bw37 encode kernel launch");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    auto err_msg = cuda_error(sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const d2h_status = cudaMemcpyAsync(
    packed.data(), d_packed, packed_bytes, cudaMemcpyDeviceToHost, stream.value());
  if (d2h_status != cudaSuccess) {
    auto err_msg = cuda_error(d2h_status, "cudaMemcpy D2H packed");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const d2h_sync_status = cudaStreamSynchronize(stream.value());
  if (d2h_sync_status != cudaSuccess) {
    auto err_msg = cuda_error(d2h_sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  cleanup();
  return packed;
}

[[nodiscard]] std::vector<uint64_t> decode_bw37_on_gpu(std::vector<uint64_t> const& packed,
                                                        uint64_t base_bits,
                                                        uint32_t total_count)
{
  auto const stream      = cudf::get_default_stream();
  auto const padded_count = static_cast<size_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));
  auto const num_blocks =
    (num_vectors + kVectorsPerBlockBw37 - 1U) / kVectorsPerBlockBw37;

  if (packed.size() != static_cast<size_t>(num_vectors) * kWordsPerVector37) {
    throw std::invalid_argument("decode_bw37_on_gpu packed size mismatch");
  }

  std::vector<uint64_t> out_host(padded_count, 0ULL);

  uint64_t* d_packed  = nullptr;
  uint64_t* d_decoded = nullptr;

  auto const packed_bytes  = packed.size() * sizeof(uint64_t);
  auto const decoded_bytes = out_host.size() * sizeof(uint64_t);

  auto const malloc_packed_status = cudaMalloc(reinterpret_cast<void**>(&d_packed), packed_bytes);
  if (malloc_packed_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(malloc_packed_status, "cudaMalloc(d_packed)"));
  }

  auto const malloc_out_status = cudaMalloc(reinterpret_cast<void**>(&d_decoded), decoded_bytes);
  if (malloc_out_status != cudaSuccess) {
    cudaFree(d_packed);
    throw std::runtime_error(cuda_error(malloc_out_status, "cudaMalloc(d_decoded)"));
  }

  auto cleanup = [&]() {
    cudaFree(d_packed);
    cudaFree(d_decoded);
  };

  auto const h2d_status = cudaMemcpyAsync(
    d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice, stream.value());
  if (h2d_status != cudaSuccess) {
    auto err_msg = cuda_error(h2d_status, "cudaMemcpy H2D packed");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const memset_status = cudaMemsetAsync(d_decoded, 0, decoded_bytes, stream.value());
  if (memset_status != cudaSuccess) {
    auto err_msg = cuda_error(memset_status, "cudaMemset decoded");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  decode_native64_bw37_test_kernel<<<num_blocks, kThreadsPerBlockBw37, 0, stream.value()>>>(
    d_packed, d_decoded, base_bits, static_cast<uint32_t>(padded_count));

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    auto err_msg = cuda_error(launch_status, "bw37 test kernel launch");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    auto err_msg = cuda_error(sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const d2h_status = cudaMemcpyAsync(
    out_host.data(), d_decoded, decoded_bytes, cudaMemcpyDeviceToHost, stream.value());
  if (d2h_status != cudaSuccess) {
    auto err_msg = cuda_error(d2h_status, "cudaMemcpy D2H decoded");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const d2h_sync_status = cudaStreamSynchronize(stream.value());
  if (d2h_sync_status != cudaSuccess) {
    auto err_msg = cuda_error(d2h_sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  cleanup();
  return out_host;
}
