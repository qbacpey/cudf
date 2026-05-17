#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "common.cuh"
#include "fls_gen/unpack/unpack.cuh"

namespace cudf::io::parquet::detail::fastlanes_cudf {

/**
 * @brief GPU kernel to decode FastLanes bit-packed data
 * 
 * Each CUDA block decodes one vector (1024 values).
 * Requires exactly 32 threads per block (one warp).
 */
// void decode_kernel(
//     const uint32_t* __restrict__ encoded,
//     uint32_t* __restrict__ decoded,
//     uint8_t bitwidth,
//     uint64_t num_vectors)
// {
//   uint64_t vec_idx = blockIdx.x;
//   if (vec_idx >= num_vectors) return;
  
//   // Shared memory for intermediate unpacked data
//   __shared__ uint32_t unpacked_smem[::fastlanes::VECTOR_SIZE];
  
//   // Calculate pointers for this vector
//   const uint32_t* enc_ptr = encoded + (vec_idx * bitwidth * 32);
//   uint32_t* dec_ptr = decoded + (vec_idx * ::fastlanes::VECTOR_SIZE);
  
//   // Call FastLanes device unpack (requires 32 threads working together)
//   unpack_device(enc_ptr, unpacked_smem, bitwidth);
//   __syncthreads();
  
//   // Copy from shared memory to global memory
//   // Each thread copies multiple elements
//   for (int i = threadIdx.x; i < ::fastlanes::VECTOR_SIZE; i += 32) {
//     dec_ptr[i] = unpacked_smem[i];
//   }
// }

/**
 * @brief Decode FastLanes bit-packed data on GPU
 * 
 * @param encoded Encoded data in device memory
 * @param decoded Output buffer in device memory (must hold num_vectors * 1024 values)
 * @param bitwidth Bits per value used during encoding
 * @param num_vectors Number of 1024-value vectors
 * @param stream CUDA stream for async execution
 */
// inline void decode_bitpack(
//     const uint32_t* encoded,
//     uint32_t* decoded,
//     uint8_t bitwidth,
//     uint64_t num_vectors,
//     cudaStream_t stream = 0)
// {
//   if (num_vectors == 0) return;
  
//   // Launch one block per vector, 32 threads (one warp) per block
//   decode_kernel<<<num_vectors, 32, 0, stream>>>(
//       encoded, decoded, bitwidth, num_vectors);
// }

}  // namespace cudf::io::parquet::detail::fastlanes