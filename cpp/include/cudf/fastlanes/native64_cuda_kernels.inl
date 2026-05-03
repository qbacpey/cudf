/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source metadata helper: cpp/examples/parquet_io/tools/tests/run_parity_matrix.py
 */

namespace cudf::io::parquet::detail::fastlanes::native64 {

constexpr uint32_t kVectorSize     = 1024;
constexpr uint32_t kLanesPerVector = 16;
constexpr uint32_t kValuesPerLane  = kVectorSize / kLanesPerVector;

constexpr uint32_t kThreadsPerBlock = 32;
constexpr uint32_t kVectorsPerBlock = kThreadsPerBlock / kLanesPerVector;

static_assert((kVectorSize % kLanesPerVector) == 0,
              "kLanesPerVector must divide 1024 for launch geometry");
static_assert(kThreadsPerBlock >= kLanesPerVector,
              "kThreadsPerBlock must be at least kLanesPerVector");
static_assert((kThreadsPerBlock % kLanesPerVector) == 0,
              "kThreadsPerBlock must be a multiple of kLanesPerVector");

// ---------------------------------------------------------------------------
// Internal encode/decode runtime helpers
// ---------------------------------------------------------------------------

template <uint8_t BW>
__device__ __forceinline__ void encode_lane_native64(uint64_t const* __restrict vector_in,
                                                     uint64_t* __restrict vector_out,
                                                     uint32_t lane,
                                                     uint64_t base_bits)
{
  if constexpr (BW == 0) {
    return;
  } else {
    constexpr uint32_t kWordsPerLane =
      ::fastlanes::words_per_lane_for_bw<BW>(kVectorSize, kLanesPerVector);
    constexpr uint64_t kMask = ::fastlanes::mask_for_bw<BW>();

    uint64_t words[kWordsPerLane] = {};

    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
      auto const in_idx    = static_cast<size_t>(i) * kLanesPerVector + lane;
      auto const delta     = (vector_in[in_idx] - base_bits) & kMask;
      auto const bit_index = i * BW;
      auto const word_idx  = bit_index >> 6;
      auto const bit_off   = bit_index & 63;

      words[word_idx] |= delta << bit_off;
      if (bit_off > (64 - BW)) {
        words[word_idx + 1] |= delta >> (64 - bit_off);
      }
    }

    for (uint32_t word_idx = 0; word_idx < kWordsPerLane; ++word_idx) {
      auto const out_idx = static_cast<size_t>(word_idx) * kLanesPerVector + lane;
      vector_out[out_idx] = words[word_idx];
    }
  }
}

template <uint8_t BW>
__device__ __forceinline__ void decode_lane_native64(uint64_t const* __restrict vector_in,
                                                     uint64_t* __restrict vector_out,
                                                     uint32_t lane,
                                                     uint64_t base_bits)
{
  if constexpr (BW == 0) {
    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
      auto const out_idx = i * kLanesPerVector + lane;
      vector_out[out_idx] = base_bits;
    }
  } else {
    constexpr uint64_t kMask = ::fastlanes::mask_for_bw<BW>();

    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
      auto const out_idx   = i * kLanesPerVector + lane;
      auto const bit_index = i * BW;
      auto const word_idx  = bit_index >> 6;
      auto const bit_off   = bit_index & 63;
      auto const lo_idx    = static_cast<size_t>(word_idx) * kLanesPerVector + lane;
      auto const lo_word   = vector_in[lo_idx];
      auto delta           = lo_word >> bit_off;

      if (bit_off > (64 - BW)) {
        auto const hi_idx = (static_cast<size_t>(word_idx) + 1) * kLanesPerVector + lane;
        auto const hi_word = vector_in[hi_idx];
        delta |= hi_word << (64 - bit_off);
      }

      vector_out[out_idx] = base_bits + (delta & kMask);
    }
  }
}

template <uint8_t BW>
__global__ void encode_native64_kernel(uint64_t const* __restrict values,
                                       uint64_t* __restrict packed,
                                       uint64_t base_bits,
                                       uint32_t padded_count)
{
  auto const tid       = static_cast<uint32_t>(threadIdx.x);
  auto const subvec    = tid / kLanesPerVector;
  auto const lane      = tid % kLanesPerVector;
  auto const vector_id = static_cast<uint32_t>(blockIdx.x) * kVectorsPerBlock + subvec;

  auto const vector_start = vector_id * kVectorSize;
  if (vector_start >= padded_count) { return; }

  auto const* vector_in = values + vector_start;
  auto* vector_out =
    packed + static_cast<size_t>(vector_id) * ::fastlanes::words_per_vector_for_bw<BW>(kVectorSize);
  encode_lane_native64<BW>(vector_in, vector_out, lane, base_bits);
}

template <uint8_t BW>
__global__ void decode_native64_kernel(uint64_t const* __restrict packed,
                                       uint64_t* __restrict decoded,
                                       uint64_t base_bits,
                                       uint32_t padded_count)
{
  auto const tid       = static_cast<uint32_t>(threadIdx.x);
  auto const subvec    = tid / kLanesPerVector;
  auto const lane      = tid % kLanesPerVector;
  auto const vector_id = static_cast<uint32_t>(blockIdx.x) * kVectorsPerBlock + subvec;

  auto const vector_start = vector_id * kVectorSize;
  if (vector_start >= padded_count) { return; }

  auto const* vector_in =
    packed + static_cast<size_t>(vector_id) * ::fastlanes::words_per_vector_for_bw<BW>(kVectorSize);
  auto* vector_out      = decoded + vector_start;
  decode_lane_native64<BW>(vector_in, vector_out, lane, base_bits);
}

// Host-launch entrypoints. `values`, `packed`, and `decoded` must all point to CUDA device memory.
// These are not intended for direct device-side launch. For runtime BW dispatch from inside a CUDA
// kernel, use `encode_lane_by_bw_device_runtime` / `decode_lane_by_bw_device_runtime` below.

#define NATIVE64_FOR_EACH_BW(M) \
  M(0)                          \
  M(1)                          \
  M(2)                          \
  M(3)                          \
  M(4)                          \
  M(5)                          \
  M(6)                          \
  M(7)                          \
  M(8)                          \
  M(9)                          \
  M(10)                         \
  M(11)                         \
  M(12)                         \
  M(13)                         \
  M(14)                         \
  M(15)                         \
  M(16)                         \
  M(17)                         \
  M(18)                         \
  M(19)                         \
  M(20)                         \
  M(21)                         \
  M(22)                         \
  M(23)                         \
  M(24)                         \
  M(25)                         \
  M(26)                         \
  M(27)                         \
  M(28)                         \
  M(29)                         \
  M(30)                         \
  M(31)                         \
  M(32)                         \
  M(33)                         \
  M(34)                         \
  M(35)                         \
  M(36)                         \
  M(37)                         \
  M(38)                         \
  M(39)                         \
  M(40)                         \
  M(41)                         \
  M(42)                         \
  M(43)                         \
  M(44)                         \
  M(45)                         \
  M(46)                         \
  M(47)                         \
  M(48)                         \
  M(49)                         \
  M(50)                         \
  M(51)                         \
  M(52)                         \
  M(53)                         \
  M(54)                         \
  M(55)                         \
  M(56)                         \
  M(57)                         \
  M(58)                         \
  M(59)                         \
  M(60)                         \
  M(61)                         \
  M(62)                         \
  M(63)                         \
  M(64)

// Device-side runtime BW dispatcher for kernels that compute BW on GPU.
// `vector_in`/`vector_out` are one-vector striped pointers and `lane` is thread lane [0, 15].
__device__ __forceinline__ void encode_lane_by_bw_device_runtime(uint8_t bw,
                                                                  uint64_t const* vector_in,
                                                                  uint64_t* vector_out,
                                                                  uint32_t lane,
                                                                  uint64_t base_bits)
{
  switch (bw) {
#define ENCODE_LANE_CASE(BW) \
  case BW: encode_lane_native64<BW>(vector_in, vector_out, lane, base_bits); return;
    NATIVE64_FOR_EACH_BW(ENCODE_LANE_CASE)
#undef ENCODE_LANE_CASE
    default: return;
  }
}

// Device-side runtime BW dispatcher for decode inside CUDA kernels.
__device__ __forceinline__ void decode_lane_by_bw_device_runtime(uint8_t bw,
                                                                  uint64_t const* vector_in,
                                                                  uint64_t* vector_out,
                                                                  uint32_t lane,
                                                                  uint64_t base_bits)
{
  switch (bw) {
#define DECODE_LANE_CASE(BW) \
  case BW: decode_lane_native64<BW>(vector_in, vector_out, lane, base_bits); return;
    NATIVE64_FOR_EACH_BW(DECODE_LANE_CASE)
#undef DECODE_LANE_CASE
    default: return;
  }
}

using encode_device_ptr_dispatch_fn =
  void (*)(uint64_t const*, uint64_t*, uint64_t, uint32_t, cudaStream_t);
using decode_device_ptr_dispatch_fn =
  void (*)(uint64_t const*, uint64_t*, uint64_t, uint32_t, cudaStream_t);

template <uint8_t BW>
void encode_device_ptr_dispatch_wrapper(uint64_t const* values,
                                        uint64_t* packed,
                                        uint64_t base_bits,
                                        uint32_t total_count,
                                        cudaStream_t stream)
{
  auto const padded_count = static_cast<uint32_t>(::fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(::fastlanes::num_vectors(total_count));
  if (num_vectors == 0) { return; }

  if constexpr (BW == 0) {
    return;
  } else {
    auto const num_blocks = (num_vectors + kVectorsPerBlock - 1U) / kVectorsPerBlock;
    encode_native64_kernel<BW><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
      values, packed, base_bits, padded_count);

    auto const launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
      throw std::runtime_error(cuda_error(launch_status, "native64 encode kernel launch"));
    }
  }
}

template <uint8_t BW>
void decode_device_ptr_dispatch_wrapper(uint64_t const* packed,
                                        uint64_t* decoded,
                                        uint64_t base_bits,
                                        uint32_t total_count,
                                        cudaStream_t stream)
{
  auto const padded_count = static_cast<uint32_t>(::fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(::fastlanes::num_vectors(total_count));
  if (num_vectors == 0) { return; }

  auto const num_blocks = (num_vectors + kVectorsPerBlock - 1U) / kVectorsPerBlock;
  decode_native64_kernel<BW><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
    packed, decoded, base_bits, padded_count);

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(launch_status, "native64 decode kernel launch"));
  }
}

inline constexpr std::array<encode_device_ptr_dispatch_fn, 65> kEncodeDevicePtrDispatch = {{
#define ENCODE_DISPATCH_ENTRY(BW) &encode_device_ptr_dispatch_wrapper<BW>,
  NATIVE64_FOR_EACH_BW(ENCODE_DISPATCH_ENTRY)
#undef ENCODE_DISPATCH_ENTRY
}};

inline constexpr std::array<decode_device_ptr_dispatch_fn, 65> kDecodeDevicePtrDispatch = {{
#define DECODE_DISPATCH_ENTRY(BW) &decode_device_ptr_dispatch_wrapper<BW>,
  NATIVE64_FOR_EACH_BW(DECODE_DISPATCH_ENTRY)
#undef DECODE_DISPATCH_ENTRY
}};

#undef NATIVE64_FOR_EACH_BW


}  // namespace cudf::io::parquet::detail::fastlanes::native64
