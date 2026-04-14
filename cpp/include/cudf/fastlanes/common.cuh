#ifndef FLS_GPU_COMMON_CUH
#define FLS_GPU_COMMON_CUH

#include <cudf/utilities/export.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastlanes::gpu {
inline void* load_arr(void* src, uint64_t bsz)
{
  void* dest = nullptr;
  cudaMalloc((void**)&dest, bsz);
  cudaMemcpy(dest, src, bsz, cudaMemcpyHostToDevice);
  return dest;
}

template <typename T>
inline T* load_arr(T* src, uint64_t bsz)
{
  T* dest = nullptr;
  cudaMalloc((void**)&dest, bsz);
  cudaMemcpy(dest, src, bsz, cudaMemcpyHostToDevice);
  return dest;
}

#define CUDA_SAFE_CALL(call)                              \
  do {                                                    \
    cudaError_t err = call;                               \
    if (cudaSuccess != err) {                             \
      fprintf(stderr,                                     \
              "Cuda error in file '%s' in line %i : %s.", \
              __FILE__,                                   \
              __LINE__,                                   \
              cudaGetErrorString(err));                   \
      exit(EXIT_FAILURE);                                 \
    }                                                     \
  } while (0)
}  // namespace gpu

namespace fastlanes {

// =============================================================================
// Constants
// =============================================================================

constexpr uint64_t VECTOR_SIZE       = 1024;  // FastLanes processes 1024 values per vector
constexpr uint64_t PAYLOAD_ALIGNMENT = 128;   // Alignment for SIMD/GPU access
constexpr uint64_t FL_VECTOR_SIZE    = VECTOR_SIZE;  // Backward compat alias

// =============================================================================
// Type Aliases
// =============================================================================

using idx_t = uint64_t;
using bsz_t = uint64_t;
using n_t   = uint64_t;

// =============================================================================
// Type Cast Mode Enum
// =============================================================================

/**
 * @brief Describes how signed int32 input data was handled during encoding.
 *
 * The decoder uses this to correctly interpret the encoded data:
 * - SIGNED_SAFE: Input values were all >= 0
 * - SIGNED_REINTERPRET: Input contained negative values
 *
 * Cast mode value 0 was used by a removed legacy UINT path and is now invalid.
 */
enum class TypeCastMode : uint8_t {
  SIGNED_SAFE        = 1,  // Original data was signed, all values >= 0
  SIGNED_REINTERPRET = 2,  // Original data was signed, contains negative values
};

/**
 * @brief Convert a signed 32-bit value to its raw unsigned bit representation.
 */
__device__ __host__ inline constexpr uint32_t int32_to_u32_bits(int32_t value)
{
  return static_cast<uint32_t>(value);
}

/**
 * @brief Convert raw unsigned bits back to a signed 32-bit value.
 */
__device__ __host__ inline constexpr int32_t u32_bits_to_int32(uint32_t value)
{
  return value <= 0x7fffffffu
           ? static_cast<int32_t>(value)
           : static_cast<int32_t>(static_cast<int64_t>(value) - (int64_t{1} << 32));
}

/**
 * @brief Convert a signed 64-bit value to its raw unsigned bit representation.
 */
__device__ __host__ inline constexpr uint64_t int64_to_u64_bits(int64_t value)
{
  return static_cast<uint64_t>(value);
}

/**
 * @brief Convert raw unsigned bits back to a signed 64-bit value.
 */
__device__ __host__ inline constexpr int64_t u64_bits_to_int64(uint64_t value)
{
  return value <= 0x7fffffffffffffffULL
           ? static_cast<int64_t>(value)
           : -int64_t{1} - static_cast<int64_t>(~value);
}

// =============================================================================
// Sizing Utilities
// =============================================================================

/**
 * @brief Calculate number of 1024-element vectors needed for a given count
 * @param count Original element count
 * @return Number of vectors (rounded up)
 */
__device__ __host__ inline constexpr uint64_t num_vectors(uint64_t count)
{
  return (count + VECTOR_SIZE - 1) / VECTOR_SIZE;
}

/**
 * @brief Calculate the number of elements after padding to VECTOR_SIZE (1024) boundary
 * @param count Original element count
 * @return Padded count (multiple of 1024)
 */
__device__ __host__ inline constexpr uint64_t padded_count(uint64_t count)
{
  return num_vectors(count) * VECTOR_SIZE;
}

/**
 * @brief Calculate encoded output size in bytes for a given count and bitwidth
 *
 * Formula: Each vector of 1024 values produces (1024 * bitwidth) / 8 bytes
 *          = 128 * bitwidth bytes per vector
 *
 * @param count Number of elements (will be padded to 1024 boundary internally)
 * @param bitwidth Bits per element after encoding (1-32)
 * @return Size in bytes of encoded output
 */
__device__ __host__ inline constexpr size_t encoded_size_bytes(uint64_t count, uint8_t bitwidth)
{
  return num_vectors(count) * VECTOR_SIZE * bitwidth / 8;
}

/**
 * @brief Compute the unsigned bit mask for a given compile-time bitwidth.
 */
template <uint8_t BW>
__device__ __host__ inline constexpr uint64_t mask_for_bw()
{
  if constexpr (BW == 0) {
    return 0ULL;
  } else if constexpr (BW >= 64) {
    return ~uint64_t{0};
  } else {
    return (uint64_t{1} << BW) - 1ULL;
  }
}

/**
 * @brief Compute encoded 64-bit words per vector for a compile-time bitwidth.
 */
template <uint8_t BW>
__device__ __host__ inline constexpr uint32_t words_per_vector_for_bw(uint32_t vector_size)
{
  return static_cast<uint32_t>(encoded_size_bytes(vector_size, BW) / sizeof(uint64_t));
}

/**
 * @brief Compute encoded 64-bit words per lane for a compile-time bitwidth.
 */
template <uint8_t BW>
__device__ __host__ inline constexpr uint32_t words_per_lane_for_bw(uint32_t vector_size,
                                                                     uint32_t lanes_per_vector)
{
  return words_per_vector_for_bw<BW>(vector_size) / lanes_per_vector;
}

/**
 * @brief Get maximum valid bitwidth for a given integer type
 * @tparam T Integer type
 * @return Maximum supported FastLanes bitwidth for this type (capped at 32)
 */
template <typename T>
__device__ __host__ inline constexpr uint8_t max_bitwidth()
{
  constexpr auto type_bits = static_cast<uint8_t>(sizeof(T) * 8);
  return type_bits < uint8_t{32} ? type_bits : uint8_t{32};
}

/**
 * @brief Return true if bitwidth is in the supported range [1, 32].
 */
__device__ __host__ inline constexpr bool is_valid_bitwidth(uint8_t bitwidth)
{
  return bitwidth >= 1 && bitwidth <= 32;
}

/**
 * @brief Return true if native64 bitwidth is in the supported range [1, 64].
 */
__device__ __host__ inline constexpr bool is_valid_native64_bitwidth(uint8_t bitwidth)
{
  return bitwidth >= 1 && bitwidth <= 64;
}

/**
 * @brief Default PRE_DELTA policy for RAW32 pages.
 */
__device__ __host__ inline constexpr bool default_pre_delta_for_raw32() { return true; }

/**
 * @brief Default PRE_DELTA policy for SPLIT64 pages.
 */
__device__ __host__ inline constexpr bool default_pre_delta_for_split64() { return true; }

/**
 * @brief Default PRE_DELTA policy for NATIVE64 pages.
 */
__device__ __host__ inline constexpr bool default_pre_delta_for_native64() { return false; }

/**
 * @brief Validate PRE_DELTA for RAW32 pages.
 */
__device__ __host__ inline constexpr bool is_pre_delta_valid_for_raw32(bool pre_delta)
{
  return pre_delta == default_pre_delta_for_raw32();
}

/**
 * @brief Validate PRE_DELTA for SPLIT64 pages.
 */
__device__ __host__ inline constexpr bool is_pre_delta_valid_for_split64(bool pre_delta)
{
  return pre_delta == default_pre_delta_for_split64();
}

/**
 * @brief Validate PRE_DELTA for NATIVE64 pages.
 */
__device__ __host__ inline constexpr bool is_pre_delta_valid_for_native64(bool pre_delta)
{
  return pre_delta == default_pre_delta_for_native64();
}

// =============================================================================
// Page Header
// =============================================================================

/**
 * @brief Metadata header prepended to every FastLanes encoded page.
 *
 * Layout in Memory (128 bytes total, aligned to PAYLOAD_ALIGNMENT):
 * ┌─────────────────────────────────────────────────────────────┐
 * │ Offset │ Size   │ Field           │ Description            │
 * ├────────┼────────┼─────────────────┼────────────────────────┤
 * │ 0      │ 1 byte │ bitwidth_lo     │ Raw/low component width│
 * │ 1      │ 1 byte │ bitwidth_hi     │ High component width   │
 * │ 2      │ 1 byte │ pre_delta       │ Payload pre-delta flag │
 * │ 3      │ 1 byte │ reserved_flags  │ Reserved (must be 0)   │
 * │ 4      │ 4 bytes│ original_count  │ Actual element count   │
 * │ 8      │ 4 bytes│ padded_count    │ Padded to 1024 boundary│
 * │ 12     │ 4 bytes│ body_size       │ Encoded body size      │
 * │ 16     │ 4 bytes│ min_value_lo    │ Low-component base bits│
 * │ 20     │ 4 bytes│ min_value_hi    │ High-component base    │
 * │ 24     │104bytes│ padding         │ Zero padding           │
 * └────────┴────────┴─────────────────┴────────────────────────┘
 * │ 128    │ ...    │ PAYLOAD         │ Encoded data           │
 * └────────┴────────┴─────────────────┴────────────────────────┘
 */
struct PageHeader {
  // Header field offsets
  static constexpr size_t OFFSET_COMPONENT_BW_LOW    = 0;
  static constexpr size_t OFFSET_COMPONENT_BW_HIGH   = 1;
  static constexpr size_t OFFSET_PRE_DELTA           = 2;
  static constexpr size_t OFFSET_RESERVED_FLAGS      = 3;
  static constexpr size_t OFFSET_ORIGINAL_COUNT      = 4;
  static constexpr size_t OFFSET_PADDED_COUNT        = 8;
  static constexpr size_t OFFSET_BODY_SIZE           = 12;
  static constexpr size_t OFFSET_MIN_VALUE_LOW_BITS  = 16;
  static constexpr size_t OFFSET_MIN_VALUE_HIGH_BITS = 20;

  // Data members
  uint8_t component_bitwidth_low;
  uint8_t component_bitwidth_high;
  bool pre_delta;
  uint32_t original_count;
  uint32_t padded_count;
  uint32_t body_size;
  uint32_t min_value_low_bits;
  uint32_t min_value_high_bits;

  /**
   * @brief Get total header size (128 bytes, aligned)
   */
  __device__ __host__ static constexpr size_t header_size() { return PAYLOAD_ALIGNMENT; }

  /**
   * @brief Serialize a scalar32 page header and encoded body into a single blob.
   *
   * @param bw Bitwidth used for encoding (1-32)
   * @param orig_count Original number of elements (before padding)
   * @param pad_count Number of elements after padding to 1024 boundary
   * @param min_val_low_bits Raw low-32 page-local minimum bits
   * @param encoded_body Pointer to encoded data
   * @param body_sz Size of encoded body in bytes
   * @return Complete blob ready for upload to GPU
   * @throws std::invalid_argument if parameters are invalid
   */
  static std::vector<uint8_t> serialize_scalar32(uint8_t bw,
                                                 uint32_t orig_count,
                                                 uint32_t pad_count,
                                                 uint32_t min_val_low_bits,
                                                 const uint8_t* encoded_body,
                                                 size_t body_sz,
                                                 bool pre_delta = default_pre_delta_for_raw32())
  {
    if (!is_valid_bitwidth(bw)) {
      throw std::invalid_argument("FastLanes: bitwidth must be in [1, 32]");
    }
    if (!is_pre_delta_valid_for_raw32(pre_delta)) {
      throw std::invalid_argument("FastLanes: RAW mode requires PRE_DELTA=true");
    }
    if (pad_count < orig_count) {
      throw std::invalid_argument("FastLanes: padded_count must be >= original_count");
    }
    if (pad_count % VECTOR_SIZE != 0 && pad_count != 0) {
      throw std::invalid_argument("FastLanes: padded_count must be multiple of 1024");
    }

    if (body_sz > std::numeric_limits<size_t>::max() - header_size()) {
      throw std::invalid_argument("FastLanes: encoded body size is too large");
    }
    std::vector<uint8_t> blob(header_size() + body_sz, uint8_t{0});

    // Write fields
    blob[OFFSET_COMPONENT_BW_LOW]  = bw;
    blob[OFFSET_COMPONENT_BW_HIGH] = 0;
    blob[OFFSET_PRE_DELTA]         = static_cast<uint8_t>(pre_delta ? 1 : 0);
    blob[OFFSET_RESERVED_FLAGS]    = 0;

    std::memcpy(blob.data() + OFFSET_ORIGINAL_COUNT, &orig_count, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_PADDED_COUNT, &pad_count, sizeof(uint32_t));

    uint32_t body_sz_u32 = static_cast<uint32_t>(body_sz);
    std::memcpy(blob.data() + OFFSET_BODY_SIZE, &body_sz_u32, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_MIN_VALUE_LOW_BITS, &min_val_low_bits, sizeof(uint32_t));

    uint32_t min_hi = 0;
    std::memcpy(blob.data() + OFFSET_MIN_VALUE_HIGH_BITS, &min_hi, sizeof(uint32_t));

    // Copy encoded body after header
    if (body_sz > 0 && encoded_body != nullptr) {
      std::memcpy(blob.data() + header_size(), encoded_body, body_sz);
    }

    return blob;
  }

  /**
   * @brief Serialize a split32 page header and encoded body.
   */
  static std::vector<uint8_t> serialize_split32(uint8_t bw_low,
                                                uint8_t bw_high,
                                                uint32_t orig_count,
                                                uint32_t pad_count,
                                                uint32_t min_val_low_bits,
                                                uint32_t min_val_high_bits,
                                                const uint8_t* encoded_body,
                                                size_t body_sz,
                                                bool pre_delta = default_pre_delta_for_split64())
  {
    if (!is_valid_bitwidth(bw_low) || !is_valid_bitwidth(bw_high)) {
      throw std::invalid_argument("FastLanes: split32 bitwidths must be in [1, 32]");
    }
    if (!is_pre_delta_valid_for_split64(pre_delta)) {
      throw std::invalid_argument("FastLanes: SPLIT64 mode requires PRE_DELTA=true");
    }
    if (pad_count < orig_count) {
      throw std::invalid_argument("FastLanes: padded_count must be >= original_count");
    }
    if (pad_count % VECTOR_SIZE != 0 && pad_count != 0) {
      throw std::invalid_argument("FastLanes: padded_count must be multiple of 1024");
    }

    if (body_sz > std::numeric_limits<size_t>::max() - header_size()) {
      throw std::invalid_argument("FastLanes: encoded body size is too large");
    }
    std::vector<uint8_t> blob(header_size() + body_sz, uint8_t{0});

    blob[OFFSET_PRE_DELTA]         = static_cast<uint8_t>(pre_delta ? 1 : 0);
    blob[OFFSET_RESERVED_FLAGS]    = 0;
    blob[OFFSET_COMPONENT_BW_LOW]  = bw_low;
    blob[OFFSET_COMPONENT_BW_HIGH] = bw_high;

    std::memcpy(blob.data() + OFFSET_ORIGINAL_COUNT, &orig_count, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_PADDED_COUNT, &pad_count, sizeof(uint32_t));

    uint32_t body_sz_u32 = static_cast<uint32_t>(body_sz);
    std::memcpy(blob.data() + OFFSET_BODY_SIZE, &body_sz_u32, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_MIN_VALUE_LOW_BITS, &min_val_low_bits, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_MIN_VALUE_HIGH_BITS, &min_val_high_bits, sizeof(uint32_t));

    // Copy encoded body after header
    if (body_sz > 0 && encoded_body != nullptr) {
      std::memcpy(blob.data() + header_size(), encoded_body, body_sz);
    }

    return blob;
  }

  /**
   * @brief Serialize a native64 page header and encoded body.
   */
  static std::vector<uint8_t> serialize_native64(uint8_t bw,
                                                 uint32_t orig_count,
                                                 uint32_t pad_count,
                                                 uint64_t min_val_bits,
                                                 const uint8_t* encoded_body,
                                                 size_t body_sz,
                                                 bool pre_delta = default_pre_delta_for_native64())
  {
    if (!is_valid_native64_bitwidth(bw)) {
      throw std::invalid_argument("FastLanes: native64 bitwidth must be in [1, 64]");
    }
    if (!is_pre_delta_valid_for_native64(pre_delta)) {
      throw std::invalid_argument("FastLanes: NATIVE64 mode requires PRE_DELTA=true");
    }
    if (pad_count < orig_count) {
      throw std::invalid_argument("FastLanes: padded_count must be >= original_count");
    }
    if (pad_count % VECTOR_SIZE != 0 && pad_count != 0) {
      throw std::invalid_argument("FastLanes: padded_count must be multiple of 1024");
    }

    if (body_sz > std::numeric_limits<size_t>::max() - header_size()) {
      throw std::invalid_argument("FastLanes: encoded body size is too large");
    }
    std::vector<uint8_t> blob(header_size() + body_sz, uint8_t{0});

    blob[OFFSET_PRE_DELTA]         = static_cast<uint8_t>(pre_delta ? 1 : 0);
    blob[OFFSET_RESERVED_FLAGS]    = 0;
    blob[OFFSET_COMPONENT_BW_LOW]  = bw;
    blob[OFFSET_COMPONENT_BW_HIGH] = 0;

    std::memcpy(blob.data() + OFFSET_ORIGINAL_COUNT, &orig_count, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_PADDED_COUNT, &pad_count, sizeof(uint32_t));

    uint32_t body_sz_u32 = static_cast<uint32_t>(body_sz);
    std::memcpy(blob.data() + OFFSET_BODY_SIZE, &body_sz_u32, sizeof(uint32_t));

    auto const min_val_low_bits  = static_cast<uint32_t>(min_val_bits);
    auto const min_val_high_bits = static_cast<uint32_t>(min_val_bits >> 32);
    std::memcpy(blob.data() + OFFSET_MIN_VALUE_LOW_BITS, &min_val_low_bits, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_MIN_VALUE_HIGH_BITS, &min_val_high_bits, sizeof(uint32_t));

    if (body_sz > 0 && encoded_body != nullptr) {
      std::memcpy(blob.data() + header_size(), encoded_body, body_sz);
    }

    return blob;
  }

  /**
   * @brief Deserialize header from raw bytes (device/host safe)
   * @param page_data Pointer to start of page data (header)
   * @return Populated PageHeader struct
   */
  __device__ __host__ static PageHeader deserialize(const uint8_t* page_data)
  {
    PageHeader h;
    h.component_bitwidth_low  = page_data[OFFSET_COMPONENT_BW_LOW];
    h.component_bitwidth_high = page_data[OFFSET_COMPONENT_BW_HIGH];
    h.pre_delta               = page_data[OFFSET_PRE_DELTA] != 0;

#ifdef __CUDA_ARCH__
    // Byte-wise reconstruction for unaligned safety on device
    auto read_u32 = [](const uint8_t* ptr) -> uint32_t {
      return uint32_t(ptr[0]) | (uint32_t(ptr[1]) << 8) | (uint32_t(ptr[2]) << 16) |
             (uint32_t(ptr[3]) << 24);
    };
    h.original_count = read_u32(page_data + OFFSET_ORIGINAL_COUNT);
    h.padded_count   = read_u32(page_data + OFFSET_PADDED_COUNT);
    h.body_size      = read_u32(page_data + OFFSET_BODY_SIZE);
    h.min_value_low_bits  = read_u32(page_data + OFFSET_MIN_VALUE_LOW_BITS);
    h.min_value_high_bits = read_u32(page_data + OFFSET_MIN_VALUE_HIGH_BITS);
#else
    std::memcpy(&h.original_count, page_data + OFFSET_ORIGINAL_COUNT, sizeof(uint32_t));
    std::memcpy(&h.padded_count, page_data + OFFSET_PADDED_COUNT, sizeof(uint32_t));
    std::memcpy(&h.body_size, page_data + OFFSET_BODY_SIZE, sizeof(uint32_t));
    std::memcpy(&h.min_value_low_bits, page_data + OFFSET_MIN_VALUE_LOW_BITS, sizeof(uint32_t));
    std::memcpy(
      &h.min_value_high_bits, page_data + OFFSET_MIN_VALUE_HIGH_BITS, sizeof(uint32_t));
#endif

    return h;
  }

  __device__ __host__ constexpr uint64_t min_value_bits() const
  {
    return (static_cast<uint64_t>(min_value_high_bits) << 32) |
           static_cast<uint64_t>(min_value_low_bits);
  }

  __device__ __host__ constexpr bool has_valid_raw_mode_metadata() const
  {
    return is_valid_bitwidth(component_bitwidth_low) && component_bitwidth_high == 0;
  }

  __device__ __host__ constexpr bool has_valid_split64_mode_metadata() const
  {
    return is_valid_bitwidth(component_bitwidth_low) &&
           is_valid_bitwidth(component_bitwidth_high);
  }

  __device__ __host__ constexpr bool has_valid_native64_mode_metadata() const
  {
    return is_valid_native64_bitwidth(component_bitwidth_low) && component_bitwidth_high == 0;
  }

  __device__ __host__ constexpr bool has_valid_common_metadata() const
  {
    if (padded_count < original_count) { return false; }
    if (padded_count != 0 && (padded_count % VECTOR_SIZE) != 0) { return false; }
    return true;
  }

  __device__ __host__ constexpr size_t expected_raw32_body_size_bytes() const
  {
    if (padded_count == 0) { return 0; }
    if (!has_valid_raw_mode_metadata()) { return 0; }
    return encoded_size_bytes(padded_count, component_bitwidth_low);
  }

  __device__ __host__ constexpr size_t expected_split64_body_size_bytes() const
  {
    if (padded_count == 0) { return 0; }
    if (!has_valid_split64_mode_metadata()) { return 0; }
    return encoded_size_bytes(padded_count, component_bitwidth_low) +
           encoded_size_bytes(padded_count, component_bitwidth_high);
  }

  __device__ __host__ constexpr size_t expected_native64_body_size_bytes() const
  {
    if (padded_count == 0) { return 0; }
    if (!has_valid_native64_mode_metadata()) { return 0; }
    return encoded_size_bytes(padded_count, component_bitwidth_low);
  }

  /**
   * @brief Get pointer to payload (encoded data starts at header_size() offset)
   */
  __device__ __host__ static const uint8_t* payload_ptr(const uint8_t* page_data)
  {
    return page_data + header_size();
  }

  __device__ __host__ static uint8_t* payload_ptr(uint8_t* page_data)
  {
    return page_data + header_size();
  }
};

/**
 * @brief Validate RAW32 page header metadata.
 */
__device__ __host__ inline constexpr bool is_valid_raw32_header(PageHeader const& header)
{
  if (!header.has_valid_common_metadata()) { return false; }
  if (!is_pre_delta_valid_for_raw32(header.pre_delta)) { return false; }
  if (!header.has_valid_raw_mode_metadata()) { return false; }

  auto const expected_body = header.expected_raw32_body_size_bytes();
  return header.body_size >= expected_body;
}

/**
 * @brief Validate SPLIT64 page header metadata.
 */
__device__ __host__ inline constexpr bool is_valid_split64_header(PageHeader const& header)
{
  if (!header.has_valid_common_metadata()) { return false; }
  if (!is_pre_delta_valid_for_split64(header.pre_delta)) { return false; }
  if (!header.has_valid_split64_mode_metadata()) { return false; }

  auto const expected_body = header.expected_split64_body_size_bytes();
  return header.body_size >= expected_body;
}

/**
 * @brief Validate NATIVE64 page header metadata.
 */
__device__ __host__ inline constexpr bool is_valid_native64_header(PageHeader const& header)
{
  if (!header.has_valid_common_metadata()) { return false; }
  if (!is_pre_delta_valid_for_native64(header.pre_delta)) { return false; }
  if (!header.has_valid_native64_mode_metadata()) { return false; }

  auto const expected_body = header.expected_native64_body_size_bytes();
  return header.body_size >= expected_body;
}

}  // namespace fastlanes

#endif  // FLS_GPU_COMMON_CUH
