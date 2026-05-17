#ifndef FLS_GPU_COMMON_CUH
#define FLS_GPU_COMMON_CUH

#include <cudf/utilities/export.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
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
 * @brief Return true if a serialized cast mode byte is currently supported.
 */
__device__ __host__ inline constexpr bool is_valid_cast_mode(uint8_t mode)
{
  return mode == static_cast<uint8_t>(TypeCastMode::SIGNED_SAFE) ||
         mode == static_cast<uint8_t>(TypeCastMode::SIGNED_REINTERPRET);
}

/**
 * @brief Return true if a TypeCastMode value is currently supported.
 */
__device__ __host__ inline constexpr bool is_valid_cast_mode(TypeCastMode mode)
{
  return is_valid_cast_mode(static_cast<uint8_t>(mode));
}

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

// =============================================================================
// Sizing Utilities
// =============================================================================

/**
 * @brief Calculate number of 1024-element vectors needed for a given count
 * @param count Original element count
 * @return Number of vectors (rounded up)
 */
inline constexpr uint64_t num_vectors(uint64_t count)
{
  return (count + VECTOR_SIZE - 1) / VECTOR_SIZE;
}

/**
 * @brief Calculate the number of elements after padding to VECTOR_SIZE (1024) boundary
 * @param count Original element count
 * @return Padded count (multiple of 1024)
 */
inline constexpr uint64_t padded_count(uint64_t count)
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
inline constexpr size_t encoded_size_bytes(uint64_t count, uint8_t bitwidth)
{
  return num_vectors(count) * VECTOR_SIZE * bitwidth / 8;
}

/**
 * @brief Get maximum valid bitwidth for a given integer type
 * @tparam T Integer type
 * @return Maximum bitwidth (8 for int8, 16 for int16, 32 for int32, 64 for int64)
 */
template <typename T>
inline constexpr uint8_t max_bitwidth()
{
  return sizeof(T) * 8;
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
 * │ 0      │ 1 byte │ bitwidth        │ Bits per encoded value │
 * │ 1      │ 1 byte │ cast_mode       │ TypeCastMode enum      │
 * │ 2      │ 2 bytes│ reserved        │ Future use (zero)      │
 * │ 4      │ 4 bytes│ original_count  │ Actual element count   │
 * │ 8      │ 4 bytes│ padded_count    │ Padded to 1024 boundary│
 * │ 12     │ 4 bytes│ body_size       │ Encoded body size      │
 * │ 16     │ 4 bytes│ min_value       │ Page-local base value  │
 * │ 20     │108bytes│ padding         │ Zero padding           │
 * └────────┴────────┴─────────────────┴────────────────────────┘
 * │ 128    │ ...    │ PAYLOAD         │ Encoded data           │
 * └────────┴────────┴─────────────────┴────────────────────────┘
 */
struct PageHeader {
  // Header field offsets
  static constexpr size_t OFFSET_BITWIDTH       = 0;
  static constexpr size_t OFFSET_CAST_MODE      = 1;
  static constexpr size_t OFFSET_RESERVED       = 2;
  static constexpr size_t OFFSET_ORIGINAL_COUNT = 4;
  static constexpr size_t OFFSET_PADDED_COUNT   = 8;
  static constexpr size_t OFFSET_BODY_SIZE      = 12;
  static constexpr size_t OFFSET_MIN_VALUE      = 16;

  // Data members
  uint8_t bitwidth;
  TypeCastMode cast_mode;
  uint32_t original_count;
  uint32_t padded_count;
  uint32_t body_size;
  uint32_t min_value;

  /**
   * @brief Get total header size (128 bytes, aligned)
   */
  __device__ __host__ static constexpr size_t header_size() { return PAYLOAD_ALIGNMENT; }

  // Backward compat alias
  __device__ __host__  static constexpr size_t padded_size() { return header_size(); }

  /**
   * @brief Serialize header and encoded body into a single blob
   *
   * @param bw Bitwidth used for encoding (1-32)
   * @param mode TypeCastMode indicating how data was cast
   * @param orig_count Original number of elements (before padding)
   * @param pad_count Number of elements after padding to 1024 boundary
   * @param min_val Raw 32-bit representation of the page-local minimum
   * @param encoded_body Pointer to encoded data
   * @param body_sz Size of encoded body in bytes
   * @return Complete blob ready for upload to GPU
   * @throws std::invalid_argument if parameters are invalid
   */
  static std::vector<uint8_t> serialize(uint8_t bw,
                                        TypeCastMode mode,
                                        uint32_t orig_count,
                                        uint32_t pad_count,
                                        uint32_t min_val,
                                        const uint8_t* encoded_body,
                                        size_t body_sz)
  {
    if (bw == 0 || bw > 32) {
      throw std::invalid_argument("FastLanes: bitwidth must be 1-32");
    }
    if (pad_count < orig_count) {
      throw std::invalid_argument("FastLanes: padded_count must be >= original_count");
    }
    if (pad_count % VECTOR_SIZE != 0 && pad_count != 0) {
      throw std::invalid_argument("FastLanes: padded_count must be multiple of 1024");
    }

    std::vector<uint8_t> blob(header_size() + body_sz);

    // Zero entire header first
    std::memset(blob.data(), 0, header_size());

    // Write fields
    blob[OFFSET_BITWIDTH]  = bw;
    blob[OFFSET_CAST_MODE] = static_cast<uint8_t>(mode);

    std::memcpy(blob.data() + OFFSET_ORIGINAL_COUNT, &orig_count, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_PADDED_COUNT, &pad_count, sizeof(uint32_t));

    uint32_t body_sz_u32 = static_cast<uint32_t>(body_sz);
    std::memcpy(blob.data() + OFFSET_BODY_SIZE, &body_sz_u32, sizeof(uint32_t));
    std::memcpy(blob.data() + OFFSET_MIN_VALUE, &min_val, sizeof(uint32_t));

    // Copy encoded body after header
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
    h.bitwidth  = page_data[OFFSET_BITWIDTH];
    h.cast_mode = static_cast<TypeCastMode>(page_data[OFFSET_CAST_MODE]);

#ifdef __CUDA_ARCH__
    // Byte-wise reconstruction for unaligned safety on device
    auto read_u32 = [](const uint8_t* ptr) -> uint32_t {
      return uint32_t(ptr[0]) | (uint32_t(ptr[1]) << 8) | (uint32_t(ptr[2]) << 16) |
             (uint32_t(ptr[3]) << 24);
    };
    h.original_count = read_u32(page_data + OFFSET_ORIGINAL_COUNT);
    h.padded_count   = read_u32(page_data + OFFSET_PADDED_COUNT);
    h.body_size      = read_u32(page_data + OFFSET_BODY_SIZE);
    h.min_value      = read_u32(page_data + OFFSET_MIN_VALUE);
#else
    std::memcpy(&h.original_count, page_data + OFFSET_ORIGINAL_COUNT, sizeof(uint32_t));
    std::memcpy(&h.padded_count, page_data + OFFSET_PADDED_COUNT, sizeof(uint32_t));
    std::memcpy(&h.body_size, page_data + OFFSET_BODY_SIZE, sizeof(uint32_t));
    std::memcpy(&h.min_value, page_data + OFFSET_MIN_VALUE, sizeof(uint32_t));
#endif
    return h;
  }

  // Backward compat alias
  __device__ __host__ static PageHeader read(const uint8_t* page_data)
  {
    return deserialize(page_data);
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

  // Backward compat aliases
  __device__ __host__ static const uint8_t* get_payload_ptr(const uint8_t* page_data)
  {
    return payload_ptr(page_data);
  }

  __device__ __host__ static uint8_t* get_payload_ptr(uint8_t* page_data)
  {
    return payload_ptr(page_data);
  }
};

}  // namespace fastlanes

#endif  // FLS_GPU_COMMON_CUH
