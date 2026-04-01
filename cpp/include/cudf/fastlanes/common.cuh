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
 * @brief Encoded payload layout mode.
 */
enum class PageLayoutMode : uint8_t {
  SCALAR32 = 1,  // Single 32-bit stream (INT32 physical columns)
  SPLIT32  = 2,  // Two 32-bit component streams (INT64 physical columns)
  NATIVE64 = 3,  // Reserved for future native 64-bit path
};

/**
 * @brief Bitwidth interpretation mode for payload stream metadata.
 */
enum class BitwidthMode : uint8_t {
  SINGLE           = 1,  // Use bitwidth field
  SPLIT_COMPONENTS = 2,  // Use component_bitwidth_low/high fields
};

__device__ __host__ inline constexpr bool is_valid_layout_mode(uint8_t mode)
{
  return mode == static_cast<uint8_t>(PageLayoutMode::SCALAR32) ||
         mode == static_cast<uint8_t>(PageLayoutMode::SPLIT32) ||
         mode == static_cast<uint8_t>(PageLayoutMode::NATIVE64);
}

__device__ __host__ inline constexpr bool is_valid_layout_mode(PageLayoutMode mode)
{
  return is_valid_layout_mode(static_cast<uint8_t>(mode));
}

__device__ __host__ inline constexpr bool is_valid_bitwidth_mode(uint8_t mode)
{
  return mode == static_cast<uint8_t>(BitwidthMode::SINGLE) ||
         mode == static_cast<uint8_t>(BitwidthMode::SPLIT_COMPONENTS);
}

__device__ __host__ inline constexpr bool is_valid_bitwidth_mode(BitwidthMode mode)
{
  return is_valid_bitwidth_mode(static_cast<uint8_t>(mode));
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
 * │ 0      │ 1 byte │ bitwidth        │ Single-stream bitwidth │
 * │ 1      │ 1 byte │ layout_mode     │ PageLayoutMode enum    │
 * │ 2      │ 1 byte │ bitwidth_mode   │ BitwidthMode enum      │
 * │ 3      │ 1 byte │ reserved        │ Future use (zero)      │
 * │ 4      │ 4 bytes│ original_count  │ Actual element count   │
 * │ 8      │ 4 bytes│ padded_count    │ Padded to 1024 boundary│
 * │ 12     │ 4 bytes│ body_size       │ Encoded body size      │
 * │ 16     │ 4 bytes│ min_value_lo    │ Low-component base bits│
 * │ 20     │ 4 bytes│ min_value_hi    │ High-component base    │
 * │ 24     │ 1 byte │ bitwidth_lo     │ Split low bitwidth     │
 * │ 25     │ 1 byte │ bitwidth_hi     │ Split high bitwidth    │
 * │ 26     │102bytes│ padding         │ Zero padding           │
 * └────────┴────────┴─────────────────┴────────────────────────┘
 * │ 128    │ ...    │ PAYLOAD         │ Encoded data           │
 * └────────┴────────┴─────────────────┴────────────────────────┘
 */
struct PageHeader {
  // Header field offsets
  static constexpr size_t OFFSET_BITWIDTH            = 0;
  static constexpr size_t OFFSET_LAYOUT_MODE         = 1;
  static constexpr size_t OFFSET_BITWIDTH_MODE       = 2;
  static constexpr size_t OFFSET_RESERVED            = 3;
  static constexpr size_t OFFSET_ORIGINAL_COUNT      = 4;
  static constexpr size_t OFFSET_PADDED_COUNT        = 8;
  static constexpr size_t OFFSET_BODY_SIZE           = 12;
  static constexpr size_t OFFSET_MIN_VALUE_LOW_BITS  = 16;
  static constexpr size_t OFFSET_MIN_VALUE_HIGH_BITS = 20;
  static constexpr size_t OFFSET_COMPONENT_BW_LOW    = 24;
  static constexpr size_t OFFSET_COMPONENT_BW_HIGH   = 25;

  // Data members
  uint8_t bitwidth;
  PageLayoutMode layout_mode;
  BitwidthMode bitwidth_mode;
  uint32_t original_count;
  uint32_t padded_count;
  uint32_t body_size;
  uint32_t min_value_low_bits;
  uint32_t min_value_high_bits;
  uint8_t component_bitwidth_low;
  uint8_t component_bitwidth_high;

  /**
   * @brief Get total header size (128 bytes, aligned)
   */
  __device__ __host__ static constexpr size_t header_size() { return PAYLOAD_ALIGNMENT; }

  // Backward compat alias
  __device__ __host__  static constexpr size_t padded_size() { return header_size(); }

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
                                                 size_t body_sz)
  {
    if (!is_valid_bitwidth(bw)) {
      throw std::invalid_argument("FastLanes: bitwidth must be in [1, 32]");
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
    blob[OFFSET_BITWIDTH]      = bw;
    blob[OFFSET_LAYOUT_MODE]   = static_cast<uint8_t>(PageLayoutMode::SCALAR32);
    blob[OFFSET_BITWIDTH_MODE] = static_cast<uint8_t>(BitwidthMode::SINGLE);
    blob[OFFSET_COMPONENT_BW_LOW]  = bw;
    blob[OFFSET_COMPONENT_BW_HIGH] = 0;

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
                                                size_t body_sz)
  {
    if (!is_valid_bitwidth(bw_low) || !is_valid_bitwidth(bw_high)) {
      throw std::invalid_argument("FastLanes: split32 bitwidths must be in [1, 32]");
    }
    if (pad_count < orig_count) {
      throw std::invalid_argument("FastLanes: padded_count must be >= original_count");
    }
    if (pad_count % VECTOR_SIZE != 0 && pad_count != 0) {
      throw std::invalid_argument("FastLanes: padded_count must be multiple of 1024");
    }

    std::vector<uint8_t> blob(header_size() + body_sz);
    std::memset(blob.data(), 0, header_size());

    blob[OFFSET_BITWIDTH]          = 0;
    blob[OFFSET_LAYOUT_MODE]       = static_cast<uint8_t>(PageLayoutMode::SPLIT32);
    blob[OFFSET_BITWIDTH_MODE]     = static_cast<uint8_t>(BitwidthMode::SPLIT_COMPONENTS);
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
   * @brief Backward-compatible scalar32 serialize wrapper.
   */
  static std::vector<uint8_t> serialize(uint8_t bw,
                                        TypeCastMode,
                                        uint32_t orig_count,
                                        uint32_t pad_count,
                                        uint64_t min_val,
                                        const uint8_t* encoded_body,
                                        size_t body_sz)
  {
    return serialize_scalar32(
      bw, orig_count, pad_count, static_cast<uint32_t>(min_val), encoded_body, body_sz);
  }

  /**
   * @brief Deserialize header from raw bytes (device/host safe)
   * @param page_data Pointer to start of page data (header)
   * @return Populated PageHeader struct
   */
  __device__ __host__ static PageHeader deserialize(const uint8_t* page_data)
  {
    PageHeader h;
    h.bitwidth = page_data[OFFSET_BITWIDTH];

    auto const raw_layout_mode = page_data[OFFSET_LAYOUT_MODE];
    h.layout_mode = is_valid_layout_mode(raw_layout_mode)
                      ? static_cast<PageLayoutMode>(raw_layout_mode)
                      : PageLayoutMode::SCALAR32;

    auto const raw_bitwidth_mode = page_data[OFFSET_BITWIDTH_MODE];
    h.bitwidth_mode = is_valid_bitwidth_mode(raw_bitwidth_mode)
                        ? static_cast<BitwidthMode>(raw_bitwidth_mode)
                        : BitwidthMode::SINGLE;

    h.component_bitwidth_low  = page_data[OFFSET_COMPONENT_BW_LOW];
    h.component_bitwidth_high = page_data[OFFSET_COMPONENT_BW_HIGH];

    if (h.component_bitwidth_low == 0) { h.component_bitwidth_low = h.bitwidth; }

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

    if (h.layout_mode == PageLayoutMode::SCALAR32 && h.component_bitwidth_low == 0) {
      h.component_bitwidth_low = h.bitwidth;
    }

    return h;
  }

  __device__ __host__ constexpr uint64_t min_value_bits() const
  {
    return (static_cast<uint64_t>(min_value_high_bits) << 32) |
           static_cast<uint64_t>(min_value_low_bits);
  }

  __device__ __host__ constexpr bool is_scalar32_layout() const
  {
    return layout_mode == PageLayoutMode::SCALAR32 && bitwidth_mode == BitwidthMode::SINGLE;
  }

  __device__ __host__ constexpr bool is_split32_layout() const
  {
    return layout_mode == PageLayoutMode::SPLIT32 &&
           bitwidth_mode == BitwidthMode::SPLIT_COMPONENTS;
  }

  __device__ __host__ constexpr bool has_valid_common_metadata() const
  {
    if (padded_count < original_count) { return false; }
    if (padded_count != 0 && (padded_count % VECTOR_SIZE) != 0) { return false; }

    if (!is_valid_layout_mode(layout_mode) || !is_valid_bitwidth_mode(bitwidth_mode)) {
      return false;
    }

    if (is_scalar32_layout()) {
      return is_valid_bitwidth(bitwidth) &&
             (component_bitwidth_low == 0 || component_bitwidth_low == bitwidth) &&
             component_bitwidth_high == 0;
    }

    if (is_split32_layout()) {
      return is_valid_bitwidth(component_bitwidth_low) &&
             is_valid_bitwidth(component_bitwidth_high);
    }

    return false;
  }

  __device__ __host__ constexpr size_t expected_body_size_bytes() const
  {
    if (padded_count == 0) { return 0; }
    if (is_scalar32_layout()) {
      return encoded_size_bytes(padded_count, bitwidth);
    }
    if (is_split32_layout()) {
      return encoded_size_bytes(padded_count, component_bitwidth_low) +
             encoded_size_bytes(padded_count, component_bitwidth_high);
    }
    return 0;
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

/**
 * @brief Validate a decoded FastLanes header for the target physical type.
 */
__device__ __host__ inline constexpr bool is_valid_for_physical(PageHeader const& header,
                                                                 bool is_int64_physical)
{
  if (!header.has_valid_common_metadata()) { return false; }

  auto const expected_body = header.expected_body_size_bytes();
  if (header.body_size < expected_body) { return false; }

  // TODO(native64): when native 64-bit payload layout is added, extend this INT64 branch
  // to allow PageLayoutMode::NATIVE64 with its own metadata/body-size validation path.
  if (is_int64_physical) { return header.is_split32_layout(); }
  return header.is_scalar32_layout();
}

}  // namespace fastlanes

#endif  // FLS_GPU_COMMON_CUH
