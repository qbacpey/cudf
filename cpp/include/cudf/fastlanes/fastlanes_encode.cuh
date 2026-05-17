#pragma once

#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/debug.hpp>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cudf::io::parquet::detail::fastlanes_cudf {

// =============================================================================
// Debug Utilities — Delegates to centralized fastlanes::debug (debug.hpp)
// =============================================================================

/// @deprecated Use fastlanes::debug::print_encoded_dump() directly.
template <typename T>
void print_encoded_dump(const T* data, size_t count, const char* label = "Encoded Dump")
{
  fastlanes::debug::print_encoded_dump(data, count, label);
}

/// @deprecated Use fastlanes::debug::print_encoded_dump() directly.
template <typename T>
void print_encoded_dump(const std::vector<T>& data, const char* label = "Encoded Dump")
{
  fastlanes::debug::print_encoded_dump(data, label);
}

// =============================================================================
// FastLanes Encoder Result
// =============================================================================

/**
 * @brief Result of encoding a single page
 */
struct EncodedPageResult {
  std::vector<uint8_t> host_blob;  // Complete blob: [Header][Encoded Body]
  rmm::device_buffer device_blob;  // Uploaded to GPU
  uint8_t bitwidth;                // Bits per value used
  fastlanes::TypeCastMode cast_mode;  // How type was handled
  uint32_t original_count;         // Original element count
  uint32_t padded_count;           // Padded element count
  size_t body_size;                // Encoded body size in bytes
  uint32_t min_value;              // Raw page-local minimum value

  uint8_t* device_ptr() { return static_cast<uint8_t*>(device_blob.data()); }
  [[nodiscard]] size_t total_size() const { return host_blob.size(); }
};

// =============================================================================
// FastLanes Encoder Class (Template)
// =============================================================================

/**
 * @brief FastLanes bit-packing encoder for 32-bit integers.
 *
 * This class encapsulates FastLanes encoding logic for signed 32-bit input:
 * - int32_t with all values >= 0: Encoded as normalized signed deltas (SIGNED_SAFE mode)
 * - int32_t with negative values: Encoded as normalized signed deltas (SIGNED_REINTERPRET mode)
 *
 * @tparam T Input type, must be int32_t
 */
template <typename T>
class FastLanesEncoder {
  static_assert(std::is_same_v<T, int32_t>, "FastLanesEncoder only supports int32_t");

  // Unsigned type used for internal encoding
  using UnsignedT = std::make_unsigned_t<T>;
  static constexpr uint8_t TYPE_BITS = sizeof(T) * 8;
  struct NormalizedPageData {
    std::vector<UnsignedT> values;
    uint8_t bitwidth;
    uint32_t min_value;
  };

 public:
  /**
   * @brief Construct encoder with optional debug output
   * @param debug_print If true, print debug information during encoding
   */
  explicit FastLanesEncoder(bool debug_print = false) : debug_print_(debug_print) {}

  /**
   * @brief Encode a page of data from GPU memory
   *
   * Performs complete workflow: download → analyze → encode → wrap header → upload
   *
   * @param d_input Device pointer to input data
   * @param count Number of elements
   * @param stream CUDA stream for operations
   * @return EncodedPageResult containing device buffer and metadata
   * @throws std::runtime_error on CUDA errors
   * @throws std::invalid_argument on invalid parameters
   */
  EncodedPageResult encode_page(const T* d_input, uint32_t count, rmm::cuda_stream_view stream)
  {
    EncodedPageResult result;

    if (count == 0) { return create_empty_result(stream); }

    if (d_input == nullptr) {
      throw std::invalid_argument("FastLanesEncoder: device input pointer is null");
    }

    // 1. Download data from GPU with padding
    uint64_t padded = fastlanes::padded_count(count);
    std::vector<T> host_input(padded, T{0});

    cuda_check(
      cudaMemcpyAsync(
        host_input.data(), d_input, count * sizeof(T), cudaMemcpyDeviceToHost, stream.value()),
      "download");
    cudaStreamSynchronize(stream.value());

    // 2. Analyze data and determine cast mode
    auto const cast_mode = analyze_data(host_input.data(), count).first;

    // 3. Normalize to page-local deltas before bitwidth selection
    auto normalized = normalize_page_data(host_input.data(), count, static_cast<uint32_t>(padded));
    auto const bitwidth = normalized.bitwidth;

    // 4. Encode using FastLanes (always works on unsigned type internally)
    size_t encoded_bytes = fastlanes::encoded_size_bytes(padded, bitwidth);
    std::vector<UnsignedT> encoded_body(encoded_bytes / sizeof(UnsignedT));

    encode_vectors(normalized.values.data(), padded, bitwidth, encoded_body.data());

    if (debug_print_) {
      using namespace fastlanes::debug;
      PageDebugInfo info = make_debug_info(bitwidth,
                                           cast_mode,
                                           count,
                                           static_cast<uint32_t>(padded),
                                           static_cast<uint32_t>(encoded_bytes),
                                           normalized.min_value,
                                           encoded_body.data(),
                                           encoded_body.size());
      print_page_debug(std::cout, info);
    }

    // 5. Serialize with header
    result.host_blob =
      fastlanes::PageHeader::serialize(bitwidth,
                                       cast_mode,
                                       count,
                                       static_cast<uint32_t>(padded),
                                       normalized.min_value,
                                       reinterpret_cast<const uint8_t*>(encoded_body.data()),
                                       encoded_bytes);

    // 6. Upload to GPU
    result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
    cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                               result.host_blob.data(),
                               result.host_blob.size(),
                               cudaMemcpyHostToDevice,
                               stream.value()),
               "upload");

    // Fill metadata
    result.bitwidth       = bitwidth;
    result.cast_mode      = cast_mode;
    result.original_count = count;
    result.padded_count   = static_cast<uint32_t>(padded);
    result.body_size      = encoded_bytes;
    result.min_value      = normalized.min_value;

    return result;
  }

  /**
   * @brief Encode multiple pages (EncodePages-compatible batch interface)
   *
   * @param h_gather_ptrs Host vector of device pointers to gathered page data
   * @param h_gather_counts Host vector of element counts per page
   * @param stream CUDA stream
   * @return Tuple of (device buffers, host pointer array, host size array)
   */
  std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
  encode_pages(const std::vector<T*>& h_gather_ptrs,
               const std::vector<uint32_t>& h_gather_counts,
               rmm::cuda_stream_view stream)
  {
    size_t num_pages = h_gather_ptrs.size();

    std::vector<rmm::device_buffer> encoded_buffers;
    std::vector<uint8_t*> h_upload_ptrs(num_pages, nullptr);
    std::vector<uint32_t> h_upload_sizes(num_pages, 0);
    encoded_buffers.reserve(num_pages);

    for (size_t i = 0; i < num_pages; ++i) {
      if (h_gather_ptrs[i] == nullptr) continue;

      uint32_t count = h_gather_counts[i];
      if (count == 0) continue;

      auto page = encode_page(h_gather_ptrs[i], count, stream);

      encoded_buffers.push_back(std::move(page.device_blob));
      h_upload_ptrs[i]  = static_cast<uint8_t*>(encoded_buffers.back().data());
      h_upload_sizes[i] = static_cast<uint32_t>(page.total_size());
    }

    return {std::move(encoded_buffers), std::move(h_upload_ptrs), std::move(h_upload_sizes)};
  }

 private:
  bool debug_print_;

  // ---------------------------------------------------------------------------
  // Private: Data Analysis
  // ---------------------------------------------------------------------------

  /**
   * @brief Analyze data to determine cast mode and presence of negative values
   */
  std::pair<fastlanes::TypeCastMode, bool> analyze_data(const T* data, uint32_t count)
  {
    bool has_negative = false;
    for (uint32_t i = 0; i < count; ++i) {
      if (data[i] < 0) {
        has_negative = true;
        break;
      }
    }
    return {has_negative ? fastlanes::TypeCastMode::SIGNED_REINTERPRET
                         : fastlanes::TypeCastMode::SIGNED_SAFE,
            has_negative};
  }

  /**
   * @brief Compute page-local normalization and the resulting bitwidth.
   */
  NormalizedPageData normalize_page_data(const T* data,
                                         uint32_t count,
                                         uint32_t padded_count)
  {
    NormalizedPageData page{};
    page.values.assign(padded_count, UnsignedT{0});

    auto min_val = std::numeric_limits<int32_t>::max();
    for (uint32_t i = 0; i < count; ++i) {
      min_val = std::min(min_val, static_cast<int32_t>(data[i]));
    }
    page.min_value = fastlanes::int32_to_u32_bits(min_val);

    auto const signed_min = static_cast<int64_t>(min_val);
    for (uint32_t i = 0; i < count; ++i) {
      auto const delta = static_cast<int64_t>(static_cast<int32_t>(data[i])) - signed_min;
      page.values[i]   = static_cast<UnsignedT>(delta);
    }

    page.bitwidth = compute_bitwidth(page.values.data(), count);
    if (page.bitwidth == TYPE_BITS) {
      // Trade-off: the intended datasets are expected to compress after page-local normalization.
      // Rather than introducing per-page fallback machinery in the write path, reject pages that
      // still need 32 bits and let the caller fail fast.
      throw std::invalid_argument("FastLanesEncoder: normalized page still requires 32-bit width");
    }

    return page;
  }

  // ---------------------------------------------------------------------------
  // Private: Encoding
  // ---------------------------------------------------------------------------

  /**
   * @brief Encode all vectors using FastLanes pack function
   */
  void encode_vectors(const UnsignedT* input, uint64_t padded, uint8_t bitwidth, UnsignedT* output)
  {
    uint64_t n_vectors = fastlanes::num_vectors(padded);

    // FastLanes pack always operates on unsigned type
    const auto* in_ptr = input;
    UnsignedT* out_ptr = output;

    // Output elements per vector = (1024 * bitwidth) / TYPE_BITS
    size_t out_elements_per_vector = (fastlanes::VECTOR_SIZE * bitwidth) / TYPE_BITS;

    for (uint64_t v = 0; v < n_vectors; ++v) {
      generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bitwidth);
      in_ptr += fastlanes::VECTOR_SIZE;
      out_ptr += out_elements_per_vector;
    }
  }

  // ---------------------------------------------------------------------------
  // Private: Utilities
  // ---------------------------------------------------------------------------

  EncodedPageResult create_empty_result(rmm::cuda_stream_view stream)
  {
    EncodedPageResult result;
    result.bitwidth       = 1;
    result.cast_mode      = fastlanes::TypeCastMode::SIGNED_SAFE;
    result.original_count = 0;
    result.padded_count   = 0;
    result.body_size      = 0;
    result.min_value      = 0;

    // Create minimal blob with just header (no body)
    result.host_blob =
      fastlanes::PageHeader::serialize(1, result.cast_mode, 0, 0, 0, nullptr, 0);

    result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
    cudaMemcpyAsync(result.device_blob.data(),
                    result.host_blob.data(),
                    result.host_blob.size(),
                    cudaMemcpyHostToDevice,
                    stream.value());

    return result;
  }

  void cuda_check(cudaError_t err, const char* operation)
  {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("FastLanesEncoder: CUDA ") + operation +
                               " failed: " + cudaGetErrorString(err));
    }
  }

  /// @deprecated Replaced by fastlanes::debug::print_encoder_summary()
  void print_debug_info(uint32_t count,
                        uint64_t padded,
                        uint8_t bitwidth,
                        fastlanes::TypeCastMode cast_mode,
                        const T* data)
  {
    fastlanes::debug::print_encoder_summary(std::cout, count, padded, bitwidth, cast_mode, data);
  }

  uint8_t compute_bitwidth(const UnsignedT* data, uint32_t count)
  {
    UnsignedT max_val = 0;
    for (uint32_t i = 0; i < count; ++i) {
      if (data[i] > max_val) { max_val = data[i]; }
    }

    if (max_val == 0) { return 1; }

    uint8_t bits = 0;
    while (max_val > 0) {
      max_val >>= 1;
      bits++;
    }
    return bits;
  }
};

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

using FastLanesInt32Encoder  = FastLanesEncoder<int32_t>;
// using FastLanesInt64Encoder  = FastLanesEncoder<int64_t>;
// using FastLanesUInt64Encoder = FastLanesEncoder<uint64_t>;

}  // namespace cudf::io::parquet::detail
