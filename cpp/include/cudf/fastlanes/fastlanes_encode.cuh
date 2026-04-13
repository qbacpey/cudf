#pragma once

#include <cudf/fastlanes/common.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <tuple>
#include <vector>

namespace cudf::io::parquet::detail::fastlanes_cudf {

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
  uint64_t min_value;              // Raw page-local minimum value

  uint8_t* device_ptr() { return static_cast<uint8_t*>(device_blob.data()); }
  [[nodiscard]] size_t total_size() const { return host_blob.size(); }
};

/**
 * @brief CPU FastLanes encoder for INT32 payloads using RAW32 page layout.
 */
class FastLanesInt32Encoder {
 public:
  /**
   * @brief Construct an INT32 FastLanes encoder.
   *
   * @param debug_print Enable verbose debug output for encoded page metadata.
   */
  explicit FastLanesInt32Encoder(bool debug_print = false);

  /**
   * @brief Encode one INT32 page into a FastLanes RAW32 payload.
   *
   * @param d_input Device pointer to page values.
   * @param count Number of values in the page.
   * @param stream CUDA stream used for staging and uploads.
   * @return Encoded page blob and metadata.
   */
  EncodedPageResult encode_page(int32_t const* d_input,
                                uint32_t count,
                                rmm::cuda_stream_view stream);

  /**
   * @brief Encode a batch of INT32 pages into FastLanes RAW32 payloads.
   *
   * @param h_gather_ptrs Host list of device page pointers.
   * @param h_gather_counts Value counts for each page pointer.
   * @param stream CUDA stream used for staging and uploads.
   * @return Uploaded buffers, per-page device pointers, and per-page encoded sizes.
   */
  std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
  encode_pages(std::vector<int32_t*> const& h_gather_ptrs,
               std::vector<uint32_t> const& h_gather_counts,
               rmm::cuda_stream_view stream);

 private:
  bool debug_print_;
};

/**
 * @brief CPU FastLanes encoder for INT64 payloads using SPLIT64 page layout.
 */
class FastLanesInt64Split32Encoder {
 public:
  /**
   * @brief Construct an INT64 SPLIT64 FastLanes encoder.
   *
   * @param debug_print Enable verbose debug output for encoded page metadata.
   */
  explicit FastLanesInt64Split32Encoder(bool debug_print = false);

  /**
   * @brief Encode one INT64 page into a FastLanes SPLIT64 payload.
   *
   * @param d_input Device pointer to page values.
   * @param count Number of values in the page.
   * @param stream CUDA stream used for staging and uploads.
   * @return Encoded page blob and metadata.
   */
  EncodedPageResult encode_page(int64_t const* d_input,
                                uint32_t count,
                                rmm::cuda_stream_view stream);

  /**
   * @brief Encode a batch of INT64 pages into FastLanes SPLIT64 payloads.
   *
   * @param h_gather_ptrs Host list of device page pointers.
   * @param h_gather_counts Value counts for each page pointer.
   * @param stream CUDA stream used for staging and uploads.
   * @return Uploaded buffers, per-page device pointers, and per-page encoded sizes.
   */
  std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
  encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
               std::vector<uint32_t> const& h_gather_counts,
               rmm::cuda_stream_view stream);

 private:
  bool debug_print_;
};

/**
 * @brief Native64 FastLanes encoder entry point for INT64 payloads.
 *
 * Native64 behavior is introduced in a staged rollout. In non-activation stages,
 * this class can surface explicit not-yet-enabled behavior instead of silently
 * falling back to SPLIT64.
 */
class FastLanesInt64NativeEncoder {
 public:
  /**
   * @brief Construct an INT64 Native64 FastLanes encoder.
   *
   * @param debug_print Enable verbose debug output for encoded page metadata.
   */
  explicit FastLanesInt64NativeEncoder(bool debug_print = false);

  /**
   * @brief Encode one INT64 page into a FastLanes Native64 payload.
   *
   * @param d_input Device pointer to page values.
   * @param count Number of values in the page.
   * @param stream CUDA stream used for staging and uploads.
   * @return Encoded page blob and metadata.
   */
  EncodedPageResult encode_page(int64_t const* d_input,
                                uint32_t count,
                                rmm::cuda_stream_view stream);

  /**
   * @brief Encode a batch of INT64 pages into FastLanes Native64 payloads.
   *
   * @param h_gather_ptrs Host list of device page pointers.
   * @param h_gather_counts Value counts for each page pointer.
   * @param stream CUDA stream used for staging and uploads.
   * @return Uploaded buffers, per-page device pointers, and per-page encoded sizes.
   */
  std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
  encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
               std::vector<uint32_t> const& h_gather_counts,
               rmm::cuda_stream_view stream);

 private:
  bool debug_print_;
};

}  // namespace cudf::io::parquet::detail::fastlanes_cudf
