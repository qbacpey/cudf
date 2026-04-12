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

class FastLanesInt32Encoder {
 public:
  explicit FastLanesInt32Encoder(bool debug_print = false);

  EncodedPageResult encode_page(int32_t const* d_input,
                                uint32_t count,
                                rmm::cuda_stream_view stream);

  std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
  encode_pages(std::vector<int32_t*> const& h_gather_ptrs,
               std::vector<uint32_t> const& h_gather_counts,
               rmm::cuda_stream_view stream);

 private:
  bool debug_print_;
};

class FastLanesInt64Encoder {
 public:
  explicit FastLanesInt64Encoder(bool debug_print = false);

  EncodedPageResult encode_page(int64_t const* d_input,
                                uint32_t count,
                                rmm::cuda_stream_view stream);

  std::tuple<std::vector<rmm::device_buffer>, std::vector<uint8_t*>, std::vector<uint32_t>>
  encode_pages(std::vector<int64_t*> const& h_gather_ptrs,
               std::vector<uint32_t> const& h_gather_counts,
               rmm::cuda_stream_view stream);

 private:
  bool debug_print_;
};

}  // namespace cudf::io::parquet::detail::fastlanes_cudf
