/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// FastLanes decode kernel.
//
// Design constraints for this first integration:
// - Exactly one warp (32 threads) per block.
// - One block per page.
// - Flat, non-null columns only (no nesting/null handling in-kernel).
// - INT32 and INT64 payload pages.
//   - INT32 output can be 8/16/32-bit logical INT classes or TIME_MILLIS as 64-bit.
//   - INT64 output is currently 64-bit logical INT only.
// - One FastLanes vector (1024 values) decoded at a time.
//
// Trade-off note:
// This path intentionally favors a simple implementation for the expected workload over
// broader type/nesting coverage. Unsupported shapes fail fast instead of adding more
// metadata plumbing or generalized decode logic.

#include "page_decode.cuh"
#include "parquet_gpu.hpp"
#include "fastlanes_parquet_common.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/fls_gen/unpack/unpack.cuh>

#include <cooperative_groups.h>

#include <string>

namespace native64_generated {
[[nodiscard]] std::string cuda_error(cudaError_t status, char const* operation);
}  // namespace native64_generated
// Native64 decode needs generated runtime device dispatch. Keep this include scoped to the
// FastLanes decode translation unit to avoid exposing generated internals broadly.
#include <cudf/fastlanes/native64_cuda_kernels.inl>

namespace cudf::io::parquet::detail {

namespace {

namespace cg = cooperative_groups;

constexpr int decode_fastlanes_block_size = 32;  // exactly one warp
constexpr int fastlanes_vector_size       = 1024;

template <typename level_t>
__device__ inline bool setup_and_validate_fastlanes_page(page_state_s* s,
                                                         PageInfo* pages,
                                                         int page_idx,
                                                         device_span<ColumnChunkDesc const> chunks,
                                                         size_t min_row,
                                                         size_t num_rows,
                                                         cudf::device_span<bool const> page_mask,
                                                         Type expected_physical_type,
                                                         Encoding expected_encoding,
                                                         decode_kernel_mask expected_kernel_mask,
                                                         int lane,
                                                         kernel_error::pointer error_code,
                                                         fastlanes::PageHeader* fastlanes_header,
                                                         uint32_t* total_value_count)
{
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{expected_kernel_mask},
                             page_processing_stage::DECODE)) {
    return false;
  }

  if (s->col.physical_type != expected_physical_type) { return false; }

  auto const has_repetition_levels = (s->col.max_level[level_type::REPETITION] > 0);

  // Keep pruned-page behavior aligned with other decode kernels.
  if (not page_mask[page_idx]) {
    auto& page = pages[page_idx];
    if (has_repetition_levels) {
      update_list_offsets_for_pruned_pages<decode_fastlanes_block_size>(s);
    }
    page.num_nulls = page.nesting[s->col.max_nesting_depth - 1].batch_size;
    page.num_nulls -= has_repetition_levels ? 0 : s->first_row;
    page.num_valids = 0;
    return false;
  }

  // Current implementation target: flat columns only.
  if (has_repetition_levels || s->col.max_nesting_depth != 1) {
    return fastlanes_set_decode_error(lane, error_code, decode_error::UNSUPPORTED_ENCODING);
  }

  // Nullable schema is allowed only when this page actually has zero nulls.
  // For optional Parquet fields with all rows populated, page.num_nulls is 0,
  // so this path is valid even though max definition level > 0.
  if (s->page.num_nulls > 0) {
    return fastlanes_set_decode_error(lane, error_code, decode_error::UNSUPPORTED_ENCODING);
  }

  // Current FastLanes integration supports 8/16/32-bit logical outputs and TIME_MILLIS
  // outputs materialized as 64-bit durations.
  if (!fastlanes_is_supported_dtype_len(s->dtype_len)) {
    return fastlanes_set_decode_error(lane, error_code, decode_error::INVALID_DATA_TYPE);
  }

  if (expected_physical_type == Type::INT64 && s->dtype_len != 8) {
    return fastlanes_set_decode_error(lane, error_code, decode_error::INVALID_DATA_TYPE);
  }

  if (s->page.encoding != expected_encoding) { return false; }

  auto const header = fastlanes::PageHeader::deserialize(s->data_start);

  if (!fastlanes_is_valid_header_for_encoding(header, expected_encoding)) {
    return fastlanes_set_decode_error(lane, error_code, decode_error::INVALID_DATA_TYPE);
  }

  *fastlanes_header = header;
  *total_value_count =
    header.original_count < s->num_input_values ? header.original_count : s->num_input_values;
  return true;
}

// ---------------------------------------------------------------------------
// INT32 Kernel: one thread-block (one warp) per page
// ---------------------------------------------------------------------------
namespace raw32 {

template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_block_size)
  decode_kernel(PageInfo* pages,
                device_span<ColumnChunkDesc const> chunks,
                size_t min_row,
                size_t num_rows,
                cudf::device_span<bool const> page_mask,
                kernel_error::pointer error_code)
{
  __shared__ uint32_t decoded_vec[fastlanes_vector_size];
  __shared__ uint32_t packed_vec_aligned[fastlanes_vector_size];
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  int const lane        = static_cast<int>(block.thread_rank());
  [[maybe_unused]] null_count_back_copier _{s, lane};

  fastlanes::PageHeader fastlanes_header{};
  uint32_t total_value_count = 0;
  if (!setup_and_validate_fastlanes_page<level_t>(s,
                                                  pages,
                                                  page_idx,
                                                  chunks,
                                                  min_row,
                                                  num_rows,
                                                  page_mask,
                                                  Type::INT32,
                                                  Encoding::FASTLANE_BITPACK_RAW,
                                                  decode_kernel_mask::FASTLANE_BITPACK_RAW,
                                                  lane,
                                                  error_code,
                                                  &fastlanes_header,
                                                  &total_value_count)) {
    return;
  }

  auto const* payload_bytes = fastlanes::PageHeader::payload_ptr(s->data_start);
  auto const packed_words_per_vector =
    static_cast<uint32_t>(fastlanes_header.component_bitwidth_low) * 32;
  auto const min_value_bits = fastlanes_header.min_value_low_bits;

  auto const leaf_level_idx = s->col.max_nesting_depth - 1;
  auto* const output_base_8 = reinterpret_cast<uint8_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_16 =
    reinterpret_cast<uint16_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_32 =
    reinterpret_cast<uint32_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_64 = reinterpret_cast<int64_t*>(s->nesting_info[leaf_level_idx].data_out);

  uint32_t value_base_idx = 0;
  while (value_base_idx < total_value_count) {
    auto const vector_index          = value_base_idx / fastlanes_vector_size;
    auto const remaining_value_count = total_value_count - value_base_idx;
    auto const values_in_vector =
      remaining_value_count < fastlanes_vector_size ? remaining_value_count : fastlanes_vector_size;

    auto const vector_input_bytes = payload_bytes + static_cast<size_t>(vector_index) *
                                                      static_cast<size_t>(packed_words_per_vector) *
                                                      sizeof(uint32_t);

    // Some mixed-column pages can place FastLanes payload at non-4-byte aligned addresses.
    // Stage packed words into aligned shared memory before calling generated unpack code,
    // which expects uint32_t-aligned inputs.
    for (uint32_t w = lane; w < packed_words_per_vector; w += decode_fastlanes_block_size) {
      auto const* b         = vector_input_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
      packed_vec_aligned[w] = (static_cast<uint32_t>(b[0])) | (static_cast<uint32_t>(b[1]) << 8) |
                              (static_cast<uint32_t>(b[2]) << 16) |
                              (static_cast<uint32_t>(b[3]) << 24);
    }
    block.sync();

    unpack_device(packed_vec_aligned, decoded_vec, fastlanes_header.component_bitwidth_low);
    block.sync();

    for (uint32_t i = lane; i < values_in_vector; i += decode_fastlanes_block_size) {
      auto const dst_pos    = static_cast<int32_t>(value_base_idx + i) - s->first_row;
      auto const delta_bits = decoded_vec[i];
      auto const value_bits = min_value_bits + delta_bits;

      if (dst_pos >= 0 && dst_pos < s->num_rows) {
        if (s->dtype_len == 8) {
          output_base_64[dst_pos] = static_cast<int64_t>(fastlanes::u32_bits_to_int32(value_bits));
        } else if (s->dtype_len == 4) {
          output_base_32[dst_pos] = value_bits;
        } else if (s->dtype_len == 2) {
          output_base_16[dst_pos] = static_cast<uint16_t>(value_bits);
        } else {
          output_base_8[dst_pos] = static_cast<uint8_t>(value_bits);
        }
      }
    }

    block.sync();
    value_base_idx += values_in_vector;
  }
}

}  // namespace raw32

// ---------------------------------------------------------------------------
// INT64 Kernel: one thread-block (one warp) per page
// ---------------------------------------------------------------------------
namespace split64 {

template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_block_size)
  decode_kernel(PageInfo* pages,
                device_span<ColumnChunkDesc const> chunks,
                size_t min_row,
                size_t num_rows,
                cudf::device_span<bool const> page_mask,
                kernel_error::pointer error_code)
{
  __shared__ uint32_t decoded_vec_low[fastlanes_vector_size];
  __shared__ uint32_t decoded_vec_high[fastlanes_vector_size];
  __shared__ uint32_t packed_vec_low_aligned[fastlanes_vector_size];
  __shared__ uint32_t packed_vec_high_aligned[fastlanes_vector_size];
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  int const lane        = static_cast<int>(block.thread_rank());
  [[maybe_unused]] null_count_back_copier _{s, lane};

  fastlanes::PageHeader fastlanes_header{};
  uint32_t total_value_count = 0;
  if (!setup_and_validate_fastlanes_page<level_t>(s,
                                                  pages,
                                                  page_idx,
                                                  chunks,
                                                  min_row,
                                                  num_rows,
                                                  page_mask,
                                                  Type::INT64,
                                                  Encoding::FASTLANE_BITPACK_SPLIT64,
                                                  decode_kernel_mask::FASTLANE_BITPACK_SPLIT64,
                                                  lane,
                                                  error_code,
                                                  &fastlanes_header,
                                                  &total_value_count)) {
    return;
  }

  auto const* payload_bytes = fastlanes::PageHeader::payload_ptr(s->data_start);
  auto const packed_words_per_vector_low =
    static_cast<uint32_t>(fastlanes_header.component_bitwidth_low) * 32;
  auto const packed_words_per_vector_high =
    static_cast<uint32_t>(fastlanes_header.component_bitwidth_high) * 32;
  auto const min_value_low_64  = fastlanes_header.min_value_low_bits;
  auto const min_value_high_64 = fastlanes_header.min_value_high_bits;

  auto const leaf_level_idx  = s->col.max_nesting_depth - 1;
  auto* const output_base_64 = reinterpret_cast<int64_t*>(s->nesting_info[leaf_level_idx].data_out);

  uint32_t value_base_idx = 0;
  while (value_base_idx < total_value_count) {
    auto const vector_index          = value_base_idx / fastlanes_vector_size;
    auto const remaining_value_count = total_value_count - value_base_idx;
    auto const values_in_vector =
      remaining_value_count < fastlanes_vector_size ? remaining_value_count : fastlanes_vector_size;

    auto const vector_input_bytes =
      payload_bytes +
      static_cast<size_t>(vector_index) *
        static_cast<size_t>(packed_words_per_vector_low + packed_words_per_vector_high) *
        sizeof(uint32_t);

    auto const* packed_low_bytes = vector_input_bytes;
    auto const* packed_high_bytes =
      vector_input_bytes + static_cast<size_t>(packed_words_per_vector_low) * sizeof(uint32_t);

    for (uint32_t w = lane; w < packed_words_per_vector_low; w += decode_fastlanes_block_size) {
      auto const* b = packed_low_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
      packed_vec_low_aligned[w] =
        (static_cast<uint32_t>(b[0])) | (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }
    for (uint32_t w = lane; w < packed_words_per_vector_high; w += decode_fastlanes_block_size) {
      auto const* b = packed_high_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
      packed_vec_high_aligned[w] =
        (static_cast<uint32_t>(b[0])) | (static_cast<uint32_t>(b[1]) << 8) |
        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }
    block.sync();

    unpack_device(packed_vec_low_aligned, decoded_vec_low, fastlanes_header.component_bitwidth_low);
    block.sync();
    unpack_device(
      packed_vec_high_aligned, decoded_vec_high, fastlanes_header.component_bitwidth_high);
    block.sync();

    for (uint32_t i = lane; i < values_in_vector; i += decode_fastlanes_block_size) {
      auto const dst_pos   = static_cast<int32_t>(value_base_idx + i) - s->first_row;
      auto const low_bits  = min_value_low_64 + decoded_vec_low[i];
      auto const high_bits = min_value_high_64 + decoded_vec_high[i];
      auto const value_bits =
        (static_cast<uint64_t>(high_bits) << 32) | static_cast<uint64_t>(low_bits);
      auto const signed_val = fastlanes::u64_bits_to_int64(value_bits);

      if (dst_pos >= 0 && dst_pos < s->num_rows) { output_base_64[dst_pos] = signed_val; }
    }

    block.sync();
    value_base_idx += values_in_vector;
  }
}

}  // namespace split64

// ---------------------------------------------------------------------------
// Native64 Kernel: one thread-block (one warp) per page
// ---------------------------------------------------------------------------
namespace native64 {

template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_block_size)
  decode_kernel(PageInfo* pages,
                device_span<ColumnChunkDesc const> chunks,
                size_t min_row,
                size_t num_rows,
                cudf::device_span<bool const> page_mask,
                kernel_error::pointer error_code)
{
  constexpr uint32_t native64_lanes_per_vector = native64_generated::kLanesPerVector;
  constexpr uint32_t native64_vectors_per_block =
    decode_fastlanes_block_size / native64_lanes_per_vector;
  constexpr uint32_t native64_values_per_block = native64_vectors_per_block * fastlanes_vector_size;
  static_assert((decode_fastlanes_block_size % native64_lanes_per_vector) == 0,
                "decode block size must be a multiple of native64 lanes per vector");
  static_assert(native64_vectors_per_block == native64_generated::kVectorsPerBlock,
                "native64 decode kernel launch geometry mismatch");

  __shared__ uint64_t decoded_vec[fastlanes_vector_size * native64_vectors_per_block];
  __shared__ uint64_t packed_vec_aligned[fastlanes_vector_size * native64_vectors_per_block];
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  int const lane        = static_cast<int>(block.thread_rank());
  [[maybe_unused]] null_count_back_copier _{s, lane};

  fastlanes::PageHeader fastlanes_header{};
  uint32_t total_value_count = 0;
  if (!setup_and_validate_fastlanes_page<level_t>(s,
                                                  pages,
                                                  page_idx,
                                                  chunks,
                                                  min_row,
                                                  num_rows,
                                                  page_mask,
                                                  Type::INT64,
                                                  Encoding::FASTLANES_DELTA_BINARY,
                                                  decode_kernel_mask::FASTLANES_DELTA_BINARY,
                                                  lane,
                                                  error_code,
                                                  &fastlanes_header,
                                                  &total_value_count)) {
    return;
  }

  auto const* payload_bytes          = fastlanes::PageHeader::payload_ptr(s->data_start);
  auto const packed_words_per_vector = static_cast<uint32_t>(
    fastlanes::encoded_size_bytes(fastlanes_vector_size, fastlanes_header.component_bitwidth_low) /
    sizeof(uint64_t));
  if (packed_words_per_vector > fastlanes_vector_size) {
    fastlanes_set_decode_error(lane, error_code, decode_error::INVALID_DATA_TYPE);
    return;
  }
  auto const base_bits = fastlanes_header.min_value_bits();

  auto const leaf_level_idx  = s->col.max_nesting_depth - 1;
  auto* const output_base_64 = reinterpret_cast<int64_t*>(s->nesting_info[leaf_level_idx].data_out);

  auto const lane_u32 = static_cast<uint32_t>(lane);
  auto const subvec   = lane_u32 / native64_lanes_per_vector;
  auto const sublane  = lane_u32 % native64_lanes_per_vector;

  uint32_t value_base_idx = 0;
  while (value_base_idx < total_value_count) {
    auto const remaining_value_count = total_value_count - value_base_idx;
    auto const values_in_block       = remaining_value_count < native64_values_per_block
                                         ? remaining_value_count
                                         : native64_values_per_block;
    auto const vectors_in_block =
      (values_in_block + fastlanes_vector_size - 1) / fastlanes_vector_size;
    auto const block_vector_base = value_base_idx / fastlanes_vector_size;

    if (subvec < vectors_in_block) {
      auto const vector_index = block_vector_base + subvec;
      auto const* vector_input_bytes =
        payload_bytes + static_cast<size_t>(vector_index) *
                          static_cast<size_t>(packed_words_per_vector) * sizeof(uint64_t);
      auto* packed_words = packed_vec_aligned + static_cast<size_t>(subvec) * fastlanes_vector_size;

      for (uint32_t w = sublane; w < packed_words_per_vector; w += native64_lanes_per_vector) {
        auto const* b = vector_input_bytes + static_cast<size_t>(w) * sizeof(uint64_t);
        packed_words[w] =
          (static_cast<uint64_t>(b[0])) | (static_cast<uint64_t>(b[1]) << 8) |
          (static_cast<uint64_t>(b[2]) << 16) | (static_cast<uint64_t>(b[3]) << 24) |
          (static_cast<uint64_t>(b[4]) << 32) | (static_cast<uint64_t>(b[5]) << 40) |
          (static_cast<uint64_t>(b[6]) << 48) | (static_cast<uint64_t>(b[7]) << 56);
      }
    }
    block.sync();

    if (subvec < vectors_in_block) {
      auto const* packed_words =
        packed_vec_aligned + static_cast<size_t>(subvec) * fastlanes_vector_size;
      auto* decoded_words = decoded_vec + static_cast<size_t>(subvec) * fastlanes_vector_size;
      native64_generated::decode_lane_by_bw_device_runtime(
        fastlanes_header.component_bitwidth_low, packed_words, decoded_words, sublane, base_bits);
    }
    block.sync();

    for (uint32_t i = lane_u32; i < values_in_block; i += decode_fastlanes_block_size) {
      auto const vec_idx    = i / fastlanes_vector_size;
      auto const intra_idx  = i % fastlanes_vector_size;
      auto const decoded_i  = static_cast<size_t>(vec_idx) * fastlanes_vector_size + intra_idx;
      auto const dst_pos    = static_cast<int32_t>(value_base_idx + i) - s->first_row;
      auto const signed_val = fastlanes::u64_bits_to_int64(decoded_vec[decoded_i]);

      if (dst_pos >= 0 && dst_pos < s->num_rows) { output_base_64[dst_pos] = signed_val; }
    }

    block.sync();
    value_base_idx += values_in_block;
  }
}

}  // namespace native64

}  // anonymous namespace

// =============================================================================
// Host-side launch wrapper
// =============================================================================

void decode_fastlanes_raw32(cudf::detail::hostdevice_span<PageInfo> pages,
                            cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                            size_t num_rows,
                            size_t min_row,
                            int level_type_size,
                            cudf::device_span<bool const> page_mask,
                            kernel_error::pointer error_code,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_fastlanes_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);
  if (level_type_size == 1) {
    raw32::decode_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  } else {
    raw32::decode_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  }
}

void decode_fastlanes_split64(cudf::detail::hostdevice_span<PageInfo> pages,
                              cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                              size_t num_rows,
                              size_t min_row,
                              int level_type_size,
                              cudf::device_span<bool const> page_mask,
                              kernel_error::pointer error_code,
                              rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_fastlanes_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);
  if (level_type_size == 1) {
    split64::decode_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  } else {
    split64::decode_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  }
}

void decode_fastlanes_native64(cudf::detail::hostdevice_span<PageInfo> pages,
                               cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                               size_t num_rows,
                               size_t min_row,
                               int level_type_size,
                               cudf::device_span<bool const> page_mask,
                               kernel_error::pointer error_code,
                               rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_fastlanes_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);
  if (level_type_size == 1) {
    native64::decode_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  } else {
    native64::decode_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  }
}

}  // namespace cudf::io::parquet::detail
