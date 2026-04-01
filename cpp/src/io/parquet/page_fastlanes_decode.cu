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

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/debug.hpp>
#include <cudf/fastlanes/fls_gen/unpack/unpack.cuh>

#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>

#include <iostream>

namespace cudf::io::parquet::detail {

namespace {

namespace cg = cooperative_groups;

constexpr int decode_fastlanes_block_size = 32;  // exactly one warp
constexpr int fastlanes_vector_size       = 1024;

constexpr int decode_fastlanes_debug_block_size = 96;  // keep old debug behavior
using fastlanes_debug_info                    = fastlanes::debug::PageDebugInfo;

__device__ inline bool fastlanes_set_decode_error(int lane,
                                                   kernel_error::pointer error_code,
                                                   decode_error code)
{
  if (lane == 0) {
    set_error(static_cast<kernel_error::value_type>(code), error_code);
  }
  return false;
}

__device__ inline bool fastlanes_is_supported_dtype_len(uint32_t dtype_len)
{
  return dtype_len == 1 || dtype_len == 2 || dtype_len == 4 || dtype_len == 8;
}

template <typename level_t>
__device__ inline bool setup_and_validate_fastlanes_page(
  page_state_s* s,
  PageInfo* pages,
  int page_idx,
  device_span<ColumnChunkDesc const> chunks,
  size_t min_row,
  size_t num_rows,
  cudf::device_span<bool const> page_mask,
  Type expected_physical_type,
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
                             mask_filter{decode_kernel_mask::FASTLANES_BINARY},
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

  auto const header = fastlanes::PageHeader::deserialize(s->data_start);
  if (!fastlanes::is_valid_for_physical(header, expected_physical_type == Type::INT64)) {
    return fastlanes_set_decode_error(lane, error_code, decode_error::INVALID_DATA_TYPE);
  }

  *fastlanes_header   = header;
  *total_value_count  = header.original_count < s->num_input_values ? header.original_count
                                                                     : s->num_input_values;
  return true;
}

// ---------------------------------------------------------------------------
// Debug kernel: one thread-block per page, reads header and payload preview
// ---------------------------------------------------------------------------
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_debug_block_size)
  decode_fastlanes_debug_kernel(PageInfo* pages,
                                device_span<ColumnChunkDesc const> chunks,
                                size_t min_row,
                                size_t num_rows,
                                cudf::device_span<bool const> page_mask,
                                fastlanes_debug_info* debug_out,
                                kernel_error::pointer error_code)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();

  fastlanes_debug_info* info = &debug_out[page_idx];

  // Zero the output struct (all threads cooperate)
  {
    auto* raw = reinterpret_cast<uint8_t*>(info);
    for (int i = block.thread_rank(); i < static_cast<int>(sizeof(fastlanes_debug_info));
         i += block.size()) {
      raw[i] = 0;
    }
  }
  block.sync();

  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::FASTLANES_BINARY},
                             page_processing_stage::DECODE)) {
    if (block.thread_rank() == 0) { info->page_valid = false; }
    return;
  }

  if (block.thread_rank() == 0) {
    info->page_valid    = true;
    info->has_cudf_info = true;

    info->num_input_values    = s->num_input_values;
    info->first_row           = s->first_row;
    info->num_rows            = s->num_rows;
    info->dtype_len           = s->dtype_len;
    info->dtype_len_in        = s->dtype_len_in;
    info->data_size           = s->data_end - s->data_start;
    info->has_repetition      = s->col.max_level[level_type::REPETITION] > 0;
    info->max_nesting_depth   = s->col.max_nesting_depth;
    info->skipped_leaf_values = s->page.skipped_leaf_values;

    auto const fl_hdr   = fastlanes::PageHeader::deserialize(s->data_start);
    info->bitwidth       = fl_hdr.bitwidth;
    info->cast_mode      = static_cast<uint8_t>(fl_hdr.layout_mode);
    info->layout_mode    = static_cast<uint8_t>(fl_hdr.layout_mode);
    info->bitwidth_mode  = static_cast<uint8_t>(fl_hdr.bitwidth_mode);
    info->component_bitwidth_low  = fl_hdr.component_bitwidth_low;
    info->component_bitwidth_high = fl_hdr.component_bitwidth_high;
    info->original_count = fl_hdr.original_count;
    info->padded_count   = fl_hdr.padded_count;
    info->body_size      = fl_hdr.body_size;
    info->min_value      = fl_hdr.min_value_bits();
    info->min_value_low_bits  = fl_hdr.min_value_low_bits;
    info->min_value_high_bits = fl_hdr.min_value_high_bits;

    auto const* payload = fastlanes::PageHeader::payload_ptr(s->data_start);
    uint32_t const body_words = fl_hdr.body_size / sizeof(uint32_t);
    uint32_t const preview_count =
      body_words < fastlanes_debug_info::MAX_PAYLOAD_PREVIEW_WORDS
        ? body_words
        : fastlanes_debug_info::MAX_PAYLOAD_PREVIEW_WORDS;
    info->payload_preview_count = preview_count;

    auto const* payload_u32 = reinterpret_cast<uint32_t const*>(payload);
    for (uint32_t i = 0; i < preview_count; ++i) {
      info->payload_preview[i] = payload_u32[i];
    }
  }
}

// ---------------------------------------------------------------------------
// INT32 Kernel: one thread-block (one warp) per page
// ---------------------------------------------------------------------------
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_block_size)
  decode_fastlanes_int32_kernel(PageInfo* pages,
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
                                                   lane,
                                                   error_code,
                                                   &fastlanes_header,
                                                   &total_value_count)) {
    return;
  }

  auto const* payload_bytes = fastlanes::PageHeader::payload_ptr(s->data_start);
  auto const packed_words_per_vector = static_cast<uint32_t>(fastlanes_header.bitwidth) * 32;
  auto const min_value_bits          = fastlanes_header.min_value_low_bits;

  auto const leaf_level_idx = s->col.max_nesting_depth - 1;
  auto* const output_base_8  = reinterpret_cast<uint8_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_16 = reinterpret_cast<uint16_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_32 = reinterpret_cast<uint32_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_64 = reinterpret_cast<int64_t*>(s->nesting_info[leaf_level_idx].data_out);

  uint32_t value_base_idx = 0;
  while (value_base_idx < total_value_count) {
    auto const vector_index = value_base_idx / fastlanes_vector_size;
    auto const remaining_value_count = total_value_count - value_base_idx;
    auto const values_in_vector =
      remaining_value_count < fastlanes_vector_size ? remaining_value_count : fastlanes_vector_size;

    auto const vector_input_bytes =
      payload_bytes + static_cast<size_t>(vector_index) *
                        static_cast<size_t>(packed_words_per_vector) * sizeof(uint32_t);

    // Some mixed-column pages can place FastLanes payload at non-4-byte aligned addresses.
    // Stage packed words into aligned shared memory before calling generated unpack code,
    // which expects uint32_t-aligned inputs.
    for (uint32_t w = lane; w < packed_words_per_vector; w += decode_fastlanes_block_size) {
      auto const* b        = vector_input_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
      packed_vec_aligned[w] = (static_cast<uint32_t>(b[0])) | (static_cast<uint32_t>(b[1]) << 8) |
                              (static_cast<uint32_t>(b[2]) << 16) |
                              (static_cast<uint32_t>(b[3]) << 24);
    }
    block.sync();

    unpack_device(packed_vec_aligned, decoded_vec, fastlanes_header.bitwidth);
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

// ---------------------------------------------------------------------------
// INT64 Kernel: one thread-block (one warp) per page
// ---------------------------------------------------------------------------
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_block_size)
  decode_fastlanes_int64_kernel(PageInfo* pages,
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
                                                   lane,
                                                   error_code,
                                                   &fastlanes_header,
                                                   &total_value_count)) {
    return;
  }

  auto const* payload_bytes = fastlanes::PageHeader::payload_ptr(s->data_start);
  // Current INT64 decode expects split32 component streams from header metadata.
  // TODO(native64): if NATIVE64 layout is added, dispatch to a dedicated native64
  // kernel path that decodes one 64-bit stream instead of low/high split streams.
  auto const packed_words_per_vector_low  = static_cast<uint32_t>(fastlanes_header.component_bitwidth_low) * 32;
  auto const packed_words_per_vector_high = static_cast<uint32_t>(fastlanes_header.component_bitwidth_high) * 32;
  auto const min_value_low_64             = fastlanes_header.min_value_low_bits;
  auto const min_value_high_64            = fastlanes_header.min_value_high_bits;

  auto const leaf_level_idx = s->col.max_nesting_depth - 1;
  auto* const output_base_64 = reinterpret_cast<int64_t*>(s->nesting_info[leaf_level_idx].data_out);

  uint32_t value_base_idx = 0;
  while (value_base_idx < total_value_count) {
    auto const vector_index = value_base_idx / fastlanes_vector_size;
    auto const remaining_value_count = total_value_count - value_base_idx;
    auto const values_in_vector =
      remaining_value_count < fastlanes_vector_size ? remaining_value_count : fastlanes_vector_size;

    auto const vector_input_bytes =
      payload_bytes + static_cast<size_t>(vector_index) *
                        static_cast<size_t>(packed_words_per_vector_low + packed_words_per_vector_high) *
                        sizeof(uint32_t);

    auto const* packed_low_bytes = vector_input_bytes;
    auto const* packed_high_bytes =
      vector_input_bytes + static_cast<size_t>(packed_words_per_vector_low) * sizeof(uint32_t);

    for (uint32_t w = lane; w < packed_words_per_vector_low; w += decode_fastlanes_block_size) {
      auto const* b            = packed_low_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
      packed_vec_low_aligned[w] = (static_cast<uint32_t>(b[0])) | (static_cast<uint32_t>(b[1]) << 8) |
                                  (static_cast<uint32_t>(b[2]) << 16) |
                                  (static_cast<uint32_t>(b[3]) << 24);
    }
    for (uint32_t w = lane; w < packed_words_per_vector_high; w += decode_fastlanes_block_size) {
      auto const* b             = packed_high_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
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
      auto const dst_pos    = static_cast<int32_t>(value_base_idx + i) - s->first_row;
      auto const low_bits   = min_value_low_64 + decoded_vec_low[i];
      auto const high_bits  = min_value_high_64 + decoded_vec_high[i];
      auto const value_bits =
        (static_cast<uint64_t>(high_bits) << 32) | static_cast<uint64_t>(low_bits);
      auto const signed_val = fastlanes::u64_bits_to_int64(value_bits);

      if (dst_pos >= 0 && dst_pos < s->num_rows) { output_base_64[dst_pos] = signed_val; }
    }

    block.sync();
    value_base_idx += values_in_vector;
  }
}

}  // anonymous namespace

// =============================================================================
// Host-side launch wrapper
// =============================================================================

void debug_decode_fastlanes_binary(cudf::detail::hostdevice_span<PageInfo> pages,
                            cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                            size_t num_rows,
                            size_t min_row,
                            int level_type_size,
                            cudf::device_span<bool const> page_mask,
                            kernel_error::pointer error_code,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  if (!fastlanes::debug::is_debug_kernel_enabled()) { return; }
  if (!fastlanes::debug::is_header_enabled() && !fastlanes::debug::is_workload_enabled()) {
    return;
  }

  rmm::device_uvector<fastlanes_debug_info> d_debug(pages.size(), stream);

  dim3 dim_block_debug(decode_fastlanes_debug_block_size, 1);
  dim3 dim_grid_debug(pages.size(), 1);

  if (level_type_size == 1) {
    decode_fastlanes_debug_kernel<uint8_t><<<dim_grid_debug, dim_block_debug, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, d_debug.data(), error_code);
  } else {
    decode_fastlanes_debug_kernel<uint16_t><<<dim_grid_debug, dim_block_debug, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, d_debug.data(), error_code);
  }

  std::vector<fastlanes_debug_info> h_debug(pages.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_debug.data(),
                                d_debug.data(),
                                pages.size() * sizeof(fastlanes_debug_info),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  for (size_t p = 0; p < pages.size(); ++p) {
    fastlanes::debug::print_page_debug(std::cout, h_debug[p], static_cast<int>(p));
  }
}

void decode_fastlanes_binary(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             cudf::device_span<bool const> page_mask,
                             kernel_error::pointer error_code,
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_fastlanes_block_size, 1);
  dim3 dim_grid(pages.size(), 1);

  if (level_type_size == 1) {
    decode_fastlanes_int32_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    // TODO(native64): replace this single INT64 split32 launch with layout-based dispatch
    // (split32 vs native64) once INT64 header validation supports NATIVE64 pages.
    decode_fastlanes_int64_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  } else {
    decode_fastlanes_int32_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    // TODO(native64): replace this single INT64 split32 launch with layout-based dispatch
    // (split32 vs native64) once INT64 header validation supports NATIVE64 pages.
    decode_fastlanes_int64_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  }
}

}  // namespace cudf::io::parquet::detail
