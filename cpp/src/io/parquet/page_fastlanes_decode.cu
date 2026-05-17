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
// - INT32 payload pages; output can be 32-bit logical INT32 classes or TIME_MILLIS as 64-bit.
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
    info->cast_mode      = static_cast<uint8_t>(fl_hdr.cast_mode);
    info->original_count = fl_hdr.original_count;
    info->padded_count   = fl_hdr.padded_count;
    info->body_size      = fl_hdr.body_size;
    info->min_value      = fl_hdr.min_value;

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
// Kernel: one thread-block (one warp) per page
// ---------------------------------------------------------------------------
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_fastlanes_block_size)
  decode_fastlanes_kernel(PageInfo* pages,
                          device_span<ColumnChunkDesc const> chunks,
                          size_t min_row,
                          size_t num_rows,
                          cudf::device_span<bool const> page_mask,
                          kernel_error::pointer error_code)
{
  __shared__ uint32_t decoded_vec[fastlanes_vector_size];
  __shared__ uint32_t packed_vec_aligned[fastlanes_vector_size];
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s         = &state_g;
  int const page_idx            = cg::this_grid().block_rank();
  auto const block              = cg::this_thread_block();
  int const lane                = static_cast<int>(block.thread_rank());
  [[maybe_unused]] null_count_back_copier _{s, lane};

  // Setup page info - use FASTLANES_BINARY mask so only FL pages are processed
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::FASTLANES_BINARY},
                             page_processing_stage::DECODE)) {
    return;
  }

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
    return;
  }

  // Current implementation target: flat columns only.
  if (has_repetition_levels || s->col.max_nesting_depth != 1) {
    if (lane == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING),
                error_code);
    }
    return;
  }

  // Nullable schema is allowed only when this page actually has zero nulls.
  // For optional Parquet fields with all rows populated, page.num_nulls is 0,
  // so this path is valid even though max definition level > 0.
  if (s->page.num_nulls > 0) {
    if (lane == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING),
                error_code);
    }
    return;
  }

  // Current FastLanes integration supports 8/16/32-bit logical outputs and TIME_MILLIS
  // outputs materialized as 64-bit durations.
  if (s->dtype_len != 1 && s->dtype_len != 2 && s->dtype_len != 4 && s->dtype_len != 8) {
    if (lane == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE),
                error_code);
    }
    return;
  }

  auto const fastlanes_header = fastlanes::PageHeader::deserialize(s->data_start);
  auto const bit_width        = fastlanes_header.bitwidth;
  auto const cast_mode        = fastlanes_header.cast_mode;
  if (!fastlanes::is_valid_cast_mode(cast_mode)) {
    if (lane == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING),
                error_code);
    }
    return;
  }

  auto const total_value_count =
    fastlanes_header.original_count < s->num_input_values ? fastlanes_header.original_count
                                                           : s->num_input_values;
  auto const min_signed_value = fastlanes::u32_bits_to_int32(fastlanes_header.min_value);

  auto const* payload_bytes = fastlanes::PageHeader::payload_ptr(s->data_start);
  auto const packed_words_per_vector = static_cast<uint32_t>(bit_width) * 32;

  auto const leaf_level_idx = s->col.max_nesting_depth - 1;
  auto* const output_base_8  = reinterpret_cast<uint8_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_16 = reinterpret_cast<uint16_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_32 = reinterpret_cast<uint32_t*>(s->nesting_info[leaf_level_idx].data_out);
  auto* const output_base_64 = reinterpret_cast<int64_t*>(s->nesting_info[leaf_level_idx].data_out);

  // Decode vector-by-vector; each vector has 1024 values and is decoded by one warp.
  uint32_t value_base_idx = 0;
  while (value_base_idx < total_value_count) {
    auto const vector_index = value_base_idx / fastlanes_vector_size;
    auto const vector_input_bytes =
      payload_bytes +
      static_cast<size_t>(vector_index) * static_cast<size_t>(packed_words_per_vector) *
        sizeof(uint32_t);

    // Some mixed-column pages can place FastLanes payload at non-4-byte aligned addresses.
    // Stage packed words into aligned shared memory before calling generated unpack code,
    // which expects uint32_t-aligned inputs.
    for (uint32_t w = lane; w < packed_words_per_vector; w += decode_fastlanes_block_size) {
      auto const* b = vector_input_bytes + static_cast<size_t>(w) * sizeof(uint32_t);
      packed_vec_aligned[w] = (static_cast<uint32_t>(b[0])) |
                              (static_cast<uint32_t>(b[1]) << 8) |
                              (static_cast<uint32_t>(b[2]) << 16) |
                              (static_cast<uint32_t>(b[3]) << 24);
    }
    block.sync();

    unpack_device(packed_vec_aligned, decoded_vec, bit_width);
    block.sync();

    auto const remaining_value_count = total_value_count - value_base_idx;
    auto const values_in_vector =
      remaining_value_count < fastlanes_vector_size ? remaining_value_count : fastlanes_vector_size;
    for (uint32_t i = lane; i < values_in_vector; i += decode_fastlanes_block_size) {
      auto const dst_pos = static_cast<int32_t>(value_base_idx + i) - s->first_row;
      auto const delta   = decoded_vec[i];
      auto const base_value = static_cast<int64_t>(min_signed_value);
      auto const signed_val = static_cast<int32_t>(base_value + static_cast<int64_t>(delta));
      auto const value_bits = fastlanes::int32_to_u32_bits(signed_val);
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
    decode_fastlanes_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  } else {
    decode_fastlanes_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
  }
}

}  // namespace cudf::io::parquet::detail
