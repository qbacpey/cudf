/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmarks/io/cuio_common.hpp>

#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

constexpr cudf::size_type num_cols = 64;

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state);

/**
 * @brief Mixed dtypes (STRING, INTEGRAL, FLOAT, DECIMAL, LIST) cycled to fill a wide schema.
 *
 * @return The dtype list
 */
[[nodiscard]] std::vector<cudf::type_id> const& mixed_dtypes();

/**
 * @brief Writes a parquet file of `mixed_dtypes()` columns with a fixed number of rows per row
 * group, using the chunked writer to reach the requested row group count.
 *
 * Shared by the metadata and setup-cost benchmarks so that the naive and hybrid scan variants
 * measure the same file layout.
 *
 * @param num_cols Number of columns in the schema
 * @param num_row_groups Number of row groups to write; must be a non-zero multiple of 10
 * @param source_type IO type of the sink to write to
 * @param write_page_index Whether to write the page index (STATISTICS_COLUMN)
 *
 * @return The source/sink pair holding the written file
 */
[[nodiscard]] cuio_source_sink_pair write_mixed_dtype_parquet_file(cudf::size_type num_cols,
                                                                   cudf::size_type num_row_groups,
                                                                   io_type source_type,
                                                                   bool write_page_index);

/**
 * @brief Writes a single-column parquet file with an explicitly controlled row group and page
 * layout.
 *
 * Shared by `parquet_read_file_shape` and `hybrid_scan_file_shape` so the naive and hybrid scan
 * variants see the same file layout on the same axes. The page byte limit and fragment size are
 * pinned so that the requested row group and page counts are what actually land in the file.
 *
 * @param dtype Type of the single column
 * @param num_rows Number of rows in the table
 * @param num_row_groups Number of row groups to request
 * @param pages_per_row_group Number of pages to request per row group
 * @param source_type IO type of the sink to write to
 * @param write_page_index Whether to write the page index (STATISTICS_COLUMN)
 *
 * @return The source/sink pair holding the written file
 */
[[nodiscard]] cuio_source_sink_pair write_file_shape_parquet_file(
  cudf::type_id dtype,
  cudf::size_type num_rows,
  cudf::size_type num_row_groups,
  cudf::size_type pages_per_row_group,
  io_type source_type,
  bool write_page_index);
