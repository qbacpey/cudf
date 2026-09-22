/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <vector>

/**
 * @brief Datasources, datasource refs, and footer buffers/spans for a set of Parquet sources.
 *
 * Mirrors the test helper of the same name in `cpp/tests/io/experimental/hybrid_scan_common.hpp`,
 * duplicated here so the benchmark does not depend on the test tree.
 */
struct multifile_bench_inputs {
  explicit multifile_bench_inputs(cudf::io::source_info const& source_info);

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::reference_wrapper<cudf::io::datasource>> datasource_refs;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  std::vector<cudf::host_span<uint8_t const>> footer_byte_spans;
};

/**
 * @brief Device buffers and spans fetched from multiple input sources
 */
struct multisource_bench_data {
  std::vector<rmm::device_buffer> buffers;
  std::vector<std::vector<cudf::device_span<uint8_t const>>> per_source_spans;
  std::vector<cudf::device_span<uint8_t const>> flat_spans;
  // Awaits completion of the async device reads backing `buffers`. Must be `.get()` before the
  // data is consumed or the timed region ends; the reads are not stream-ordered.
  std::future<void> io_future;
};

/**
 * @brief Groups a flat byte range list by source using the given source map
 */
[[nodiscard]] std::vector<std::vector<cudf::io::text::byte_range_info>>
group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources);

/**
 * @brief Fetches byte ranges from multiple sources, returning per-source and flattened spans
 */
[[nodiscard]] multisource_bench_data fetch_multisource_device_data(
  multifile_bench_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Concatenate a vector of tables, returning the single table unchanged if only one
 */
[[nodiscard]] std::unique_ptr<cudf::table> concatenate_tables(
  std::vector<std::unique_ptr<cudf::table>>&& tables,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);
