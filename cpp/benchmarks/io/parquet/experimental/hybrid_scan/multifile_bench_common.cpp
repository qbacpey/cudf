/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "multifile_bench_common.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <iterator>

multifile_bench_inputs::multifile_bench_inputs(cudf::io::source_info const& source_info)
  : datasources{cudf::io::make_datasources(source_info)}
{
  datasource_refs.reserve(datasources.size());
  footer_buffers.reserve(datasources.size());
  footer_byte_spans.reserve(datasources.size());

  for (auto const& datasource : datasources) {
    datasource_refs.emplace_back(*datasource);
    footer_buffers.emplace_back(cudf::io::parquet::fetch_footer_to_host(datasource_refs.back()));
    footer_byte_spans.emplace_back(*footer_buffers.back());
  }
}

std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources)
{
  auto const& [byte_ranges, source_map] = byte_ranges_and_source_map;
  CUDF_EXPECTS(byte_ranges.size() == source_map.size(), "Invalid source map size");

  auto byte_ranges_per_source =
    std::vector<std::vector<cudf::io::text::byte_range_info>>(num_sources);
  std::for_each(byte_ranges.begin(),
                byte_ranges.end(),
                [&, range_index = std::size_t{0}](auto const& range) mutable {
                  auto const source_index = source_map[range_index++];
                  CUDF_EXPECTS(source_index >= 0 and static_cast<std::size_t>(source_index) <
                                                       byte_ranges_per_source.size(),
                               "Invalid byte range source index");
                  byte_ranges_per_source[source_index].push_back(range);
                });
  return byte_ranges_per_source;
}

multisource_bench_data fetch_multisource_device_data(
  multifile_bench_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const byte_ranges_per_source =
    group_byte_ranges_by_source(byte_ranges_and_source_map, inputs.datasources.size());
  auto [buffers, per_source_spans, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    inputs.datasource_refs,
    cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{byte_ranges_per_source},
    cudf::io::parquet::io_submission_policy::SERIALIZE,
    stream,
    mr);

  auto flat_spans = std::vector<cudf::device_span<uint8_t const>>{};
  for (auto const& source_spans : per_source_spans) {
    flat_spans.insert(flat_spans.end(), source_spans.begin(), source_spans.end());
  }

  return {std::move(buffers),
          std::move(per_source_spans),
          std::move(flat_spans),
          std::move(tasks)};
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>>&& tables,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  auto table_views = std::vector<cudf::table_view>{};
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [](auto const& tbl) {
      return tbl->view();
    });
  return cudf::concatenate(table_views, stream, mr);
}
