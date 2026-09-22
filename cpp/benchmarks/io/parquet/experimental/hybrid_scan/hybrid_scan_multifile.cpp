/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "multifile_bench_common.hpp"

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <BS_thread_pool.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace io_parquet = cudf::io::parquet;
namespace exp        = cudf::io::parquet::experimental;

// Which read strategy to time.
enum class read_api : int32_t {
  SINGLE_SEQ,   // one hybrid_scan_reader per file, driven sequentially
  SINGLE_POOL,  // one hybrid_scan_reader per file, driven from a thread pool
  MULTIFILE     // a single hybrid_scan_multifile reader spanning all files
};

// Which window of the read path to time. `ALL` is the headline end-to-end number; the rest break
// down where the time goes so the multi-file win can be attributed.
enum class read_phase : int32_t {
  FOOTER,       // fetch footer bytes to host + construct reader(s)
  PLAN,         // select row groups + compute column-chunk byte ranges
  IO,           // fetch column-chunk bytes to device
  MATERIALIZE,  // decode column chunks into tables
  CONCAT,       // concatenate per-file tables (single-file arms only)
  ALL           // the whole path
};

}  // namespace

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  read_api,
  [](read_api value) {
    switch (value) {
      case read_api::SINGLE_SEQ: return "SINGLE_SEQ";
      case read_api::SINGLE_POOL: return "SINGLE_POOL";
      case read_api::MULTIFILE: return "MULTIFILE";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  read_phase,
  [](read_phase value) {
    switch (value) {
      case read_phase::FOOTER: return "FOOTER";
      case read_phase::PLAN: return "PLAN";
      case read_phase::IO: return "IO";
      case read_phase::MATERIALIZE: return "MATERIALIZE";
      case read_phase::CONCAT: return "CONCAT";
      case read_phase::ALL: return "ALL";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

namespace {

// Columns used for the NARROW configuration.
std::vector<std::string> const narrow_columns = {
  "l_orderkey", "l_quantity", "l_extendedprice", "l_shipdate"};

// Locate the parquet files for a given file-count sweep point under CUDF_BENCH_TPCH_ROOT.
// Expected layout: $ROOT/sf<SF>_p<num_files>/lineitem/*.parquet
// The scale factor defaults to 100 and can be overridden with CUDF_BENCH_TPCH_SF (e.g. "200").
std::vector<std::string> find_tpch_files(int64_t num_files)
{
  auto const* root = std::getenv("CUDF_BENCH_TPCH_ROOT");
  CUDF_EXPECTS(root != nullptr, "CUDF_BENCH_TPCH_ROOT is not set");

  auto const* sf = std::getenv("CUDF_BENCH_TPCH_SF");
  auto const dir =
    std::filesystem::path(root) /
    ("sf" + std::string(sf != nullptr ? sf : "100") + "_p" + std::to_string(num_files)) /
    "lineitem";
  CUDF_EXPECTS(std::filesystem::exists(dir), "Dataset directory does not exist: " + dir.string());

  std::vector<std::string> files;
  for (auto const& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".parquet") { files.push_back(entry.path().string()); }
  }
  std::sort(files.begin(), files.end());
  CUDF_EXPECTS(static_cast<int64_t>(files.size()) == num_files,
               "Expected " + std::to_string(num_files) + " files in " + dir.string() +
                 " but found " + std::to_string(files.size()));
  return files;
}

cudf::io::parquet_reader_options make_read_opts(cudf::io::source_info const& source_info,
                                                bool narrow)
{
  auto builder = cudf::io::parquet_reader_options::builder(source_info);
  if (narrow) { builder.column_names(narrow_columns); }
  return builder.build();
}

// -----------------------------------------------------------------------------------------------
// MULTIFILE arm: one reader spanning all sources.
// -----------------------------------------------------------------------------------------------
template <read_phase Phase, typename Timer>
std::unique_ptr<cudf::table> run_multifile(Timer& timer,
                                           cudf::io::source_info const& source_info,
                                           bool narrow,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  constexpr auto all = Phase == read_phase::ALL;

  if constexpr (Phase == read_phase::FOOTER or all) { timer.start(); }
  auto inputs = multifile_bench_inputs(source_info);
  auto opts   = make_read_opts(source_info, narrow);
  auto reader = exp::hybrid_scan_multifile{inputs.footer_byte_spans, opts};
  if constexpr (Phase == read_phase::FOOTER) { timer.stop(); }

  if constexpr (Phase == read_phase::PLAN or all) { timer.start(); }
  auto const row_groups = reader.all_row_groups(opts);
  auto const ranges     = reader.all_column_chunks_byte_ranges(row_groups, opts);
  if constexpr (Phase == read_phase::PLAN) { timer.stop(); }

  if constexpr (Phase == read_phase::IO or all) { timer.start(); }
  auto data = fetch_multisource_device_data(inputs, ranges, stream, mr);
  // The device reads are not stream-ordered; await the fetch future explicitly so the I/O is
  // fully inside the timed window.
  data.io_future.get();
  if constexpr (Phase == read_phase::IO) { timer.stop(); }

  if constexpr (Phase == read_phase::MATERIALIZE or all) { timer.start(); }
  auto result = reader.materialize_all_columns(row_groups, data.flat_spans, opts, stream, mr);
  // Sync inside the timed window so async decode is fully accounted for before the timer stops.
  stream.sync();
  cudaDeviceSynchronize();  // belt-and-suspenders: catch any non-stream-ordered residual work
  if constexpr (Phase == read_phase::MATERIALIZE or all) { timer.stop(); }

  return std::move(result.tbl);
}

// -----------------------------------------------------------------------------------------------
// Single-file body shared by the sequential and pooled arms. Reads one file end to end.
// -----------------------------------------------------------------------------------------------
std::unique_ptr<cudf::table> read_one_file(cudf::io::datasource& datasource,
                                           cudf::io::parquet_reader_options const& opts,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const footer = io_parquet::fetch_footer_to_host(datasource);
  auto reader       = exp::hybrid_scan_reader{*footer, opts};
  auto const rgs    = reader.all_row_groups(opts);
  auto const ranges = reader.all_column_chunks_byte_ranges(rgs, opts);
  auto [buf, spans, tasks] = io_parquet::fetch_byte_ranges_to_device_async(
    datasource, ranges, io_parquet::io_submission_policy::SERIALIZE, stream, mr);
  tasks.get();  // await the non-stream-ordered device reads before decoding
  auto tbl = reader.materialize_all_columns(rgs, spans, opts, stream, mr).tbl;
  stream.sync();
  return tbl;
}

// -----------------------------------------------------------------------------------------------
// SINGLE_SEQ arm: drive the per-file readers one after another on a single stream.
// -----------------------------------------------------------------------------------------------
template <read_phase Phase, typename Timer>
std::unique_ptr<cudf::table> run_single_seq(Timer& timer,
                                            cudf::io::source_info const& source_info,
                                            bool narrow,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto datasources = cudf::io::make_datasources(source_info);
  auto opts        = make_read_opts(source_info, narrow);

  std::vector<std::unique_ptr<cudf::table>> parts;
  parts.reserve(datasources.size());

  // For per-phase timing we re-walk the files and time only the matching window. The single-file
  // reader does not expose sub-steps separately, so each phase re-runs the loop timing its slice.
  if constexpr (Phase == read_phase::FOOTER) {
    for (auto& ds : datasources) {
      timer.start();
      auto const footer = io_parquet::fetch_footer_to_host(*ds);
      auto reader       = exp::hybrid_scan_reader{*footer, opts};
      timer.stop();
    }
    return nullptr;
  } else if constexpr (Phase == read_phase::PLAN) {
    for (auto& ds : datasources) {
      auto const footer = io_parquet::fetch_footer_to_host(*ds);
      auto reader       = exp::hybrid_scan_reader{*footer, opts};
      timer.start();
      auto const rgs    = reader.all_row_groups(opts);
      auto const ranges = reader.all_column_chunks_byte_ranges(rgs, opts);
      timer.stop();
    }
    return nullptr;
  } else if constexpr (Phase == read_phase::IO) {
    for (auto& ds : datasources) {
      auto const footer = io_parquet::fetch_footer_to_host(*ds);
      auto reader       = exp::hybrid_scan_reader{*footer, opts};
      auto const rgs    = reader.all_row_groups(opts);
      auto const ranges = reader.all_column_chunks_byte_ranges(rgs, opts);
      timer.start();
      auto [buf, spans, tasks] = io_parquet::fetch_byte_ranges_to_device_async(
        *ds, ranges, io_parquet::io_submission_policy::SERIALIZE, stream, mr);
      tasks.get();
      stream.sync();
      timer.stop();
    }
    return nullptr;
  } else if constexpr (Phase == read_phase::MATERIALIZE) {
    for (auto& ds : datasources) {
      auto const footer = io_parquet::fetch_footer_to_host(*ds);
      auto reader       = exp::hybrid_scan_reader{*footer, opts};
      auto const rgs    = reader.all_row_groups(opts);
      auto const ranges = reader.all_column_chunks_byte_ranges(rgs, opts);
      auto [buf, spans, tasks] = io_parquet::fetch_byte_ranges_to_device_async(
        *ds, ranges, io_parquet::io_submission_policy::SERIALIZE, stream, mr);
      tasks.get();
      timer.start();
      auto tbl = reader.materialize_all_columns(rgs, spans, opts, stream, mr).tbl;
      stream.sync();
      timer.stop();
      parts.push_back(std::move(tbl));
    }
    return nullptr;
  } else if constexpr (Phase == read_phase::CONCAT) {
    for (auto& ds : datasources) {
      parts.push_back(read_one_file(*ds, opts, stream, mr));
    }
    stream.sync();
    timer.start();
    auto out = concatenate_tables(std::move(parts), stream, mr);
    stream.sync();
    timer.stop();
    return out;
  } else {  // ALL
    timer.start();
    for (auto& ds : datasources) {
      parts.push_back(read_one_file(*ds, opts, stream, mr));
    }
    auto out = concatenate_tables(std::move(parts), stream, mr);
    stream.sync();
    timer.stop();
    return out;
  }
}

// -----------------------------------------------------------------------------------------------
// SINGLE_POOL arm: drive the per-file readers from a thread pool over forked streams.
// -----------------------------------------------------------------------------------------------
template <read_phase Phase, typename Timer>
std::unique_ptr<cudf::table> run_single_pool(Timer& timer,
                                             cudf::io::source_info const& source_info,
                                             bool narrow,
                                             int64_t num_threads,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto datasources = cudf::io::make_datasources(source_info);
  auto opts        = make_read_opts(source_info, narrow);
  auto const n     = static_cast<int64_t>(datasources.size());

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  BS::thread_pool pool(num_threads);

  std::vector<std::unique_ptr<cudf::table>> parts(n);

  // The pooled arm is only meaningfully timed end to end; sub-phase timing of a threaded loop is
  // not comparable to the sequential breakdown, so non-ALL phases fall back to timing ALL.
  timer.start();
  pool.detach_sequence(int64_t{0}, n, [&](int64_t i) {
    auto const s = streams[i % num_threads];
    parts[i]     = read_one_file(*datasources[i], opts, s, mr);
  });
  pool.wait();
  cudf::detail::join_streams(streams, cudf::get_default_stream());
  auto out = concatenate_tables(std::move(parts), stream, mr);
  stream.sync();
  timer.stop();
  return out;
}

}  // namespace

template <read_api Api, read_phase Phase>
void BM_hybrid_scan_multifile(nvbench::state& state,
                              nvbench::type_list<nvbench::enum_type<Api>,
                                                 nvbench::enum_type<Phase>>)
{
  auto const num_files   = state.get_int64("num_files");
  auto const num_threads = state.get_int64("num_threads");
  auto const narrow      = state.get_string("columns") == "NARROW";

  // num_threads is only meaningful for the pooled arm.
  if constexpr (Api != read_api::SINGLE_POOL) {
    if (num_threads != 1) {
      state.skip("num_threads only applies to SINGLE_POOL");
      return;
    }
  }

  auto const files       = find_tpch_files(num_files);
  auto const source_info = cudf::io::source_info(files);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  auto const stream           = cudf::get_default_stream();
  auto const mr               = cudf::get_current_device_resource_ref();

  std::size_t total_rows = 0;
  // Wall-clock per sample, measured around the whole operation. The kvikio device reads are not
  // stream-ordered, so nvbench's CUDA-event (GPU) timer does not capture them; the wall clock does.
  double wall_seconds = 0.0;
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(files);

               auto const wall_start = std::chrono::steady_clock::now();
               std::unique_ptr<cudf::table> result;
               if constexpr (Api == read_api::MULTIFILE) {
                 result = run_multifile<Phase>(timer, source_info, narrow, stream, mr);
               } else if constexpr (Api == read_api::SINGLE_SEQ) {
                 result = run_single_seq<Phase>(timer, source_info, narrow, stream, mr);
               } else {
                 result = run_single_pool<Phase>(
                   timer, source_info, narrow, num_threads, stream, mr);
               }
               // Block until every async read and kernel has fully completed before reading the
               // wall clock, so off-stream kvikio I/O is included.
               cudaDeviceSynchronize();
               auto const wall_stop = std::chrono::steady_clock::now();
               wall_seconds = std::chrono::duration<double>(wall_stop - wall_start).count();
               if (result) { total_rows = result->num_rows(); }
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(total_rows) / time, "rows_per_sec");
  // Throughput on the honest wall-clock metric.
  state.add_element_count(static_cast<double>(total_rows) / wall_seconds, "rows_per_sec_wall");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using read_apis =
  nvbench::enum_type_list<read_api::SINGLE_SEQ, read_api::SINGLE_POOL, read_api::MULTIFILE>;

using read_phases = nvbench::enum_type_list<read_phase::FOOTER,
                                            read_phase::PLAN,
                                            read_phase::IO,
                                            read_phase::MATERIALIZE,
                                            read_phase::CONCAT,
                                            read_phase::ALL>;

NVBENCH_BENCH_TYPES(BM_hybrid_scan_multifile, NVBENCH_TYPE_AXES(read_apis, read_phases))
  .set_name("hybrid_scan_multifile")
  .set_type_axes_names({"api", "phase"})
  .set_min_samples(4)
  .add_int64_axis("num_files", {1, 8, 64, 256, 1024})
  .add_int64_axis("num_threads", {1, 8})
  .add_string_axis("columns", {"NARROW", "ALL"});
