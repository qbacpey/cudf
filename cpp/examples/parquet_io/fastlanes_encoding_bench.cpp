/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "timer.hpp"

#include <cudf/copying.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream_ref>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/**
 * @file fastlanes_encoding_bench.cpp
 * @brief Encode/decode benchmark comparing FastLanes with the standard Parquet encodings.
 *
 * `sweep` mode loads each requested column of a Parquet file onto the GPU once, then for every
 * (encoding, compression) pair writes that column into a host buffer with the chunked writer and
 * reads it back in row-group batches. For each pair it reports the output size and the median
 * write/read times, checks that the decoded data equals the source, and records the data page
 * encodings from the footer so that silent fallbacks are visible.
 *
 * `read` mode times warm-cache batched reads of whole Parquet files.
 *
 * Results are appended to a CSV file (a header is written when the file is new). A CUDA stream
 * wait cannot be interrupted, so a watchdog thread bounds each write/read: when one exceeds
 * --hang-timeout-s it appends the current case to the CSV with an error and exits with code 3,
 * letting a driver resume the remaining cases (see --combos).
 */

namespace {

namespace pq = cudf::io::parquet;

std::size_t constexpr mib = std::size_t{1} << 20;

struct bench_options {
  std::string mode;
  std::vector<std::string> inputs;
  std::string table;
  std::string label;
  std::vector<std::string> columns;
  std::vector<std::string> encodings{
    "PLAIN", "DICTIONARY", "DELTA_BINARY_PACKED", "BYTE_STREAM_SPLIT", "FASTLANES"};
  std::vector<std::string> compressions{"NONE", "SNAPPY", "ZSTD"};
  std::vector<std::pair<std::string, std::string>> combos;
  std::string output;
  int hang_timeout_s                     = 600;
  int warmup                             = 1;
  int repeats                            = 3;
  cudf::size_type row_group_rows         = 122'880;
  cudf::size_type page_rows              = 20'480;
  // Pages and row groups are built from whole page fragments; -1 uses `page_rows` so pages have
  // exactly `page_rows` rows, 0 keeps cuDF's default fragment sizing (5000 rows for these columns).
  cudf::size_type fragment_rows = -1;
  cudf::size_type write_slice_row_groups = 100;
  cudf::size_type read_batch_row_groups  = 256;
  std::size_t pass_read_limit            = 4096 * mib;
  bool v2_headers                        = true;
  bool validate                          = true;
};

std::vector<std::string> split(std::string const& s, char delim = ',')
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (not item.empty()) { out.push_back(item); }
  }
  return out;
}

std::string csv_escape(std::string s)
{
  std::replace(s.begin(), s.end(), ',', ';');
  std::replace(s.begin(), s.end(), '\n', ' ');
  return s;
}

/**
 * @brief Watchdog that turns a hung write/read into a recorded CSV row and a fast exit.
 */
class hang_watchdog {
 public:
  hang_watchdog()
    : _thread([this] {
        while (not _stop.load()) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          auto const deadline = _deadline_ms.load();
          if (deadline != 0 and now_ms() > deadline) { fire(); }
        }
      })
  {
  }

  ~hang_watchdog()
  {
    _stop.store(true);
    _thread.join();
  }

  /// Start timing one operation; `hang_row` is appended to `csv_path` if it does not finish.
  void arm(std::string const& csv_path, std::string const& hang_row, int timeout_s)
  {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _csv_path = csv_path;
      _hang_row = hang_row;
    }
    _deadline_ms.store(now_ms() + int64_t{timeout_s} * 1000);
  }

  void disarm() { _deadline_ms.store(0); }

 private:
  static int64_t now_ms()
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }

  void fire()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::cerr << "[watchdog] operation exceeded the hang timeout; recording and exiting\n";
    std::ofstream csv(_csv_path, std::ios::app);
    csv << _hang_row << '\n';
    csv.flush();
    std::_Exit(3);
  }

  std::atomic<bool> _stop{false};
  std::atomic<int64_t> _deadline_ms{0};
  std::mutex _mutex;
  std::string _csv_path;
  std::string _hang_row;
  std::thread _thread;
};

double median(std::vector<double> v)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  auto const n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double min_of(std::vector<double> const& v)
{
  return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

double max_of(std::vector<double> const& v)
{
  return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

std::string encoding_name(pq::Encoding enc)
{
  switch (enc) {
    case pq::Encoding::PLAIN: return "PLAIN";
    case pq::Encoding::PLAIN_DICTIONARY: return "PLAIN_DICTIONARY";
    case pq::Encoding::RLE: return "RLE";
    case pq::Encoding::BIT_PACKED: return "BIT_PACKED";
    case pq::Encoding::DELTA_BINARY_PACKED: return "DELTA_BINARY_PACKED";
    case pq::Encoding::DELTA_LENGTH_BYTE_ARRAY: return "DELTA_LENGTH_BYTE_ARRAY";
    case pq::Encoding::DELTA_BYTE_ARRAY: return "DELTA_BYTE_ARRAY";
    case pq::Encoding::RLE_DICTIONARY: return "RLE_DICTIONARY";
    case pq::Encoding::BYTE_STREAM_SPLIT: return "BYTE_STREAM_SPLIT";
    case pq::Encoding::FASTLANE_BITPACK_RAW: return "FASTLANE_BITPACK_RAW";
    case pq::Encoding::FASTLANES_DELTA_BINARY: return "FASTLANES_DELTA_BINARY";
    case pq::Encoding::FASTLANE_BITPACK_SPLIT64: return "FASTLANE_BITPACK_SPLIT64";
    default: return "UNKNOWN(" + std::to_string(static_cast<int>(enc)) + ")";
  }
}

/**
 * @brief Maps the pseudo-encoding "FASTLANES" to the FastLanes mode for the column's physical
 * type: INT64 columns use FASTLANES_DELTA_BINARY, INT32-backed columns use FASTLANE_BITPACK_RAW.
 */
cudf::io::column_encoding resolve_encoding(std::string const& name, cudf::data_type type)
{
  if (name == "FASTLANES") {
    auto const id = type.id();
    return (id == cudf::type_id::INT64 or id == cudf::type_id::UINT64)
             ? cudf::io::column_encoding::FASTLANES_DELTA_BINARY
             : cudf::io::column_encoding::FASTLANE_BITPACK_RAW;
  }
  return get_encoding_type(name);
}

std::vector<pq::Encoding> expected_data_page_encodings(cudf::io::column_encoding enc)
{
  using ce = cudf::io::column_encoding;
  switch (enc) {
    case ce::PLAIN: return {pq::Encoding::PLAIN};
    case ce::DICTIONARY: return {pq::Encoding::PLAIN_DICTIONARY, pq::Encoding::RLE_DICTIONARY};
    case ce::DELTA_BINARY_PACKED: return {pq::Encoding::DELTA_BINARY_PACKED};
    case ce::BYTE_STREAM_SPLIT: return {pq::Encoding::BYTE_STREAM_SPLIT};
    case ce::FASTLANE_BITPACK_RAW: return {pq::Encoding::FASTLANE_BITPACK_RAW};
    case ce::FASTLANES_DELTA_BINARY: return {pq::Encoding::FASTLANES_DELTA_BINARY};
    default: return {};
  }
}

struct footer_summary {
  std::map<std::string, int64_t> data_pages_by_encoding;
  int64_t data_pages       = 0;
  int64_t unexpected_pages = 0;
  int64_t row_groups       = 0;

  [[nodiscard]] std::string encodings_string() const
  {
    std::string out;
    for (auto const& [name, count] : data_pages_by_encoding) {
      if (not out.empty()) { out += ';'; }
      out += name + ":" + std::to_string(count);
    }
    return out;
  }
};

footer_summary summarize_footer(std::vector<char> const& buffer,
                                cudf::io::column_encoding requested)
{
  CUDF_EXPECTS(buffer.size() > 12, "Output buffer is too small to contain a Parquet footer");
  uint32_t footer_len = 0;
  std::memcpy(&footer_len, buffer.data() + buffer.size() - 8, sizeof(footer_len));
  CUDF_EXPECTS(footer_len + 12 <= buffer.size(), "Invalid Parquet footer length");
  auto const* footer_begin =
    reinterpret_cast<uint8_t const*>(buffer.data() + buffer.size() - 8 - footer_len);

  pq::experimental::hybrid_scan_reader const reader(
    cudf::host_span<uint8_t const>(footer_begin, footer_len), cudf::io::parquet_reader_options{});
  auto const metadata = reader.parquet_metadata();
  auto const expected = expected_data_page_encodings(requested);
  auto const is_expected = [&](pq::Encoding e) {
    return std::find(expected.begin(), expected.end(), e) != expected.end();
  };

  footer_summary summary;
  summary.row_groups = static_cast<int64_t>(metadata.row_groups.size());
  for (auto const& rg : metadata.row_groups) {
    for (auto const& chunk : rg.columns) {
      auto const& md = chunk.meta_data;
      if (not md.encoding_stats.has_value()) { continue; }
      for (auto const& stat : *md.encoding_stats) {
        if (stat.page_type != pq::PageType::DATA_PAGE and
            stat.page_type != pq::PageType::DATA_PAGE_V2) {
          continue;
        }
        summary.data_pages += stat.count;
        summary.data_pages_by_encoding[encoding_name(stat.encoding)] += stat.count;
        if (not is_expected(stat.encoding)) { summary.unexpected_pages += stat.count; }
      }
    }
  }
  return summary;
}

std::unique_ptr<cudf::table> load_column(std::string const& path,
                                         std::string const& column,
                                         bench_options const& cfg,
                                         cuda::stream_ref stream)
{
  auto const opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info(path))
                      .column_names({column})
                      .build();
  auto reader = cudf::io::chunked_parquet_reader(0, cfg.pass_read_limit, opts, stream);
  std::vector<std::unique_ptr<cudf::table>> chunks;
  while (reader.has_next()) {
    chunks.push_back(std::move(reader.read_chunk().tbl));
  }
  return concatenate_tables(std::move(chunks), stream);
}

/**
 * @brief Writes a single-column table into `buffer`, returning the elapsed milliseconds.
 *
 * The table is handed to the chunked writer in slices whose row counts are multiples of the row
 * group size, so the row group layout is identical for every encoding.
 */
double write_column(cudf::table_view const& tbl,
                    std::string const& name,
                    cudf::io::column_encoding encoding,
                    cudf::io::compression_type compression,
                    bench_options const& cfg,
                    std::vector<char>& buffer,
                    cuda::stream_ref stream)
{
  buffer.clear();

  cudf::io::table_input_metadata metadata(tbl);
  metadata.column_metadata[0].set_name(name).set_encoding(encoding);

  auto writer_builder =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&buffer));
  writer_builder.metadata(metadata)
    .compression(compression)
    .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP)
    .row_group_size_rows(cfg.row_group_rows)
    .max_page_size_rows(cfg.page_rows)
    .write_v2_headers(cfg.v2_headers);
  if (auto const frag = cfg.fragment_rows < 0 ? cfg.page_rows : cfg.fragment_rows; frag > 0) {
    writer_builder.max_page_fragment_size(frag);
  }
  auto const writer_opts = writer_builder.build();

  auto const num_rows   = tbl.num_rows();
  auto const slice_rows = static_cast<cudf::size_type>(
    std::min<int64_t>(int64_t{cfg.row_group_rows} * cfg.write_slice_row_groups,
                      std::numeric_limits<cudf::size_type>::max()));
  std::vector<cudf::size_type> bounds;
  for (cudf::size_type begin = 0; begin < num_rows; begin += std::min(slice_rows, num_rows - begin)) {
    bounds.push_back(begin);
    bounds.push_back(begin + std::min(slice_rows, num_rows - begin));
  }
  auto const slices = cudf::slice(tbl, bounds, stream);

  stream.sync();
  timer t;
  {
    cudf::io::chunked_parquet_writer writer(writer_opts, stream);
    for (auto const& slice : slices) {
      writer.write(slice);
    }
    writer.close();
  }
  stream.sync();
  return t.elapsed_millis();
}

/**
 * @brief Reads `source` in batches of `cfg.read_batch_row_groups` row groups, calling `on_batch`
 * with each decoded table.
 *
 * Batched `read_parquet` bounds device memory like the chunked reader does, but the chunked
 * reader's pass limit makes cuDF size ZSTD scratch space with
 * nvcompBatchedZstdDecompressGetTempSizeSync, which never returns for some inputs on the GPU used
 * for these measurements.
 */
template <typename Fn>
void read_in_row_group_batches(cudf::io::source_info const& source,
                               bench_options const& cfg,
                               cuda::stream_ref stream,
                               Fn&& on_batch)
{
  auto const num_row_groups = cudf::io::read_parquet_metadata(source).num_rowgroups();
  for (cudf::size_type begin = 0; begin < num_row_groups; begin += cfg.read_batch_row_groups) {
    auto const end = std::min(begin + cfg.read_batch_row_groups, num_row_groups);
    std::vector<cudf::size_type> row_groups(end - begin);
    std::iota(row_groups.begin(), row_groups.end(), begin);
    auto const opts =
      cudf::io::parquet_reader_options::builder(source).row_groups({row_groups}).build();
    on_batch(cudf::io::read_parquet(opts, stream).tbl->view());
  }
}

std::pair<double, int64_t> timed_read(cudf::io::source_info const& source,
                                      bench_options const& cfg,
                                      cuda::stream_ref stream)
{
  stream.sync();
  timer t;
  int64_t rows = 0;
  read_in_row_group_batches(
    source, cfg, stream, [&](cudf::table_view const& batch) { rows += batch.num_rows(); });
  stream.sync();
  return {t.elapsed_millis(), rows};
}

bool matches_source(std::vector<char> const& buffer,
                    cudf::table_view const& source_tbl,
                    bench_options const& cfg,
                    cuda::stream_ref stream)
{
  auto const source =
    cudf::io::source_info(cudf::host_span<char const>(buffer.data(), buffer.size()));
  cudf::size_type offset = 0;
  bool equal             = true;
  read_in_row_group_batches(source, cfg, stream, [&](cudf::table_view const& batch) {
    auto const rows = batch.num_rows();
    if (not equal or offset + rows > source_tbl.num_rows()) {
      equal = false;
      return;
    }
    std::vector<cudf::size_type> const bounds{offset, offset + rows};
    auto const expected = cudf::slice(source_tbl, bounds, stream).front();
    equal  = cudf::tables_equal(expected, batch, cudf::null_equality::EQUAL, stream);
    offset += rows;
  });
  return equal and offset == source_tbl.num_rows();
}

bool file_is_empty(std::string const& path)
{
  return not std::filesystem::exists(path) or std::filesystem::file_size(path) == 0;
}

void run_sweep(bench_options const& cfg)
{
  CUDF_EXPECTS(cfg.inputs.size() == 1, "sweep mode takes exactly one --input file");
  CUDF_EXPECTS(not cfg.columns.empty(), "sweep mode requires --columns");
  CUDF_EXPECTS(not cfg.output.empty(), "sweep mode requires --output");

  hang_watchdog watchdog;
  auto const stream       = cudf::get_default_stream();
  bool const write_header = file_is_empty(cfg.output);
  std::ofstream csv(cfg.output, std::ios::app);
  if (write_header) {
    csv << "table,column,type,rows,raw_bytes,requested,resolved,compression,row_group_rows,"
           "page_rows,fragment_rows,v2_headers,repeats,bytes,bits_per_value,ratio_vs_raw,"
           "write_ms_median,"
           "write_ms_min,write_ms_max,read_ms_median,read_ms_min,read_ms_max,write_gbps,"
           "read_gbps,row_groups,data_pages,page_encodings,unexpected_pages,validated,error\n";
  }

  for (auto const& column : cfg.columns) {
    std::cerr << "[sweep] loading " << cfg.table << "." << column << "\n";
    auto const tbl     = load_column(cfg.inputs.front(), column, cfg, stream);
    auto const view    = tbl->view();
    auto const type    = view.column(0).type();
    auto const rows    = static_cast<int64_t>(view.num_rows());
    auto const raw     = rows * static_cast<int64_t>(cudf::size_of(type));
    auto const type_nm = cudf::type_to_name(type);

    std::vector<char> buffer;
    buffer.reserve(static_cast<std::size_t>(raw) + static_cast<std::size_t>(raw) / 2 + 64 * mib);

    auto combos = cfg.combos;
    if (combos.empty()) {
      for (auto const& enc : cfg.encodings) {
        for (auto const& comp : cfg.compressions) {
          combos.emplace_back(enc, comp);
        }
      }
    }

    for (auto const& [requested, comp_name] : combos) {
      auto const encoding    = resolve_encoding(requested, type);
      auto const compression = get_compression_type(comp_name);
      std::vector<double> write_ms;
      std::vector<double> read_ms;
      footer_summary footer;
      bool validated = false;
      std::string error;
      std::size_t bytes = 0;

      auto const format_row = [&]() {
        auto const gbps = [&](double ms) {
          return ms > 0 ? static_cast<double>(raw) / (ms * 1e-3) / 1e9 : 0.0;
        };
        std::ostringstream row;
        row << cfg.table << ',' << column << ',' << type_nm << ',' << rows << ',' << raw << ','
            << requested << ',' << get_encoding_string(encoding) << ',' << comp_name << ','
            << cfg.row_group_rows << ',' << cfg.page_rows << ','
            << (cfg.fragment_rows < 0 ? cfg.page_rows : cfg.fragment_rows) << ','
            << (cfg.v2_headers ? 1 : 0) << ','
            << cfg.repeats << ',' << bytes << ','
            << (rows > 0 ? 8.0 * static_cast<double>(bytes) / static_cast<double>(rows) : 0.0)
            << ','
            << (bytes > 0 ? static_cast<double>(raw) / static_cast<double>(bytes) : 0.0) << ','
            << median(write_ms) << ',' << min_of(write_ms) << ',' << max_of(write_ms) << ','
            << median(read_ms) << ',' << min_of(read_ms) << ',' << max_of(read_ms) << ','
            << gbps(median(write_ms)) << ',' << gbps(median(read_ms)) << ',' << footer.row_groups
            << ',' << footer.data_pages << ',' << footer.encodings_string() << ','
            << footer.unexpected_pages << ',' << (validated ? 1 : 0) << ',' << csv_escape(error);
        return row.str();
      };
      auto const guarded = [&](char const* op, auto&& fn) {
        error = std::string{"hang: "} + op + " did not finish within " +
                std::to_string(cfg.hang_timeout_s) + " s";
        watchdog.arm(cfg.output, format_row(), cfg.hang_timeout_s);
        auto result = fn();
        watchdog.disarm();
        error.clear();
        return result;
      };

      try {
        for (int i = 0; i < cfg.warmup + cfg.repeats; ++i) {
          auto const w = guarded("write", [&] {
            return write_column(view, column, encoding, compression, cfg, buffer, stream);
          });
          auto const r = guarded("read", [&] {
            return timed_read(
              cudf::io::source_info(cudf::host_span<char const>(buffer.data(), buffer.size())),
              cfg,
              stream);
          });
          CUDF_EXPECTS(r.second == rows, "Row count mismatch after read-back");
          if (i >= cfg.warmup) {
            write_ms.push_back(w);
            read_ms.push_back(r.first);
          }
        }
        bytes     = buffer.size();
        footer    = summarize_footer(buffer, encoding);
        validated = cfg.validate and
                    guarded("validate", [&] { return matches_source(buffer, view, cfg, stream); });
      } catch (std::exception const& e) {
        watchdog.disarm();
        error = e.what();
      }

      csv << format_row() << '\n';
      csv.flush();

      std::cerr << "[sweep] " << cfg.table << "." << column << " " << requested << "/"
                << comp_name << ": bytes=" << bytes << " write_ms=" << median(write_ms)
                << " read_ms=" << median(read_ms) << " pages=" << footer.encodings_string()
                << " validated=" << validated << (error.empty() ? "" : " error=" + error) << "\n";
    }
  }
}

void run_read(bench_options const& cfg)
{
  CUDF_EXPECTS(not cfg.inputs.empty(), "read mode requires --input");
  CUDF_EXPECTS(not cfg.output.empty(), "read mode requires --output");

  hang_watchdog watchdog;
  auto const stream       = cudf::get_default_stream();
  bool const write_header = file_is_empty(cfg.output);
  std::ofstream csv(cfg.output, std::ios::app);
  if (write_header) {
    csv << "label,file,file_bytes,rows,repeats,read_ms_median,read_ms_min,read_ms_max,rows_per_s,"
           "file_gbps,error\n";
  }

  for (auto const& path : cfg.inputs) {
    std::vector<double> read_ms;
    int64_t rows = 0;
    std::string error;
    auto const file_bytes = static_cast<int64_t>(std::filesystem::file_size(path));
    auto const format_row = [&]() {
      auto const med = median(read_ms);
      std::ostringstream row;
      row << cfg.label << ',' << path << ',' << file_bytes << ',' << rows << ',' << cfg.repeats
          << ',' << med << ',' << min_of(read_ms) << ',' << max_of(read_ms) << ','
          << (med > 0 ? static_cast<double>(rows) / (med * 1e-3) : 0.0) << ','
          << (med > 0 ? static_cast<double>(file_bytes) / (med * 1e-3) / 1e9 : 0.0) << ','
          << csv_escape(error);
      return row.str();
    };
    try {
      auto const source = cudf::io::source_info(path);
      for (int i = 0; i < cfg.warmup + cfg.repeats; ++i) {
        error = "hang: read did not finish within " + std::to_string(cfg.hang_timeout_s) + " s";
        watchdog.arm(cfg.output, format_row(), cfg.hang_timeout_s);
        auto const r = timed_read(source, cfg, stream);
        watchdog.disarm();
        error.clear();
        rows = r.second;
        if (i >= cfg.warmup) { read_ms.push_back(r.first); }
      }
    } catch (std::exception const& e) {
      watchdog.disarm();
      error = e.what();
    }
    auto const med = median(read_ms);
    csv << format_row() << '\n';
    csv.flush();
    std::cerr << "[read] " << cfg.label << " " << path << ": rows=" << rows
              << " read_ms=" << med << (error.empty() ? "" : " error=" + error) << "\n";
  }
}

void print_usage()
{
  std::cout
    << "Usage:\n"
       "  fastlanes_encoding_bench sweep --input=FILE --table=NAME --columns=a,b --output=CSV\n"
       "      [--encodings=PLAIN,DICTIONARY,DELTA_BINARY_PACKED,BYTE_STREAM_SPLIT,FASTLANES]\n"
       "      [--compressions=NONE,SNAPPY,ZSTD] [--warmup=1] [--repeats=3]\n"
       "      [--row-group-rows=122880] [--page-rows=20480] [--write-slice-row-groups=100]\n"
       "      [--fragment-rows=N] (default: --page-rows, so pages are exact; 0 = cuDF default)\n"
       "      [--v1-headers] [--no-validate] [--combos=ENCODING/CODEC,...]\n"
       "  fastlanes_encoding_bench read --input=FILE[,FILE...] --label=NAME --output=CSV\n"
       "      [--warmup=1] [--repeats=3]\n"
       "Common: [--read-batch-row-groups=256] [--pass-read-limit-mb=4096] [--hang-timeout-s=600]\n"
       "Reads are timed as batched read_parquet calls over --read-batch-row-groups row groups;\n"
       "--pass-read-limit-mb only bounds the chunked read that loads sweep columns.\n"
       "A write/read exceeding --hang-timeout-s is recorded as an error row and the program\n"
       "exits with code 3; --combos lets a driver resume the remaining cases.\n"
       "FASTLANES resolves to FASTLANES_DELTA_BINARY for INT64 columns and FASTLANE_BITPACK_RAW\n"
       "for INT32-backed columns.\n";
}

bench_options parse_args(int argc, char const** argv)
{
  CUDF_EXPECTS(argc >= 2, "missing mode");
  bench_options cfg;
  cfg.mode = argv[1];
  CUDF_EXPECTS(cfg.mode == "sweep" or cfg.mode == "read", "mode must be 'sweep' or 'read'");

  for (int i = 2; i < argc; ++i) {
    std::string const arg = argv[i];
    auto const eq         = arg.find('=');
    auto const key        = arg.substr(0, eq);
    auto const value      = eq == std::string::npos ? std::string{} : arg.substr(eq + 1);
    if (key == "--input") {
      cfg.inputs = split(value);
    } else if (key == "--table") {
      cfg.table = value;
    } else if (key == "--label") {
      cfg.label = value;
    } else if (key == "--columns") {
      cfg.columns = split(value);
    } else if (key == "--encodings") {
      cfg.encodings = split(value);
    } else if (key == "--compressions") {
      cfg.compressions = split(value);
    } else if (key == "--combos") {
      for (auto const& combo : split(value)) {
        auto const slash = combo.find('/');
        CUDF_EXPECTS(slash != std::string::npos, "--combos entries must be ENCODING/CODEC");
        cfg.combos.emplace_back(combo.substr(0, slash), combo.substr(slash + 1));
      }
    } else if (key == "--hang-timeout-s") {
      cfg.hang_timeout_s = std::stoi(value);
    } else if (key == "--output") {
      cfg.output = value;
    } else if (key == "--warmup") {
      cfg.warmup = std::stoi(value);
    } else if (key == "--repeats") {
      cfg.repeats = std::stoi(value);
    } else if (key == "--row-group-rows") {
      cfg.row_group_rows = std::stoi(value);
    } else if (key == "--page-rows") {
      cfg.page_rows = std::stoi(value);
    } else if (key == "--fragment-rows") {
      cfg.fragment_rows = std::stoi(value);
    } else if (key == "--write-slice-row-groups") {
      cfg.write_slice_row_groups = std::stoi(value);
    } else if (key == "--read-batch-row-groups") {
      cfg.read_batch_row_groups = std::stoi(value);
    } else if (key == "--pass-read-limit-mb") {
      cfg.pass_read_limit = std::stoull(value) * mib;
    } else if (key == "--v1-headers") {
      cfg.v2_headers = false;
    } else if (key == "--no-validate") {
      cfg.validate = false;
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  CUDF_EXPECTS(cfg.repeats > 0 and cfg.warmup >= 0, "invalid --repeats/--warmup");
  return cfg;
}

}  // namespace

int main(int argc, char const** argv)
{
  if (argc < 2 or std::string(argv[1]) == "-h" or std::string(argv[1]) == "--help") {
    print_usage();
    return argc < 2 ? 1 : 0;
  }

  bench_options cfg;
  try {
    cfg = parse_args(argc, argv);
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    print_usage();
    return 1;
  }

  auto resource = create_memory_resource(/*is_pool_used=*/true);
  cudf::set_current_device_resource(resource);

  try {
    if (cfg.mode == "sweep") {
      run_sweep(cfg);
    } else {
      run_read(cfg);
    }
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
