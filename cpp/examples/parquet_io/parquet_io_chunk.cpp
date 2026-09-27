/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "io_source.hpp"
#include "timer.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file parquet_io_chunk.cpp
 * @brief Demonstrates Row Group by Row Group reading and writing of Parquet files.
 *
 * This program reads a Parquet file one Row Group at a time, applies encoding
 * and compression settings, writes using chunked writer, then validates output.
 *
 * Key Features:
 * - Row Group preservation: Output row groups have same row counts as input
 * - Configurable per-column encoding via command line
 * - Configurable compression type
 * - Configurable parallelization (batch size) for GPU memory vs speed trade-off
 * - Detailed logging with per-RG timing and compression ratios
 *
 * Trade-offs and Assumptions:
 * - ASSUMPTION: All Row Groups have similar row counts. The page size calculation
 *   is based on the first Row Group and applied to all subsequent Row Groups.
 * - ASSUMPTION: Schema is consistent across all Row Groups (standard Parquet behavior).
 * - TRADE-OFF: Row Group size in output is set to match input RG row count, ensuring
 *   1:1 Row Group mapping. This may not be optimal for all use cases.
 * - TRADE-OFF: Higher parallelization (batch_size) = more GPU memory but potentially faster.
 *
 * Usage: parquet_io_chunk <input.parquet> <output.parquet> <encoding> <compression>
 *                         [options]
 */

// ============================================================================
// Configuration Constants
// ============================================================================

/**
 * @brief Default number of pages per Row Group
 */
constexpr int DEFAULT_PAGES_PER_ROW_GROUP = 200;

/**
 * @brief Default batch size for parallel RG processing
 * @note Higher values use more GPU memory but may improve throughput
 */
constexpr int DEFAULT_BATCH_SIZE = 64;

/**
 * @brief Maximum recommended batch size to prevent GPU OOM
 */
constexpr int MAX_BATCH_SIZE = 9999;

// ============================================================================
// Logger Class
// ============================================================================

/**
 * @brief Simple logger that writes to both console and optionally to a file
 */
class Logger {
 public:
  Logger() : file_enabled_(false), start_time_(std::chrono::steady_clock::now()) {}

  void enable_file_logging(std::string const& filepath)
  {
    log_file_.open(filepath, std::ios::out | std::ios::trunc);
    if (log_file_.is_open()) {
      file_enabled_ = true;
      log_filepath_ = filepath;
    } else {
      std::cerr << "Warning: Could not open log file: " << filepath << std::endl;
    }
  }

  void close()
  {
    if (log_file_.is_open()) { log_file_.close(); }
  }

  std::string get_log_filepath() const { return log_filepath_; }

  // Log with timestamp
  void log(std::string const& message, bool console_output = true)
  {
    auto now      = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    std::ostringstream oss;
    oss << "[" << std::setw(8) << std::fixed << std::setprecision(2) 
        << (duration.count() / 1000.0) << "s] " << message;
    
    std::string formatted = oss.str();
    
    if (console_output) { std::cout << formatted << std::endl; }
    if (file_enabled_) { log_file_ << formatted << std::endl; }
  }

  // Log without timestamp (for headers, etc.)
  void log_raw(std::string const& message, bool console_output = true)
  {
    if (console_output) { std::cout << message << std::endl; }
    if (file_enabled_) { log_file_ << message << std::endl; }
  }

  // Log RG processing stats
  void log_rg_stats(int rg_idx,
                    int total_rgs,
                    int64_t rows,
                    double read_time_ms,
                    double write_time_ms,
                    double total_time_ms)
  {
    std::ostringstream oss;
    oss << "  RG " << std::setw(3) << rg_idx << "/" << total_rgs << " | " << std::setw(10) << rows
        << " rows | "
        << "read: " << std::setw(7) << std::fixed << std::setprecision(2) << read_time_ms << "ms | "
        << "write: " << std::setw(7) << std::fixed << std::setprecision(2) << write_time_ms
        << "ms | "
        << "total: " << std::setw(7) << std::fixed << std::setprecision(2) << total_time_ms << "ms";
    log(oss.str());
  }

  // Log batch processing summary
  void log_batch_summary(int batch_start,
                         int batch_end,
                         int total_rgs,
                         double batch_time_ms,
                         int64_t batch_rows)
  {
    std::ostringstream oss;
    oss << "  Batch [" << batch_start << "-" << batch_end - 1 << "] completed: " << std::fixed
        << std::setprecision(2) << batch_time_ms << "ms for " << batch_rows << " rows ("
        << std::setprecision(0) << (batch_rows / (batch_time_ms / 1000.0)) << " rows/sec)";
    log(oss.str());
  }

  // Log compression summary
  void log_compression_summary(std::string const& input_file,
                               std::string const& output_file,
                               int64_t input_size,
                               int64_t output_size)
  {
    double ratio = (input_size > 0) ? (static_cast<double>(output_size) / input_size) : 0.0;
    double compression_pct = (1.0 - ratio) * 100.0;

    std::ostringstream oss;
    oss << "\n=== Compression Summary ===" << std::endl;
    oss << "  Input file:       " << input_file << std::endl;
    oss << "  Output file:      " << output_file << std::endl;
    oss << "  Input size:       " << std::setw(12) << input_size << " bytes ("
        << std::fixed << std::setprecision(2) << (input_size / 1024.0 / 1024.0) << " MB)"
        << std::endl;
    oss << "  Output size:      " << std::setw(12) << output_size << " bytes ("
        << std::fixed << std::setprecision(2) << (output_size / 1024.0 / 1024.0) << " MB)"
        << std::endl;
    oss << "  Compression ratio: " << std::fixed << std::setprecision(4) << ratio << "x"
        << std::endl;
    oss << "  Space savings:    " << std::fixed << std::setprecision(2) << compression_pct << "%";
    log_raw(oss.str());
  }

 private:
  std::ofstream log_file_;
  bool file_enabled_;
  std::string log_filepath_;
  std::chrono::steady_clock::time_point start_time_;
};

// Global logger instance
Logger g_logger;

// ============================================================================
// CLI Configuration Structure
// ============================================================================

/**
 * @brief Configuration parsed from command line arguments
 */
struct cli_config {
  std::string input_filepath  = "example.parquet";
  std::string output_filepath = "output.parquet";
  std::string encoding_arg    = "DELTA_BINARY_PACKED";

  cudf::io::column_encoding default_encoding = cudf::io::column_encoding::DELTA_BINARY_PACKED;
  std::map<std::string, cudf::io::column_encoding> encoding_map;

  cudf::io::compression_type compression = cudf::io::compression_type::ZSTD;

  bool enable_stats      = false;
  bool enable_v2_headers = false;
  bool skip_validation   = false;

  // Parallelization settings
  int batch_size = DEFAULT_BATCH_SIZE;

  // Logging settings
  bool enable_logging    = false;
  std::string log_file   = "";  // Empty means auto-generate with timestamp
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Generate a timestamp string for log filenames
 */
std::string generate_timestamp_string()
{
  auto now       = std::chrono::system_clock::now();
  auto now_time  = std::chrono::system_clock::to_time_t(now);
  auto now_local = std::localtime(&now_time);

  std::ostringstream oss;
  oss << std::put_time(now_local, "%Y%m%d_%H%M%S");
  return oss.str();
}

/**
 * @brief Get file size in bytes
 */
int64_t get_file_size(std::string const& filepath)
{
  try {
    return static_cast<int64_t>(std::filesystem::file_size(filepath));
  } catch (...) {
    return -1;
  }
}

/**
 * @brief Read a specific row group from a parquet file
 */
cudf::io::table_with_metadata read_row_group(cudf::io::source_info const& source_info,
                                             int row_group_index,
                                             cuda::stream_ref stream)
{
  auto read_options =
    cudf::io::parquet_reader_options::builder(source_info).row_groups({{row_group_index}}).build();
  return cudf::io::read_parquet(read_options, stream);
}

/**
 * @brief Read entire parquet file (for validation)
 */
cudf::io::table_with_metadata read_parquet(std::string const& filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_parquet(options);
}

// ============================================================================
// Processing Functions
// ============================================================================

/**
 * @brief Process a Parquet file Row Group by Row Group with configurable batch size.
 *
 * @param config CLI configuration
 *
 * The batch_size parameter controls how many Row Groups are queued before
 * synchronization. Higher values may improve GPU utilization but use more memory.
 */
void process_parquet_by_row_group(cli_config const& config)
{
  auto stream = cudf::get_default_stream();

  // -------------------------------------------------------------------------
  // Step 1: Read metadata to get the number of row groups and schema info
  // -------------------------------------------------------------------------
  auto const source_info    = cudf::io::source_info(config.input_filepath);
  auto const metadata       = cudf::io::read_parquet_metadata(source_info);
  auto const num_row_groups = static_cast<int>(metadata.num_rowgroups());
  auto const num_rows_total = metadata.num_rows();
  auto const input_file_size = get_file_size(config.input_filepath);

  g_logger.log_raw("=== Input File Info ===");
  g_logger.log_raw("  File:             " + config.input_filepath);
  g_logger.log_raw("  Size:             " + std::to_string(input_file_size) + " bytes (" +
                   std::to_string(input_file_size / 1024 / 1024) + " MB)");
  g_logger.log_raw("  Total Row Groups: " + std::to_string(num_row_groups));
  g_logger.log_raw("  Total Rows:       " + std::to_string(num_rows_total));
  g_logger.log_raw("");

  // -------------------------------------------------------------------------
  // Step 2: Read first row group to get schema metadata for writer
  // -------------------------------------------------------------------------
  auto first_rg             = read_row_group(source_info, 0, stream);
  auto table_input_metadata = cudf::io::table_input_metadata{first_rg.metadata};

  // Apply encodings to metadata
  g_logger.log_raw("=== Encoding Configuration ===");
  apply_encodings(
    table_input_metadata, config.encoding_map, config.default_encoding, /*verbose=*/true);
  g_logger.log_raw("");

  // -------------------------------------------------------------------------
  // Step 3: Configure writer options
  // -------------------------------------------------------------------------
  auto sink_info         = cudf::io::sink_info(config.output_filepath);
  auto first_rg_tbl_rows = first_rg.tbl->num_rows();

  auto const max_page_size_rows =
    std::max<cudf::size_type>(1, first_rg_tbl_rows / DEFAULT_PAGES_PER_ROW_GROUP);

  // Determine stats level
  auto stats_level = config.enable_stats ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                         : cudf::io::statistics_freq::STATISTICS_ROWGROUP;

  g_logger.log_raw("=== Writer Configuration ===");
  g_logger.log_raw("  Pages per RG (target): " + std::to_string(DEFAULT_PAGES_PER_ROW_GROUP));
  g_logger.log_raw("  Max page size (rows):  " + std::to_string(max_page_size_rows));
  g_logger.log_raw("  Row Group size (rows): " + std::to_string(first_rg_tbl_rows) +
                   " (from first RG)");
  g_logger.log_raw("  V2 Headers:            " +
                   std::string(config.enable_v2_headers ? "enabled" : "disabled"));
  g_logger.log_raw("  Page Statistics:       " +
                   std::string(config.enable_stats ? "enabled" : "disabled"));
  g_logger.log_raw("  Batch Size:            " + std::to_string(config.batch_size) +
                   " (parallelization level)");
  g_logger.log_raw("");

  auto writer_options_builder = cudf::io::chunked_parquet_writer_options::builder(sink_info)
                                  .metadata(table_input_metadata)
                                  .compression(config.compression)
                                  .stats_level(stats_level)
                                  .row_group_size_rows(first_rg_tbl_rows)
                                  .max_page_size_rows(max_page_size_rows)
                                  .write_v2_headers(config.enable_v2_headers);

  cudf::io::chunked_parquet_writer writer(writer_options_builder.build(), stream);

  // -------------------------------------------------------------------------
  // Step 4: Process Row Groups in batches
  // -------------------------------------------------------------------------
  g_logger.log_raw("=== Processing Row Groups (batch_size=" + std::to_string(config.batch_size) +
                   ") ===");
  timer total_timer;

  // Statistics tracking
  std::vector<double> rg_times_ms;
  int64_t total_rows_processed = 0;

  // Write first row group (already loaded)
  {
    timer rg_timer;
    g_logger.log("  Processing RG 0/" + std::to_string(num_row_groups) + " (" +
                 std::to_string(first_rg.tbl->num_rows()) + " rows)...");
    writer.write(first_rg.tbl->view());
    stream.sync();
    double elapsed_ms = rg_timer.elapsed_millis();
    rg_times_ms.push_back(elapsed_ms);
    total_rows_processed += first_rg.tbl->num_rows();
    g_logger.log("    -> completed in " + std::to_string(elapsed_ms) + " ms");
  }
  first_rg.tbl.reset();

  // Process remaining row groups in batches
  int const batch_size = config.batch_size;

  for (int batch_start = 1; batch_start < num_row_groups; batch_start += batch_size) {
    int const batch_end = std::min(batch_start + batch_size, num_row_groups);
    int const current_batch_size = batch_end - batch_start;

    timer batch_timer;
    int64_t batch_rows = 0;

    // Vector to hold tables for the current batch (keeps them alive until sync)
    std::vector<cudf::io::table_with_metadata> batch_tables;
    batch_tables.reserve(current_batch_size);

    // Read and write all RGs in this batch
    for (int rg_idx = batch_start; rg_idx < batch_end; ++rg_idx) {
      timer rg_read_timer;
      batch_tables.push_back(read_row_group(source_info, rg_idx, stream));
      double read_time_ms = rg_read_timer.elapsed_millis();

      auto& table_with_meta = batch_tables.back();
      int64_t rg_rows       = table_with_meta.tbl->num_rows();
      batch_rows += rg_rows;

      timer rg_write_timer;
      writer.write(table_with_meta.tbl->view());
      double write_time_ms = rg_write_timer.elapsed_millis();

      g_logger.log("  Queued RG " + std::to_string(rg_idx) + "/" + std::to_string(num_row_groups) +
                   " (" + std::to_string(rg_rows) + " rows) - read: " +
                   std::to_string(read_time_ms) + "ms, write: " + std::to_string(write_time_ms) +
                   "ms");
    }

    // Synchronize after the batch
    timer sync_timer;
    stream.sync();
    double sync_time_ms = sync_timer.elapsed_millis();

    double batch_time_ms = batch_timer.elapsed_millis();
    rg_times_ms.push_back(batch_time_ms);
    total_rows_processed += batch_rows;

    g_logger.log("  Batch [" + std::to_string(batch_start) + "-" + std::to_string(batch_end - 1) +
                 "] synchronized in " + std::to_string(sync_time_ms) + "ms (total batch: " +
                 std::to_string(batch_time_ms) + "ms, " + std::to_string(batch_rows) + " rows)");

    // Clear batch tables to free GPU memory
    batch_tables.clear();
  }

  // -------------------------------------------------------------------------
  // Step 5: Finalize the output file
  // -------------------------------------------------------------------------
  writer.close();

  double total_time_ms = total_timer.elapsed_millis();
  auto output_file_size = get_file_size(config.output_filepath);

  g_logger.log_raw("");
  g_logger.log_raw("=== Processing Complete ===");
  g_logger.log_raw("  Total time:       " + std::to_string(total_time_ms) + " ms (" +
                   std::to_string(total_time_ms / 1000.0) + " sec)");
  g_logger.log_raw("  Rows processed:   " + std::to_string(total_rows_processed));
  g_logger.log_raw("  Throughput:       " +
                   std::to_string(static_cast<int64_t>(total_rows_processed /
                                                       (total_time_ms / 1000.0))) +
                   " rows/sec");
  g_logger.log_raw("  Output file:      " + config.output_filepath);

  // Log compression summary
  g_logger.log_compression_summary(
    config.input_filepath, config.output_filepath, input_file_size, output_file_size);
}

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * @brief Validate output file by comparing row group by row group with input
 */
bool validate_row_group_by_row_group(std::string const& input_file, std::string const& output_file)
{
  auto stream = cudf::get_default_stream();

  g_logger.log_raw("");
  g_logger.log_raw("=== Validating Output (Row Group by Row Group) ===");

  auto const input_source  = cudf::io::source_info(input_file);
  auto const output_source = cudf::io::source_info(output_file);

  auto const input_metadata  = cudf::io::read_parquet_metadata(input_source);
  auto const output_metadata = cudf::io::read_parquet_metadata(output_source);

  auto const input_num_rg  = static_cast<int>(input_metadata.num_rowgroups());
  auto const output_num_rg = static_cast<int>(output_metadata.num_rowgroups());

  if (input_num_rg != output_num_rg) {
    g_logger.log_raw("VALIDATION FAILED: Row group count mismatch!");
    g_logger.log_raw("  Input:  " + std::to_string(input_num_rg) + " row groups");
    g_logger.log_raw("  Output: " + std::to_string(output_num_rg) + " row groups");
    return false;
  }

  bool all_passed = true;
  timer validation_timer;

  for (int rg_idx = 0; rg_idx < input_num_rg; ++rg_idx) {
    timer rg_timer;
    auto input_rg  = read_row_group(input_source, rg_idx, stream);
    auto output_rg = read_row_group(output_source, rg_idx, stream);
    stream.sync();

    try {
      check_tables_equal(input_rg.tbl->view(), output_rg.tbl->view());
      g_logger.log("  RG " + std::to_string(rg_idx) + " validated in " +
                   std::to_string(rg_timer.elapsed_millis()) + " ms");
    } catch (std::exception const& e) {
      g_logger.log_raw("VALIDATION FAILED at Row Group " + std::to_string(rg_idx) + ": " +
                       e.what());
      all_passed = false;
    }
  }

  if (all_passed) {
    g_logger.log_raw("  All " + std::to_string(input_num_rg) +
                     " row groups validated successfully in " +
                     std::to_string(validation_timer.elapsed_millis()) + " ms");
  }

  return all_passed;
}

// ============================================================================
// CLI Parsing
// ============================================================================

/**
 * @brief Check if argument is an option flag
 */
bool is_option(std::string const& arg) { return arg.size() > 2 && arg.substr(0, 2) == "--"; }

/**
 * @brief Parse command line arguments into config structure
 */
cli_config parse_args(int argc, char const** argv)
{
  cli_config config;

  // Collect positional and optional arguments separately
  std::vector<std::string> positional_args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      throw std::invalid_argument("help");
    } else if (arg == "--enable-stats") {
      config.enable_stats = true;
    } else if (arg == "--enable-v2-headers") {
      config.enable_v2_headers = true;
    } else if (arg == "--skip-validation") {
      config.skip_validation = true;
    } else if (arg.rfind("--batch-size=", 0) == 0) {
      // Parse --batch-size=N
      std::string value = arg.substr(13);
      config.batch_size = std::stoi(value);
      if (config.batch_size < 1) {
        throw std::invalid_argument("batch-size must be >= 1");
      }
      if (config.batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Warning: batch-size " << config.batch_size << " exceeds recommended max ("
                  << MAX_BATCH_SIZE << "). High GPU memory usage expected." << std::endl;
      }
    } else if (arg == "--enable-log") {
      config.enable_logging = true;
    } else if (arg.rfind("--log-file=", 0) == 0) {
      config.enable_logging = true;
      config.log_file       = arg.substr(11);
    } else if (is_option(arg)) {
      throw std::invalid_argument("Unknown option: " + arg);
    } else {
      positional_args.push_back(arg);
    }
  }

  // Process positional arguments
  if (positional_args.size() >= 1) { config.input_filepath = positional_args[0]; }
  if (positional_args.size() >= 2) { config.output_filepath = positional_args[1]; }
  if (positional_args.size() >= 3) { config.encoding_arg = positional_args[2]; }
  if (positional_args.size() >= 4) {
    config.compression = get_compression_type(positional_args[3]);
  }

  // Parse encoding argument
  if (config.encoding_arg.find(':') != std::string::npos) {
    config.encoding_map = parse_column_encodings(config.encoding_arg);
  } else {
    config.default_encoding = get_encoding_type(config.encoding_arg);
  }

  return config;
}

/**
 * @brief Print usage information
 */
void print_usage()
{
  std::cout
    << "\nUsage: parquet_io_chunk <input.parquet> <output.parquet> <encoding> <compression>\n"
       "                        [options]\n\n"
       "Positional Arguments:\n"
       "  <input.parquet>   : Path to input Parquet file\n"
       "  <output.parquet>  : Path to output Parquet file\n"
       "  <encoding>        : Encoding specification. Can be:\n"
       "                      1. A single global encoding (e.g., DELTA_BINARY_PACKED)\n"
       "                      2. A comma-separated map of \"column:encoding\" pairs\n"
       "                         (e.g., \"col_a:PLAIN,col_b:DICTIONARY\")\n"
       "  <compression>     : Compression type for output file\n\n"
       "Options:\n"
       "  --batch-size=N      : Number of Row Groups to process in parallel (default: "
    << DEFAULT_BATCH_SIZE
    << ")\n"
       "                        Higher values use more GPU memory but may improve throughput.\n"
       "                        Recommended max: "
    << MAX_BATCH_SIZE
    << "\n"
       "  --enable-stats      : Enable page-level statistics in output\n"
       "  --enable-v2-headers : Enable Parquet V2 data page headers\n"
       "  --skip-validation   : Skip validation after processing\n"
       "  --enable-log        : Enable logging to file (auto-generated filename)\n"
       "  --log-file=PATH     : Enable logging to specified file\n"
       "  -h, --help          : Show this help message\n\n"
       "Available Encoding Types:\n"
       "  DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,\n"
       "  DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY\n\n"
       "Available Compression Types:\n"
       "  NONE, AUTO, SNAPPY, LZ4, ZSTD, CASCADED, BITCOMP, GDEFLATE, ANS\n\n"
       "Examples:\n"
       "  # Basic usage with default batch size\n"
       "  ./parquet_io_chunk input.parquet output.parquet DELTA_BINARY_PACKED SNAPPY\n\n"
       "  # Higher parallelization (uses more GPU memory)\n"
       "  ./parquet_io_chunk input.parquet output.parquet PLAIN ZSTD --batch-size=4\n\n"
       "  # With logging enabled\n"
       "  ./parquet_io_chunk input.parquet output.parquet PLAIN SNAPPY --enable-log\n\n"
       "  # With specific log file\n"
       "  ./parquet_io_chunk input.parquet output.parquet PLAIN SNAPPY \\\n"
       "    --log-file=my_conversion.log --batch-size=2\n\n"
       "  # Per-column encoding with all options\n"
       "  ./parquet_io_chunk input.parquet output.parquet \\\n"
       "    \"id:DELTA_BINARY_PACKED,name:DICTIONARY\" ZSTD \\\n"
       "    --batch-size=4 --enable-v2-headers --enable-log\n\n"
       "Notes:\n"
       "  - Row Groups in output preserve the same row counts as input Row Groups.\n"
       "  - Page size is computed based on first Row Group (assumes similar RG sizes).\n"
       "  - Default pages per Row Group: "
    << DEFAULT_PAGES_PER_ROW_GROUP
    << "\n"
       "  - Higher batch-size values queue more RGs before GPU sync, potentially improving\n"
       "    throughput but using more GPU memory. Monitor GPU memory usage.\n\n";
}

/**
 * @brief Get compression string for display
 */
std::string get_compression_string(cudf::io::compression_type compression)
{
  switch (compression) {
    case cudf::io::compression_type::NONE: return "NONE";
    case cudf::io::compression_type::AUTO: return "AUTO";
    case cudf::io::compression_type::SNAPPY: return "SNAPPY";
    case cudf::io::compression_type::LZ4: return "LZ4";
    case cudf::io::compression_type::ZSTD: return "ZSTD";
    // case cudf::io::compression_type::CASCADED: return "CASCADED";
    // case cudf::io::compression_type::BITCOMP: return "BITCOMP";
    // case cudf::io::compression_type::GDEFLATE: return "GDEFLATE";
    // case cudf::io::compression_type::ANS: return "ANS";
    default: return "UNKNOWN";
  }
}

/**
 * @brief Main function
 */
int main(int argc, char const** argv)
{
  if (argc < 2) {
    print_usage();
    return 0;
  }

  cli_config config;

  try {
    config = parse_args(argc, argv);
  } catch (std::invalid_argument const& e) {
    if (std::string(e.what()) == "help") {
      print_usage();
      return 0;
    }
    std::cerr << "Error: " << e.what() << std::endl;
    print_usage();
    return 1;
  }

  // Setup logging if enabled
  if (config.enable_logging) {
    std::string log_filepath = config.log_file;
    if (log_filepath.empty()) {
      // Auto-generate log filename with timestamp
      std::string timestamp   = generate_timestamp_string();
      std::string compression = get_compression_string(config.compression);
      log_filepath = "parquet_chunk_" + compression + "_" + timestamp + ".log";
    }
    g_logger.enable_file_logging(log_filepath);
  }

  // Print configuration
  g_logger.log_raw("=== Parquet Row Group Chunked Processing ===");
  g_logger.log_raw("  Input file:    " + config.input_filepath);
  g_logger.log_raw("  Output file:   " + config.output_filepath);
  if (config.encoding_map.empty()) {
    g_logger.log_raw("  Encoding:      " + get_encoding_string(config.default_encoding) +
                     " (global)");
  } else {
    g_logger.log_raw("  Encoding:      Per-column map (" +
                     std::to_string(config.encoding_map.size()) + " columns)");
  }
  g_logger.log_raw("  Compression:   " + get_compression_string(config.compression));
  g_logger.log_raw("  Batch Size:    " + std::to_string(config.batch_size));
  g_logger.log_raw("  V2 Headers:    " +
                   std::string(config.enable_v2_headers ? "enabled" : "disabled"));
  g_logger.log_raw("  Page Stats:    " +
                   std::string(config.enable_stats ? "enabled" : "disabled"));
  g_logger.log_raw("  Validation:    " +
                   std::string(config.skip_validation ? "skipped" : "enabled"));
  if (config.enable_logging) {
    g_logger.log_raw("  Log file:      " + g_logger.get_log_filepath());
  }
  g_logger.log_raw("");

  // Initialize memory resource
  auto resource = init_memory_resource(/*is_pool_used=*/true);
  cudf::set_current_device_resource(resource);

  try {
    // Step 1: Process file row group by row group
    process_parquet_by_row_group(config);

    // Step 2: Validate (unless skipped)
    if (!config.skip_validation) {
      if (!validate_row_group_by_row_group(config.input_filepath, config.output_filepath)) {
        g_logger.close();
        return 1;
      }
    }

    g_logger.log_raw("");
    g_logger.log_raw("=== SUCCESS ===");

  } catch (std::exception const& e) {
    g_logger.log_raw("ERROR: " + std::string(e.what()));
    g_logger.close();
    return 1;
  }

  g_logger.close();
  return 0;
}