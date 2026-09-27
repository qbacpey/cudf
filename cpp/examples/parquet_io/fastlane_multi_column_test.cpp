/**
 * @file fastlane_multi_column_test.cpp
 * @brief Comprehensive test for FastLanes encoding with:
 *   - Multiple columns with different types (INT32, UINT32, INT64, UINT64)
 *   - Configurable number of pages per column
 *   - Support for sequential and random data patterns
 *   - Configurable max random values
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fastlanes/fastlanes_encode.cuh>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream_ref>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <variant>
#include <vector>

// =============================================================================
// Configuration Types
// =============================================================================

enum class DataPattern { SEQUENTIAL, RANDOM };

struct ColumnConfig {
  cudf::type_id type_id;
  std::string name;
  DataPattern pattern;
  int64_t seq_start;          // For SEQUENTIAL pattern
  uint64_t max_random_value;  // For RANDOM pattern
  unsigned random_seed;       // Seed for reproducibility
};

struct TestConfig {
  std::vector<ColumnConfig> columns;
  size_t num_rows;
  size_t max_page_rows;
  size_t max_page_bytes;
  std::string test_name;
};

// =============================================================================
// Data Generation Helpers
// =============================================================================

template <typename T>
std::vector<T> generate_sequential_data(size_t count, int64_t start)
{
  std::vector<T> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = static_cast<T>(start + static_cast<int64_t>(i));
  }
  return data;
}

template <typename T>
std::vector<T> generate_random_data(size_t count, uint64_t max_value, unsigned seed)
{
  std::vector<T> data(count);
  std::mt19937_64 rng(seed);

  if constexpr (std::is_signed_v<T>) {
    std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(max_value));
    for (size_t i = 0; i < count; ++i) {
      data[i] = static_cast<T>(dist(rng));
    }
  } else {
    std::uniform_int_distribution<uint64_t> dist(0, max_value);
    for (size_t i = 0; i < count; ++i) {
      data[i] = static_cast<T>(dist(rng));
    }
  }
  return data;
}

// =============================================================================
// Type-Erased Column Data Holder
// =============================================================================

using ColumnData = std::variant<std::vector<int32_t>,
                                std::vector<uint32_t>,
                                std::vector<int64_t>,
                                std::vector<uint64_t>>;

ColumnData generate_column_data(const ColumnConfig& config, size_t num_rows)
{
  switch (config.type_id) {
    case cudf::type_id::INT32:
      if (config.pattern == DataPattern::SEQUENTIAL) {
        return generate_sequential_data<int32_t>(num_rows, config.seq_start);
      } else {
        return generate_random_data<int32_t>(num_rows, config.max_random_value, config.random_seed);
      }
    case cudf::type_id::UINT32:
      if (config.pattern == DataPattern::SEQUENTIAL) {
        return generate_sequential_data<uint32_t>(num_rows, config.seq_start);
      } else {
        return generate_random_data<uint32_t>(num_rows, config.max_random_value, config.random_seed);
      }
    case cudf::type_id::INT64:
      if (config.pattern == DataPattern::SEQUENTIAL) {
        return generate_sequential_data<int64_t>(num_rows, config.seq_start);
      } else {
        return generate_random_data<int64_t>(num_rows, config.max_random_value, config.random_seed);
      }
    case cudf::type_id::UINT64:
      if (config.pattern == DataPattern::SEQUENTIAL) {
        return generate_sequential_data<uint64_t>(num_rows, config.seq_start);
      } else {
        return generate_random_data<uint64_t>(num_rows, config.max_random_value, config.random_seed);
      }
    default: throw std::runtime_error("Unsupported type");
  }
}

// =============================================================================
// Column Creation Helper
// =============================================================================

template <typename T>
std::unique_ptr<cudf::column> create_column_impl(const std::vector<T>& host_data,
                                                  cudf::type_id tid,
                                                  cuda::stream_ref stream)
{
  rmm::device_uvector<T> d_data(host_data.size(), stream);
  cudaMemcpyAsync(d_data.data(),
                  host_data.data(),
                  host_data.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaStreamSynchronize(stream.get());

  return std::make_unique<cudf::column>(
    cudf::data_type{tid},
    static_cast<cudf::size_type>(host_data.size()),
    rmm::device_buffer{d_data.data(), d_data.size() * sizeof(T), stream},
    rmm::device_buffer{},
    0);
}

std::unique_ptr<cudf::column> create_column(const ColumnData& data,
                                             cudf::type_id tid,
                                             cuda::stream_ref stream)
{
  return std::visit(
    [&](const auto& vec) { return create_column_impl(vec, tid, stream); }, data);
}

// =============================================================================
// Verification Helper
// =============================================================================

template <typename T>
bool verify_column(const std::vector<T>& original,
                   const cudf::column_view& result_col,
                   const std::string& col_name)
{
  if (static_cast<size_t>(result_col.size()) != original.size()) {
    std::cerr << "[" << col_name << "] Size mismatch! Expected " << original.size() << " but got "
              << result_col.size() << "\n";
    return false;
  }

  std::vector<T> result_data(result_col.size());
  cudaMemcpy(
    result_data.data(), result_col.head<T>(), result_col.size() * sizeof(T), cudaMemcpyDeviceToHost);

  bool match = (original == result_data);
  if (!match) {
    std::cerr << "[" << col_name << "] Data mismatch!\n";
    std::cerr << "  First 10 original: ";
    for (size_t i = 0; i < std::min<size_t>(10, original.size()); ++i) {
      std::cerr << static_cast<int64_t>(original[i]) << " ";
    }
    std::cerr << "\n  First 10 result:   ";
    for (size_t i = 0; i < std::min<size_t>(10, result_data.size()); ++i) {
      std::cerr << static_cast<int64_t>(result_data[i]) << " ";
    }
    std::cerr << "\n";
  }
  return match;
}

bool verify_column_data(const ColumnData& original,
                        const cudf::column_view& result_col,
                        const std::string& col_name)
{
  return std::visit(
    [&](const auto& vec) { return verify_column(vec, result_col, col_name); }, original);
}

// =============================================================================
// Print Helpers
// =============================================================================

void print_column_data(const ColumnData& data, const std::string& name, size_t max_print = 10)
{
  std::cout << "  " << name << " first " << max_print << " values: ";
  std::visit(
    [&](const auto& vec) {
      for (size_t i = 0; i < std::min(max_print, vec.size()); ++i) {
        if constexpr (std::is_same_v<std::decay_t<decltype(vec[i])>, uint64_t>) {
          std::cout << static_cast<uint64_t>(vec[i]) << " ";
        } else {
          std::cout << static_cast<int64_t>(vec[i]) << " ";
        }
      }
    },
    data);
  std::cout << "\n";
}

const char* type_id_to_string(cudf::type_id tid)
{
  switch (tid) {
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT64: return "UINT64";
    default: return "UNKNOWN";
  }
}

const char* pattern_to_string(DataPattern p)
{
  return p == DataPattern::SEQUENTIAL ? "SEQUENTIAL" : "RANDOM";
}

// =============================================================================
// Main Test Runner
// =============================================================================

bool run_multi_column_test(const TestConfig& config)
{
  std::cout << "\n============================================================\n";
  std::cout << "Test: " << config.test_name << "\n";
  std::cout << "============================================================\n";
  std::cout << "Rows: " << config.num_rows << ", MaxPageRows: " << config.max_page_rows
            << ", MaxPageBytes: " << config.max_page_bytes << "\n";
  std::cout << "Columns: " << config.columns.size() << "\n";

  // Generate data for all columns
  std::vector<ColumnData> all_data;
  all_data.reserve(config.columns.size());

  for (const auto& col_config : config.columns) {
    std::cout << "  - " << col_config.name << ": " << type_id_to_string(col_config.type_id) << ", "
              << pattern_to_string(col_config.pattern);
    if (col_config.pattern == DataPattern::SEQUENTIAL) {
      std::cout << " (start=" << col_config.seq_start << ")";
    } else {
      std::cout << " (max=" << col_config.max_random_value << ", seed=" << col_config.random_seed
                << ")";
    }
    std::cout << "\n";

    all_data.push_back(generate_column_data(col_config, config.num_rows));
  }

  std::cout << "\nGenerated data:\n";
  for (size_t i = 0; i < config.columns.size(); ++i) {
    print_column_data(all_data[i], config.columns[i].name);
  }

  // Create cudf columns
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(config.columns.size());

  for (size_t i = 0; i < config.columns.size(); ++i) {
    columns.push_back(
      create_column(all_data[i], config.columns[i].type_id, cudf::get_default_stream()));
  }

  // Create table view
  std::vector<cudf::column_view> col_views;
  col_views.reserve(columns.size());
  for (const auto& col : columns) {
    col_views.push_back(col->view());
  }
  cudf::table_view input(col_views);

  // Setup metadata
  cudf::io::table_input_metadata metadata(input);
  for (size_t i = 0; i < config.columns.size(); ++i) {
    metadata.column_metadata[i].set_name(config.columns[i].name);
    auto const requested_encoding =
      (config.columns[i].type_id == cudf::type_id::INT64 ||
       config.columns[i].type_id == cudf::type_id::UINT64)
        ? cudf::io::column_encoding::FASTLANE_BITPACK_SPLIT64
        : cudf::io::column_encoding::FASTLANE_BITPACK_RAW;
    metadata.column_metadata[i].set_encoding(requested_encoding);
  }

  // Write
  std::string filepath = "test_" + config.test_name + ".parquet";
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(metadata)
      .write_v2_headers(true)
      .max_page_size_bytes(config.max_page_bytes)
      .max_page_size_rows(config.max_page_rows);

  std::cout << "\nWriting parquet file...\n";
  cudf::io::write_parquet(out_opts);
  std::cout << "Write complete!\n";

  // Read back
  std::cout << "Reading back...\n";
  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  // Verify all columns
  bool all_match = true;
  std::cout << "\nVerification:\n";
  for (size_t i = 0; i < config.columns.size(); ++i) {
    bool col_match =
      verify_column_data(all_data[i], result.tbl->view().column(i), config.columns[i].name);
    std::cout << "  " << config.columns[i].name << ": " << (col_match ? "PASSED" : "FAILED")
              << "\n";
    all_match = all_match && col_match;
  }

  std::cout << "\nTest Result: " << (all_match ? "PASSED" : "FAILED") << "\n";

  std::remove(filepath.c_str());
  return all_match;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
  std::cout << "=== FastLanes Multi-Column Comprehensive Test ===\n";

  bool all_passed = true;

  // Test 1: Mixed types, sequential data, small pages
  {
    TestConfig config;
    config.test_name      = "mixed_types_sequential";
    config.num_rows       = 10000;
    config.max_page_rows  = 2048;
    config.max_page_bytes = 4096;
    config.columns        = {
      {cudf::type_id::INT32, "col_int32", DataPattern::SEQUENTIAL, 100, 0, 0},
      {cudf::type_id::UINT32, "col_uint32", DataPattern::SEQUENTIAL, 200, 0, 0},
      // {cudf::type_id::INT64, "col_int64", DataPattern::SEQUENTIAL, 300, 0, 0},
      // {cudf::type_id::UINT64, "col_uint64", DataPattern::SEQUENTIAL, 400, 0, 0},
    };
    all_passed &= run_multi_column_test(config);
  }

  // Test 2: Mixed types, random data, small pages
  {
    TestConfig config;
    config.test_name      = "mixed_types_random";
    config.num_rows       = 10000;
    config.max_page_rows  = 2048;
    config.max_page_bytes = 4096;
    config.columns        = {
      {cudf::type_id::INT32, "col_int32", DataPattern::RANDOM, 0, 100000, 42},
      {cudf::type_id::UINT32, "col_uint32", DataPattern::RANDOM, 0, 100000, 43},
      // {cudf::type_id::INT64, "col_int64", DataPattern::RANDOM, 0, 10000000000ULL, 44},
      // {cudf::type_id::UINT64, "col_uint64", DataPattern::RANDOM, 0, 10000000000ULL, 45},
    };
    all_passed &= run_multi_column_test(config);
  }

  // Test 3: Same type (INT32), different patterns
  {
    TestConfig config;
    config.test_name      = "int32_mixed_patterns";
    config.num_rows       = 20000;
    config.max_page_rows  = 1024;
    config.max_page_bytes = 8192;
    config.columns        = {
      {cudf::type_id::INT32, "seq_small", DataPattern::SEQUENTIAL, 0, 0, 0},
      {cudf::type_id::INT32, "seq_large", DataPattern::SEQUENTIAL, 1000000, 0, 0},
      {cudf::type_id::INT32, "rand_small", DataPattern::RANDOM, 0, 255, 100},
      {cudf::type_id::INT32, "rand_large", DataPattern::RANDOM, 0, 1000000000, 101},
    };
    all_passed &= run_multi_column_test(config);
  }

  // Test 4: Large data, single page per column
  {
    TestConfig config;
    config.test_name      = "large_single_page";
    config.num_rows       = 5000;
    config.max_page_rows  = 100000;  // Large enough to fit all in one page
    config.max_page_bytes = 1000000;
    config.columns        = {
      {cudf::type_id::INT32, "int32_big", DataPattern::SEQUENTIAL, 0, 0, 0},
      // {cudf::type_id::INT64, "int64_big", DataPattern::RANDOM, 0, 999999999999ULL, 200},
    };
    all_passed &= run_multi_column_test(config);
  }

  // Test 5: Very small random values (tests bitwidth optimization)
  // {
  //   TestConfig config;
  //   config.test_name      = "small_values_bitwidth";
  //   config.num_rows       = 8000;
  //   config.max_page_rows  = 512;
  //   config.max_page_bytes = 4096;
  //   config.columns        = {
  //     {cudf::type_id::INT32, "max_1bit", DataPattern::RANDOM, 0, 1, 300},     // 1-bit values
  //     {cudf::type_id::INT32, "max_4bit", DataPattern::RANDOM, 0, 15, 301},    // 4-bit values
  //     {cudf::type_id::INT32, "max_8bit", DataPattern::RANDOM, 0, 255, 302},   // 8-bit values
  //     // {cudf::type_id::INT64, "max_16bit", DataPattern::RANDOM, 0, 65535, 303}, // 16-bit values
  //   };
  //   all_passed &= run_multi_column_test(config);
  // }

  // Test 6: Many small pages
  // {
  //   TestConfig config;
  //   config.test_name      = "many_small_pages";
  //   config.num_rows       = 50000;
  //   config.max_page_rows  = 128;  // Very small pages
  //   config.max_page_bytes = 1024;
  //   config.columns        = {
  //     {cudf::type_id::UINT32, "uint32_many_pages", DataPattern::SEQUENTIAL, 0, 0, 0},
  //     // {cudf::type_id::UINT64, "uint64_many_pages", DataPattern::SEQUENTIAL, 0, 0, 0},
  //   };
  //   all_passed &= run_multi_column_test(config);
  // }

  std::cout << "\n============================================================\n";
  std::cout << "FINAL RESULT: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
  std::cout << "============================================================\n";

  return all_passed ? 0 : 1;
}
