#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fastlanes/fastlanes_encode.cuh>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream_ref>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// =============================================================================
// Test Configuration
// =============================================================================

enum class DataPattern { SEQUENTIAL, RANDOM };

struct TestConfig {
  cudf::type_id type_id;
  size_t num_rows;
  size_t max_page_rows;
  size_t max_page_bytes;
  DataPattern pattern;
  uint64_t max_random_value;  // For RANDOM pattern
  int64_t seq_start;          // For SEQUENTIAL pattern
  std::string name;
};

// =============================================================================
// Data Generation Helpers
// =============================================================================

template <typename T>
std::vector<T> generate_sequential_data(size_t count, int64_t start = 100)
{
  std::vector<T> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = static_cast<T>(start + static_cast<int64_t>(i));
  }
  return data;
}

template <typename T>
std::vector<T> generate_random_data(size_t count, uint64_t max_value, unsigned seed = 42)
{
  std::vector<T> data(count);
  std::mt19937_64 rng(seed);

  if constexpr (std::is_signed_v<T>) {
    // For signed types, generate values in range [0, max_value]
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
// Column Creation Helper
// =============================================================================

template <typename T>
std::unique_ptr<cudf::column> create_column(const std::vector<T>& host_data,
                                            cuda::stream_ref stream)
{
  rmm::device_uvector<T> d_data(host_data.size(), stream);
  cudaMemcpyAsync(d_data.data(),
                  host_data.data(),
                  host_data.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaStreamSynchronize(stream.get());

  cudf::type_id tid;
  if constexpr (std::is_same_v<T, int32_t>) {
    tid = cudf::type_id::INT32;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    tid = cudf::type_id::UINT32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    tid = cudf::type_id::INT64;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    tid = cudf::type_id::UINT64;
  }

  return std::make_unique<cudf::column>(
    cudf::data_type{tid},
    static_cast<cudf::size_type>(host_data.size()),
    rmm::device_buffer{d_data.data(), d_data.size() * sizeof(T), stream},
    rmm::device_buffer{},
    0);
}

// =============================================================================
// Verification Helper
// =============================================================================

template <typename T>
bool verify_roundtrip(const std::vector<T>& original,
                      const cudf::column_view& result_col,
                      const std::string& test_name)
{
  if (static_cast<size_t>(result_col.size()) != original.size()) {
    std::cerr << "[" << test_name << "] Size mismatch! Expected " << original.size() << " but got "
              << result_col.size() << "\n";
    return false;
  }

  std::vector<T> result_data(result_col.size());
  cudaMemcpy(result_data.data(),
             result_col.head<T>(),
             result_col.size() * sizeof(T),
             cudaMemcpyDeviceToHost);

  bool match = (original == result_data);
  if (!match) {
    std::cerr << "[" << test_name << "] Data mismatch!\n";
    std::cerr << "  First 10 original: ";
    for (size_t i = 0; i < std::min<size_t>(10, original.size()); ++i) {
      std::cerr << original[i] << " ";
    }
    std::cerr << "\n  First 10 result:   ";
    for (size_t i = 0; i < std::min<size_t>(10, result_data.size()); ++i) {
      std::cerr << result_data[i] << " ";
    }
    std::cerr << "\n";
  }
  return match;
}

// =============================================================================
// Type-Dispatched Test Runner
// =============================================================================

template <typename T>
bool run_single_type_test(const TestConfig& config)
{
  std::cout << "\n--- Testing " << config.name << " ---\n";
  std::cout << "Type: " << sizeof(T) * 8 << "-bit " << (std::is_signed_v<T> ? "signed" : "unsigned")
            << "\n";
  std::cout << "Rows: " << config.num_rows << ", MaxPageRows: " << config.max_page_rows
            << ", MaxPageBytes: " << config.max_page_bytes << "\n";

  // Generate data
  std::vector<T> host_data;
  if (config.pattern == DataPattern::SEQUENTIAL) {
    host_data = generate_sequential_data<T>(config.num_rows, config.seq_start);
    std::cout << "Pattern: SEQUENTIAL (start=" << config.seq_start << ")\n";
  } else {
    host_data = generate_random_data<T>(config.num_rows, config.max_random_value);
    std::cout << "Pattern: RANDOM (max=" << config.max_random_value << ")\n";
  }

  std::cout << "First 10 values: ";
  for (size_t i = 0; i < std::min<size_t>(10, host_data.size()); ++i) {
    if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
      std::cout << static_cast<int64_t>(host_data[i]) << " ";
    } else {
      std::cout << static_cast<int32_t>(host_data[i]) << " ";
    }
  }
  std::cout << "\n";

  // Create column
  auto col = create_column(host_data, cudf::get_default_stream());
  std::vector<cudf::column_view> cols{col->view()};
  cudf::table_view input(cols);

  // Setup metadata
  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name(config.name);
  auto const requested_encoding =
    (config.type_id == cudf::type_id::INT64 || config.type_id == cudf::type_id::UINT64)
      ? cudf::io::column_encoding::FASTLANE_BITPACK_SPLIT64
      : cudf::io::column_encoding::FASTLANE_BITPACK_RAW;
  metadata.column_metadata[0].set_encoding(requested_encoding);

  // Write
  std::string filepath = "test_" + config.name + ".parquet";
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(metadata)
      .write_v2_headers(true)
      .max_page_size_bytes(config.max_page_bytes)
      .max_page_size_rows(config.max_page_rows);

  std::cout << "Writing parquet file...\n";
  cudf::io::write_parquet(out_opts);
  std::cout << "Write complete!\n";

  // Read back
  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  // Verify
  bool success = verify_roundtrip(host_data, result.tbl->view().column(0), config.name);
  std::cout << "Result: " << (success ? "PASSED" : "FAILED") << "\n";

  std::remove(filepath.c_str());
  return success;
}

// =============================================================================
// Main Test
// =============================================================================

int main()
{
  std::cout << "=== FastLanes Multi-Page Multi-Type Test ===\n";

  bool all_passed = true;

  // Test configurations for different types
  std::vector<TestConfig> configs = {
    // INT32 tests
    // {cudf::type_id::INT32, 50000, 512, 4096, DataPattern::SEQUENTIAL, 0, 100, "int32_seq"},
    // {cudf::type_id::INT32, 50000, 512, 4096, DataPattern::RANDOM, 1000000, 0, "int32_rand"},

    // UINT32 tests
    {cudf::type_id::UINT32, 2048, 1024, 4096, DataPattern::SEQUENTIAL, 0, 100, "uint32_seq"},
    // {cudf::type_id::UINT32, 50000, 512, 4096, DataPattern::RANDOM, 1000000, 0, "uint32_rand"},

    // // INT64 tests
    // {cudf::type_id::INT64, 50000, 512, 4096, DataPattern::SEQUENTIAL, 0, 100, "int64_seq"},
    // {cudf::type_id::INT64,
    //  50000,
    //  512,
    //  4096,
    //  DataPattern::RANDOM,
    //  1000000000000ULL,
    //  0,
    //  "int64_rand"},

    // // UINT64 tests
    // {cudf::type_id::UINT64, 50000, 512, 4096, DataPattern::SEQUENTIAL, 0, 100, "uint64_seq"},
    // {cudf::type_id::UINT64,
    //  50000,
    //  512,
    //  4096,
    //  DataPattern::RANDOM,
    //  1000000000000ULL,
    //  0,
    //  "uint64_rand"},
  };

  for (const auto& config : configs) {
    bool result = false;
    switch (config.type_id) {
      case cudf::type_id::INT32: result = run_single_type_test<int32_t>(config); break;
      case cudf::type_id::UINT32: result = run_single_type_test<uint32_t>(config); break;
      // case cudf::type_id::INT64: result = run_single_type_test<int64_t>(config); break;
      // case cudf::type_id::UINT64: result = run_single_type_test<uint64_t>(config); break;
      default: std::cerr << "Unsupported type!\n"; result = false;
    }
    all_passed = all_passed && result;
  }

  std::cout << "\n============================================\n";
  std::cout << "FINAL RESULT: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
  std::cout << "============================================\n";

  return all_passed ? 0 : 1;
}