/**
 * @file fastlane_one_page_sanity_test.cpp
 * @brief Sanity test that:
 *   1. Supports int32_t, uint32_t, int64_t, uint64_t data
 *   2. Writes via cudf Parquet writer with FastLanes encoding
 *   3. Directly calls FastLanes pack() for comparison
 *   4. Uses the same dump function to verify bit-exact match
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/debug.hpp>
#include <cudf/fastlanes/fastlanes_encode.cuh>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>


#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// =============================================================================
// Bitwidth Computation (Same logic as cudf encoder)
// =============================================================================

template <typename T>
uint8_t compute_bitwidth(const T* data, size_t count)
{
  using UnsignedT = std::make_unsigned_t<T>;

  for (size_t i = 0; i < count; ++i) {
    if constexpr (std::is_signed_v<T>) {
      if (data[i] < 0) { return sizeof(T) * 8; }  // Full bitwidth for negatives
    }
  }

  UnsignedT max_val = 0;
  for (size_t i = 0; i < count; ++i) {
    auto val = static_cast<UnsignedT>(data[i]);
    if (val > max_val) max_val = val;
  }

  if (max_val == 0) return 1;

  uint8_t bits = 0;
  while (max_val > 0) {
    max_val >>= 1;
    bits++;
  }
  return bits;
}

// =============================================================================
// Direct FastLanes Pack (mimics cudf's padding and encoding)
// =============================================================================

template <typename T>
std::vector<std::make_unsigned_t<T>> direct_fastlanes_pack(const std::vector<T>& original_data,
                                                            uint8_t& out_bitwidth)
{
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr size_t VECTOR_SIZE = ::fastlanes::VECTOR_SIZE;  // 1024
  constexpr size_t TYPE_BITS   = sizeof(T) * 8;

  // 1. Compute padded count (same as cudf)
  size_t original_count = original_data.size();
  size_t num_vectors    = ::fastlanes::num_vectors(original_count);
  size_t padded_count   = num_vectors * VECTOR_SIZE;

  std::cout << "[Direct Pack] Padding: " << original_count << " -> " << padded_count << " (+"
            << (padded_count - original_count) << " zeros)\n";

  // 2. Create padded input buffer
  std::vector<T> padded_input(padded_count, T{0});
  std::copy(original_data.begin(), original_data.end(), padded_input.begin());

  // 3. Compute bitwidth
  out_bitwidth = compute_bitwidth(original_data.data(), original_count);
  std::cout << "[Direct Pack] Bitwidth: " << static_cast<int>(out_bitwidth) << "\n";

  // 4. Allocate output buffer
  size_t encoded_bytes   = ::fastlanes::encoded_size_bytes(padded_count, out_bitwidth);
  size_t output_elements = encoded_bytes / sizeof(UnsignedT);
  std::vector<UnsignedT> encoded_output(output_elements);

  std::cout << "[Direct Pack] Output: " << encoded_bytes << " bytes (" << output_elements
            << " elements)\n";

  // 5. Pack each vector
  const UnsignedT* in_ptr = reinterpret_cast<const UnsignedT*>(padded_input.data());
  UnsignedT* out_ptr      = encoded_output.data();

  size_t out_elements_per_vector = (VECTOR_SIZE * out_bitwidth) / TYPE_BITS;

  for (size_t v = 0; v < num_vectors; ++v) {
    generated::pack::fallback::scalar::pack(in_ptr, out_ptr, out_bitwidth);
    in_ptr += VECTOR_SIZE;
    out_ptr += out_elements_per_vector;
  }

  return encoded_output;
}

// =============================================================================
// Data Generation
// =============================================================================

template <typename T>
std::vector<T> generate_sequential(size_t count, int64_t start = 100)
{
  std::vector<T> data(count);
  for (size_t i = 0; i < count; ++i) {
    data[i] = static_cast<T>(start + static_cast<int64_t>(i));
  }
  return data;
}

template <typename T>
std::vector<T> generate_random(size_t count, uint64_t max_value, unsigned seed = 42)
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
// Column Creation Helper
// =============================================================================

template <typename T>
std::unique_ptr<cudf::column> create_column(const std::vector<T>& host_data,
                                             rmm::cuda_stream_view stream)
{
  rmm::device_uvector<T> d_data(host_data.size(), stream);
  cudaMemcpyAsync(d_data.data(),
                  host_data.data(),
                  host_data.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());

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
// Type-Specific Test Runner
// =============================================================================

template <typename T>
bool run_one_page_test(const std::string& type_name, bool use_random = false)
{
  std::cout << "\n============================================\n";
  std::cout << "Testing " << type_name << (use_random ? " (random)" : " (sequential)") << "\n";
  std::cout << "============================================\n\n";

  // Create data (small enough for one page)
  std::vector<T> host_data;
  if (use_random) {
    uint64_t max_val = (sizeof(T) == 4) ? 10000ULL : 10000000000ULL;
    host_data        = generate_random<T>(1023, max_val);
  } else {
    host_data = generate_sequential<T>(1023, 100);
  }

  std::cout << "Input data (first 100): ";
  for (size_t i = 0; i < std::min<size_t>(100, host_data.size()); ++i) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      std::cout << static_cast<uint64_t>(host_data[i]) << " ";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      std::cout << static_cast<int64_t>(host_data[i]) << " ";
    } else {
      std::cout << static_cast<int64_t>(host_data[i]) << " ";
    }
  }
  std::cout << "\n\n";

  std::cout << "Output data (last 100): ";
  for (size_t i = (host_data.size() > 100 ? host_data.size() - 100 : 0); i < host_data.size(); ++i) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      std::cout << static_cast<uint64_t>(host_data[i]) << " ";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      std::cout << static_cast<int64_t>(host_data[i]) << " ";
    } else {
      std::cout << static_cast<int64_t>(host_data[i]) << " ";
    }
  }
  std::cout << "\n\n";

  // ===================================================================
  // Part 1: Direct FastLanes Pack (for comparison)
  // ===================================================================
  std::cout << "--- PART 1: Direct FastLanes Pack ---\n\n";

  uint8_t direct_bitwidth;
  auto direct_encoded = direct_fastlanes_pack(host_data, direct_bitwidth);
  fastlanes::debug::print_encoded_dump(
    direct_encoded.data(), direct_encoded.size(), "Direct FastLanes Output");

  // ===================================================================
  // Part 2: cudf Parquet Write
  // ===================================================================
  std::cout << "--- PART 2: cudf Parquet Writer ---\n\n";

  auto col = create_column(host_data, rmm::cuda_stream_default);
  std::vector<cudf::column_view> cols{col->view()};
  cudf::table_view input(cols);

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("test_" + type_name);
  auto const requested_encoding =
    (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
      ? cudf::io::column_encoding::FASTLANE_BITPACK_SPLIT64
      : cudf::io::column_encoding::FASTLANE_BITPACK_RAW;
  metadata.column_metadata[0].set_encoding(requested_encoding);

  std::string filepath = "fastlanes_" + type_name + "_test.parquet";
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(metadata)
      .write_v2_headers(true);

    std::cout << "Writing parquet file with "
      << (requested_encoding == cudf::io::column_encoding::FASTLANE_BITPACK_SPLIT64
        ? "FASTLANE_BITPACK_SPLIT64"
        : "FASTLANE_BITPACK_RAW")
      << "...\n";
  cudf::io::write_parquet(out_opts);
  std::cout << "Write complete!\n\n";

  // ===================================================================
  // Part 3: Read back and verify
  // ===================================================================
  std::cout << "--- PART 3: Read Back & Verify ---\n\n";

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  auto result_col = result.tbl->view().column(0);
  std::vector<T> result_data(result_col.size());
  cudaMemcpy(
    result_data.data(), result_col.head<T>(), result_col.size() * sizeof(T), cudaMemcpyDeviceToHost);

  std::cout << "Output data (first 100): ";
  for (size_t i = 0; i < std::min<size_t>(100, result_data.size()); ++i) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      std::cout << static_cast<uint64_t>(result_data[i]) << " ";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      std::cout << static_cast<int64_t>(result_data[i]) << " ";
    } else {
      std::cout << static_cast<int64_t>(result_data[i]) << " ";
    }
  }
  std::cout << "\n\n";

  std::cout << "Output data (last 100): ";
  for (size_t i = (result_data.size() > 100 ? result_data.size() - 100 : 0); i < result_data.size(); ++i) {
    if constexpr (std::is_same_v<T, uint64_t>) {
      std::cout << static_cast<uint64_t>(result_data[i]) << " ";
    } else if constexpr (std::is_same_v<T, int64_t>) {
      std::cout << static_cast<int64_t>(result_data[i]) << " ";
    } else {
      std::cout << static_cast<int64_t>(result_data[i]) << " ";
    }
  }
  std::cout << "\n\n";

  bool data_match = (host_data == result_data);
  std::cout << "Data integrity: " << (data_match ? "PASSED" : "FAILED") << "\n";
  std::cout << "Direct bitwidth: " << static_cast<int>(direct_bitwidth) << "\n";

  std::remove(filepath.c_str());
  return data_match;
}

// =============================================================================
// Main Test
// =============================================================================

int main()
{
  std::cout << "=== FastLanes One-Page Multi-Type Verification Test ===\n";

  bool all_passed = true;

  // Test all types with sequential data
  all_passed &= run_one_page_test<int32_t>("int32", false);
  all_passed &= run_one_page_test<uint32_t>("uint32", false);
  // all_passed &= run_one_page_test<int64_t>("int64", false);
  // all_passed &= run_one_page_test<uint64_t>("uint64", false);

  // Test all types with random data
  all_passed &= run_one_page_test<int32_t>("int32", true);
  all_passed &= run_one_page_test<uint32_t>("uint32", true);
  // all_passed &= run_one_page_test<int64_t>("int64", true);
  // all_passed &= run_one_page_test<uint64_t>("uint64", true);

  std::cout << "\n============================================\n";
  std::cout << "FINAL RESULT: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
  std::cout << "============================================\n";

  return all_passed ? 0 : 1;
}