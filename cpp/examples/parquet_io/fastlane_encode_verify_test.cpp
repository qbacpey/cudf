/**
 * @file fastlane_encode_verify_test.cpp
 * @brief Sanity test that verifies cudf FastLanes encoding matches direct pack() calls.
 *
 * This test:
 * 1. Creates test data and pads it to 1024 boundary (mimicking cudf)
 * 2. Directly calls FastLanes pack() function
 * 3. Dumps the encoded bits
 * 4. Can be compared against cudf's encoder output
 */

#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/debug.hpp>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// =============================================================================
// Bitwidth Computation (Same logic as cudf encoder)
// =============================================================================

template <typename T>
uint8_t compute_bitwidth(const T* data, size_t count)
{
  using UnsignedT = std::make_unsigned_t<T>;

  UnsignedT max_val = 0;
  for (size_t i = 0; i < count; ++i) {
    // For signed types, check if negative (would need full bitwidth)
    if constexpr (std::is_signed_v<T>) {
      if (data[i] < 0) {
        // Negative value detected - need full bitwidth
        return sizeof(T) * 8;
      }
    }
    UnsignedT val = static_cast<UnsignedT>(data[i]);
    if (val > max_val) max_val = val;
  }

  if (max_val == 0) return 1;  // Minimum 1 bit

  uint8_t bits = 0;
  while (max_val > 0) {
    max_val >>= 1;
    bits++;
  }
  return bits;
}

// =============================================================================
// Direct FastLanes Pack Test
// =============================================================================

template <typename T>
void test_fastlanes_pack_direct(const std::vector<T>& original_data)
{
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr size_t VECTOR_SIZE = ::fastlanes::VECTOR_SIZE;  // 1024

  std::cout << "============================================\n";
  std::cout << "Direct FastLanes Pack Test\n";
  std::cout << "Type: " << (std::is_signed_v<T> ? "int" : "uint") << (sizeof(T) * 8) << "_t\n";
  std::cout << "Original count: " << original_data.size() << "\n";
  std::cout << "============================================\n\n";

  // 1. Compute padded count (same as cudf)
  size_t original_count = original_data.size();
  size_t num_vectors    = ::fastlanes::num_vectors(original_count);
  size_t padded_count   = num_vectors * VECTOR_SIZE;

  std::cout << "Padding: " << original_count << " -> " << padded_count << " (+"
            << (padded_count - original_count) << " zeros)\n";
  std::cout << "Number of vectors: " << num_vectors << "\n\n";

  // 2. Create padded input buffer (mimicking cudf)
  std::vector<T> padded_input(padded_count, T{0});
  std::copy(original_data.begin(), original_data.end(), padded_input.begin());

  // 3. Compute bitwidth
  uint8_t bitwidth = compute_bitwidth(original_data.data(), original_count);
  std::cout << "Computed bitwidth: " << static_cast<int>(bitwidth) << "\n\n";

  // 4. Print input data (first 20 values)
  std::cout << "Input data (first 20, padded):\n";
  for (size_t i = 0; i < std::min<size_t>(20, padded_count); ++i) {
    std::cout << padded_input[i] << " ";
  }
  std::cout << "\n\n";

  // 5. Allocate output buffer
  // Output size = num_vectors * VECTOR_SIZE * bitwidth / 8 bytes
  // In T-sized elements = num_vectors * VECTOR_SIZE * bitwidth / (sizeof(T) * 8)
  size_t encoded_bytes    = ::fastlanes::encoded_size_bytes(padded_count, bitwidth);
  size_t output_elements  = encoded_bytes / sizeof(UnsignedT);
  std::vector<UnsignedT> encoded_output(output_elements);

  std::cout << "Encoded output size: " << encoded_bytes << " bytes (" << output_elements
            << " elements)\n\n";

  // 6. Call FastLanes pack for each vector
  const UnsignedT* in_ptr = reinterpret_cast<const UnsignedT*>(padded_input.data());
  UnsignedT* out_ptr      = encoded_output.data();

  // Output elements per vector = (VECTOR_SIZE * bitwidth) / (sizeof(T) * 8)
  size_t out_elements_per_vector = (VECTOR_SIZE * bitwidth) / (sizeof(T) * 8);

  std::cout << "Output elements per vector: " << out_elements_per_vector << "\n\n";

  for (size_t v = 0; v < num_vectors; ++v) {
    std::cout << "Packing vector " << v << "...\n";
    generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bitwidth);
    in_ptr += VECTOR_SIZE;
    out_ptr += out_elements_per_vector;
  }

  // 7. Dump encoded output (same format as cudf)
  fastlanes::debug::print_encoded_dump(encoded_output, "Direct Pack Encoded Stream");

  // 8. Show reinterpret_cast equivalence for positive values
  std::cout << "============================================\n";
  std::cout << "Reinterpret Cast Verification\n";
  std::cout << "============================================\n";
  for (size_t i = 0; i < std::min<size_t>(5, original_data.size()); ++i) {
    T signed_val             = original_data[i];
    UnsignedT as_unsigned    = *reinterpret_cast<const UnsignedT*>(&signed_val);
    UnsignedT static_casted  = static_cast<UnsignedT>(signed_val);
    bool bits_match          = (as_unsigned == static_casted);

    std::cout << "Value[" << i << "]: " << signed_val << "\n";
    std::cout << "  reinterpret_cast: 0x" << std::hex << as_unsigned << std::dec << "\n";
    std::cout << "  static_cast:      0x" << std::hex << static_casted << std::dec << "\n";
    std::cout << "  Bits match: " << (bits_match ? "YES" : "NO (negative value)") << "\n\n";
  }
}

// =============================================================================
// Main
// =============================================================================

int main()
{
  std::cout << "=== FastLanes Encoding Verification Test ===\n\n";

  // Test 1: Small int32_t data (fits in one vector with padding)
  {
    std::cout << "\n########## TEST 1: Small int32_t data ##########\n\n";
    std::vector<int32_t> data(128);
    std::iota(data.begin(), data.end(), 100);  // 100, 101, 102, ..., 227
    test_fastlanes_pack_direct(data);
  }

  // Test 2: uint32_t data
  {
    std::cout << "\n########## TEST 2: uint32_t data ##########\n\n";
    std::vector<uint32_t> data(128);
    std::iota(data.begin(), data.end(), 1000);  // 1000, 1001, ..., 1127
    test_fastlanes_pack_direct(data);
  }

  // Test 3: Exactly 1024 values (one full vector, no padding)
  {
    std::cout << "\n########## TEST 3: Exactly 1024 int32_t values ##########\n\n";
    std::vector<int32_t> data(1024);
    std::iota(data.begin(), data.end(), 0);  // 0, 1, 2, ..., 1023
    test_fastlanes_pack_direct(data);
  }

  // Test 4: int32_t with negative values (forces 32-bit encoding)
  {
    std::cout << "\n########## TEST 4: int32_t with negatives ##########\n\n";
    std::vector<int32_t> data = {-5, -3, -1, 0, 1, 3, 5, 100, 200, 300};
    test_fastlanes_pack_direct(data);
  }

  // Test 5: int32_t for comparison
  {
    std::cout << "\n########## TEST 5: int32_t (for comparison) ##########\n\n";
    std::vector<int32_t> data(128);
    std::iota(data.begin(), data.end(), 100);
    test_fastlanes_pack_direct(data);
  }

  std::cout << "\n=== All Tests Complete ===\n";
  return 0;
}
