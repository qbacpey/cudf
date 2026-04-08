/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/debug.hpp>
#include <cudf/fastlanes/fls_gen/ffor/ffor.hpp>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>
#include <cudf/fastlanes/fls_gen/unffor/unffor.hpp>
#include <cudf/fastlanes/fls_gen/unpack/unpack.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t vector_size            = static_cast<size_t>(fastlanes::VECTOR_SIZE);
constexpr size_t max_preview_values     = 16;
constexpr size_t max_preview_bytes      = 96;
constexpr size_t max_preview_word_count = 24;

struct int64_demo_case {
  std::string name;
  std::string description;
  std::vector<int64_t> values;
};

uint8_t bit_width_u64(uint64_t value)
{
  uint8_t bits = 0;
  while (value != 0) {
    value >>= 1;
    ++bits;
  }
  return bits;
}

uint8_t compute_ffor_bitwidth(std::vector<int64_t> const& values, int64_t base, uint64_t* max_delta)
{
  uint64_t max_delta_local = 0;
  uint64_t const base_bits = fastlanes::int64_to_u64_bits(base);

  for (auto const value : values) {
    uint64_t const value_bits = fastlanes::int64_to_u64_bits(value);
    uint64_t const delta      = value_bits - base_bits;
    max_delta_local           = std::max(max_delta_local, delta);
  }

  if (max_delta != nullptr) { *max_delta = max_delta_local; }
  return bit_width_u64(max_delta_local);
}

void print_value_preview(std::vector<int64_t> const& values, std::string const& label)
{
  auto const show_count = std::min(max_preview_values, values.size());
  std::cout << label << " (first " << show_count << "):\n";
  for (size_t i = 0; i < show_count; ++i) {
    auto const bits = fastlanes::int64_to_u64_bits(values[i]);
    std::cout << "  [" << std::setw(2) << i << "] " << std::setw(20) << values[i] << "  (0x"
              << std::hex << std::setw(16) << std::setfill('0') << bits << std::dec
              << std::setfill(' ') << ")\n";
  }
  std::cout << "\n";
}

void print_delta_preview(std::vector<int64_t> const& values, int64_t base)
{
  auto const show_count = std::min(max_preview_values, values.size());
  uint64_t const base_bits = fastlanes::int64_to_u64_bits(base);

  std::cout << "FFOR deltas from base (first " << show_count << "):\n";
  for (size_t i = 0; i < show_count; ++i) {
    uint64_t const value_bits = fastlanes::int64_to_u64_bits(values[i]);
    uint64_t const delta      = value_bits - base_bits;
    std::cout << "  [" << std::setw(2) << i << "] " << std::setw(20) << delta << "  (0x"
              << std::hex << std::setw(16) << std::setfill('0') << delta << std::dec
              << std::setfill(' ') << ")\n";
  }
  std::cout << "\n";
}

void print_byte_preview(void const* data, size_t total_bytes, std::string const& label)
{
  std::cout << label << " (" << total_bytes << " bytes total)\n";
  if (total_bytes == 0) {
    std::cout << "  <empty>\n\n";
    return;
  }

  auto const* bytes      = static_cast<uint8_t const*>(data);
  auto const bytes_to_show = std::min(max_preview_bytes, total_bytes);

  std::cout << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes_to_show; ++i) {
    if ((i % 16) == 0) {
      std::cout << "  [" << std::setw(4) << i << "] ";
    }
    std::cout << std::setw(2) << static_cast<uint32_t>(bytes[i]) << ' ';
    if ((i % 16) == 15 || (i + 1) == bytes_to_show) { std::cout << '\n'; }
  }
  std::cout << std::dec << std::setfill(' ');

  if (bytes_to_show < total_bytes) {
    std::cout << "  ... (truncated, showing first " << bytes_to_show << " bytes)\n";
  }
  std::cout << "\n";
}

std::vector<int64_demo_case> build_demo_cases()
{
  std::vector<int64_demo_case> cases;

  {
    std::vector<int64_t> values(vector_size, -7);
    cases.push_back({"constant",
                     "All values are equal. Deltas are zero, so bitwidth should be 0.",
                     std::move(values)});
  }

  {
    int64_t const start = 4'000'000'000'000'000'000LL;
    std::vector<int64_t> values(vector_size, 0);
    for (size_t i = 0; i < vector_size; ++i) {
      values[i] = start + static_cast<int64_t>(i);
    }
    cases.push_back({"large_base_monotonic",
                     "Very large absolute values but tiny delta range (0..1023). This should still compress well.",
                     std::move(values)});
  }

  {
    int64_t const start = 1'650'000'000'000'000'000LL;
    std::vector<int64_t> values(vector_size, 0);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> small_noise(0, 63);
    for (size_t i = 0; i < vector_size; ++i) {
      values[i] = start + small_noise(rng);
    }
    cases.push_back({"small_noise",
                     "Values stay in a narrow local band. Small bitwidth demonstrates strong compression.",
                     std::move(values)});
  }

  {
    int64_t const start = 2'000'000'000'000LL;
    std::vector<int64_t> values(vector_size, 0);
    for (size_t i = 0; i < vector_size; ++i) {
      values[i] = start + static_cast<int64_t>(i % 31);
    }
    values[777] = start + (int64_t{1} << 40);
    cases.push_back({"single_outlier",
                     "One outlier forces a much larger bitwidth, showing sensitivity to extreme values.",
                     std::move(values)});
  }

  {
    std::vector<int64_t> values(vector_size, std::numeric_limits<int64_t>::min());
    values[1] = std::numeric_limits<int64_t>::max();
    for (size_t i = 2; i < vector_size; ++i) {
      values[i] = std::numeric_limits<int64_t>::min() + static_cast<int64_t>(i);
    }
    cases.push_back({"full_span_extremes",
                     "Covers almost full int64 range. Bitwidth reaches 64, so compression benefit disappears.",
                     std::move(values)});
  }

  return cases;
}

bool run_case(int64_demo_case const& demo_case, size_t case_index, size_t case_count)
{
  if (demo_case.values.size() != vector_size) {
    std::cerr << "Case '" << demo_case.name << "' has invalid input size\n";
    return false;
  }

  int64_t const base = *std::min_element(demo_case.values.begin(), demo_case.values.end());

  uint64_t max_delta = 0;
  uint8_t const bitwidth = compute_ffor_bitwidth(demo_case.values, base, &max_delta);

  size_t const raw_bytes     = demo_case.values.size() * sizeof(int64_t);
  size_t const encoded_bytes = fastlanes::encoded_size_bytes(demo_case.values.size(), bitwidth);
  size_t const encoded_words = encoded_bytes / sizeof(int64_t);

  std::vector<int64_t> encoded(encoded_words == 0 ? 1 : encoded_words, 0);
  std::vector<int64_t> decoded(vector_size, 0);

  fastlanes::generated::ffor::fallback::scalar::ffor(
    demo_case.values.data(), encoded.data(), bitwidth, &base);
  fastlanes::generated::unffor::fallback::scalar::unffor(
    encoded.data(), decoded.data(), bitwidth, &base);

  bool const pass = (decoded == demo_case.values);

  std::cout << "\n============================================================\n";
  std::cout << "Case " << case_index << "/" << case_count << ": " << demo_case.name << "\n";
  std::cout << "============================================================\n";
  std::cout << demo_case.description << "\n\n";

  std::cout << "[FFOR64] Input count        : " << demo_case.values.size() << "\n";
  std::cout << "[FFOR64] Base (min value)   : " << base << "\n";
  std::cout << "[FFOR64] Max delta          : " << max_delta << " (0x" << std::hex << max_delta
            << std::dec << ")\n";
  std::cout << "[FFOR64] Computed bitwidth  : " << static_cast<int>(bitwidth) << "\n";
  std::cout << "[FFOR64] Raw size           : " << raw_bytes << " bytes\n";
  std::cout << "[FFOR64] Encoded size       : " << encoded_bytes << " bytes\n";
  if (raw_bytes > 0 && encoded_bytes > 0) {
    double const ratio = static_cast<double>(raw_bytes) / static_cast<double>(encoded_bytes);
    double const saved = 100.0 * (1.0 - static_cast<double>(encoded_bytes) / static_cast<double>(raw_bytes));
    std::cout << "[FFOR64] Compression ratio  : " << std::fixed << std::setprecision(2) << ratio
              << "x (" << saved << "% smaller)\n";
  } else if (encoded_bytes == 0) {
    std::cout << "[FFOR64] Compression ratio  : perfect payload elimination (bw=0)\n";
  }
  std::cout << "\n";

  print_value_preview(demo_case.values, "Input values");
  print_delta_preview(demo_case.values, base);

  print_byte_preview(demo_case.values.data(), raw_bytes, "Raw input bytes before encoding");
  print_byte_preview(encoded.data(), encoded_bytes, "Encoded payload bytes after FFOR encoding");

  auto const encoded_preview_words = std::min(max_preview_word_count, encoded_words);
  if (encoded_preview_words > 0) {
    fastlanes::debug::print_encoded_dump(
      encoded.data(), encoded_preview_words, "Encoded payload words (preview)");
    if (encoded_preview_words < encoded_words) {
      std::cout << "(truncated, showing first " << encoded_preview_words << " of "
                << encoded_words << " words)\n\n";
    }
  }

  print_byte_preview(decoded.data(), raw_bytes, "Decoded bytes after UNFFOR decoding");
  print_value_preview(decoded, "Decoded values");

  std::cout << "Roundtrip check: " << (pass ? "PASS" : "FAIL") << "\n";

  if (!pass) {
    for (size_t i = 0; i < demo_case.values.size(); ++i) {
      if (demo_case.values[i] != decoded[i]) {
        std::cerr << "First mismatch at index " << i << ": in=" << demo_case.values[i]
                  << ", out=" << decoded[i] << "\n";
        break;
      }
    }
  }

  return pass;
}

}  // namespace

int main()
{
  std::cout << "=== FastLanes INT64 CPU FFOR Demonstration ===\n\n";
  std::cout << "This demo uses one 1024-value int64 vector per case.\n";
  std::cout << "For one vector, encoded payload bytes = 128 * bitwidth.\n";
  std::cout << "It prints values + raw bytes before encoding and payload bytes after encoding.\n\n";

  auto const cases = build_demo_cases();

  bool all_passed = true;
  for (size_t i = 0; i < cases.size(); ++i) {
    all_passed = run_case(cases[i], i + 1, cases.size()) && all_passed;
  }

  std::cout << "\n============================================================\n";
  std::cout << "Final result: " << (all_passed ? "ALL CASES PASSED" : "SOME CASES FAILED") << "\n";
  std::cout << "============================================================\n";

  return all_passed ? 0 : 1;
}
