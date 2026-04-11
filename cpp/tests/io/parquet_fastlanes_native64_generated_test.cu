/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>
#include <cudf/fastlanes/fls_gen/unpack/unpack.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class ParquetFastLanesNative64GeneratedTest : public cudf::test::BaseFixture {};

constexpr uint32_t kVectorSize     = static_cast<uint32_t>(fastlanes::VECTOR_SIZE);
constexpr uint32_t kLanesPerVector = 16;
constexpr uint32_t kValuesPerLane  = 64;
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime       = 1099511628211ULL;

enum class data_pattern : uint8_t { randomized, adversarial, pathological };

[[nodiscard]] std::string pattern_name(data_pattern pattern)
{
  switch (pattern) {
    case data_pattern::randomized: return "randomized";
    case data_pattern::adversarial: return "adversarial";
    case data_pattern::pathological: return "pathological";
  }
  return "unknown";
}

[[nodiscard]] uint64_t splitmix64(uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

[[nodiscard]] uint64_t bitwidth_mask(uint8_t bw)
{
  if (bw == 0) { return 0ULL; }
  if (bw >= 64) { return std::numeric_limits<uint64_t>::max(); }
  return (uint64_t{1} << bw) - 1ULL;
}

[[nodiscard]] uint64_t checksum_words(std::vector<uint64_t> const& words)
{
  if (words.empty()) { return kFnvOffsetBasis; }

  uint64_t hash = kFnvOffsetBasis;
  auto const* p = reinterpret_cast<uint8_t const*>(words.data());
  for (size_t i = 0; i < words.size() * sizeof(uint64_t); ++i) {
    hash ^= p[i];
    hash *= kFnvPrime;
  }
  return hash;
}

[[nodiscard]] uint64_t make_delta(uint64_t index, uint64_t seed, uint8_t bw, data_pattern pattern)
{
  auto const mask = bitwidth_mask(bw);
  if (bw == 0) { return 0ULL; }

  switch (pattern) {
    case data_pattern::randomized:
      return splitmix64(seed + index * 0x9e3779b97f4a7c15ULL) & mask;
    case data_pattern::adversarial: {
      switch (index % 8ULL) {
        case 0: return 0ULL;
        case 1: return mask;
        case 2: return 1ULL;
        case 3: return mask - 1ULL;
        case 4: return uint64_t{1} << (index % bw);
        case 5: return mask ^ (uint64_t{1} << (index % bw));
        case 6: return ((index / 8ULL) % 2ULL == 0ULL) ? 0ULL : mask;
        default: return splitmix64(seed ^ (index * 0x517cc1b727220a95ULL)) & mask;
      }
    }
    case data_pattern::pathological: {
      switch (index % 6ULL) {
        case 0: return (uint64_t{1} << (index % bw)) - 1ULL;
        case 1: return uint64_t{1} << (index % bw);
        case 2: return (0x1555555555555555ULL >> (index % 17ULL)) & mask;
        case 3: return (0x2aaaaaaaaaaaaaaaULL >> (index % 13ULL)) & mask;
        case 4: return (mask - (index & 0x3ffULL)) & mask;
        default: return splitmix64(seed + (index << 1U)) & mask;
      }
    }
  }

  return 0ULL;
}

[[nodiscard]] std::vector<uint64_t> make_padded_deltas(uint32_t valid_count,
                                                       uint64_t seed,
                                                       uint8_t bw,
                                                       data_pattern pattern)
{
  auto const padded = static_cast<uint32_t>(fastlanes::padded_count(valid_count));
  std::vector<uint64_t> deltas(padded, 0ULL);
  for (uint32_t i = 0; i < valid_count; ++i) {
    deltas[i] = make_delta(i, seed, bw, pattern);
  }
  return deltas;
}

[[nodiscard]] std::vector<uint64_t> add_base_to_deltas(std::vector<uint64_t> const& deltas,
                                                        uint64_t base_bits)
{
  std::vector<uint64_t> values(deltas.size(), base_bits);
  for (size_t i = 0; i < deltas.size(); ++i) {
    values[i] = base_bits + deltas[i];
  }
  return values;
}

[[nodiscard]] std::vector<uint64_t> pack_bw_vectors_cpu(std::vector<uint64_t> const& padded_deltas,
                                                        uint8_t bw)
{
  if (padded_deltas.size() % kVectorSize != 0) {
    throw std::invalid_argument("pack_bw_vectors_cpu expects padded 1024-multiple input");
  }

  auto const words_per_vector =
    static_cast<size_t>(fastlanes::encoded_size_bytes(kVectorSize, bw) / sizeof(uint64_t));
  auto const num_vectors = padded_deltas.size() / kVectorSize;

  if (words_per_vector == 0) { return {}; }

  std::vector<uint64_t> packed(num_vectors * words_per_vector, 0ULL);
  for (size_t v = 0; v < num_vectors; ++v) {
    generated::pack::fallback::scalar::pack(padded_deltas.data() + v * kVectorSize,
                                            packed.data() + v * words_per_vector,
                                            bw);
  }

  return packed;
}

[[nodiscard]] std::vector<uint64_t> unpack_bw_vectors_cpu(std::vector<uint64_t> const& packed,
                                                          size_t padded_count,
                                                          uint8_t bw)
{
  if (padded_count % kVectorSize != 0) {
    throw std::invalid_argument("unpack_bw_vectors_cpu expects padded_count multiple of 1024");
  }

  auto const words_per_vector =
    static_cast<size_t>(fastlanes::encoded_size_bytes(kVectorSize, bw) / sizeof(uint64_t));

  if (words_per_vector == 0) {
    return std::vector<uint64_t>(padded_count, 0ULL);
  }

  auto const num_vectors = padded_count / kVectorSize;
  if (packed.size() != num_vectors * words_per_vector) {
    throw std::invalid_argument("unpack_bw_vectors_cpu packed size mismatch");
  }

  std::vector<uint64_t> unpacked(padded_count, 0ULL);
  for (size_t v = 0; v < num_vectors; ++v) {
    generated::unpack::fallback::scalar::unpack(packed.data() + v * words_per_vector,
                                                unpacked.data() + v * kVectorSize,
                                                bw);
  }

  return unpacked;
}

[[nodiscard]] std::string cuda_error(cudaError_t status, char const* operation)
{
  std::ostringstream oss;
  oss << operation << " failed: " << cudaGetErrorString(status);
  return oss.str();
}

#include <cudf/fastlanes/native64_cuda_kernels.inl>

struct parity_evidence {
  size_t cpu_roundtrip_mismatch{};
  size_t gpu_pack_mismatch{};
  size_t decode_from_cpu_mismatch_u64{};
  size_t decode_from_cpu_mismatch_i64{};
  size_t roundtrip_mismatch_u64{};
  size_t roundtrip_mismatch_i64{};
  uint64_t cpu_packed_checksum{};
  uint64_t gpu_packed_checksum{};
};

[[nodiscard]] parity_evidence run_bw_parity_case(uint8_t bw,
                                                 uint32_t total_count,
                                                 uint64_t base_bits,
                                                 uint64_t seed,
                                                 data_pattern pattern)
{
  auto const deltas = make_padded_deltas(total_count, seed, bw, pattern);
  auto const values = add_base_to_deltas(deltas, base_bits);

  auto const cpu_packed = pack_bw_vectors_cpu(deltas, bw);
  auto const gpu_packed = native64_generated::encode_by_bw_gpu(bw, values, base_bits, total_count);

  auto const cpu_unpacked      = unpack_bw_vectors_cpu(cpu_packed, deltas.size(), bw);
  auto const gpu_decode_cpu_pk =
    native64_generated::decode_by_bw_gpu(bw, cpu_packed, base_bits, total_count);
  auto const gpu_roundtrip =
    native64_generated::decode_by_bw_gpu(bw, gpu_packed, base_bits, total_count);

  auto evidence = parity_evidence{.cpu_packed_checksum = checksum_words(cpu_packed),
                                  .gpu_packed_checksum = checksum_words(gpu_packed)};

  for (size_t i = 0; i < cpu_packed.size(); ++i) {
    if (cpu_packed[i] != gpu_packed[i]) { ++evidence.gpu_pack_mismatch; }
  }

  for (uint32_t i = 0; i < total_count; ++i) {
    if (cpu_unpacked[i] != deltas[i]) { ++evidence.cpu_roundtrip_mismatch; }

    auto const cpu_bits          = base_bits + cpu_unpacked[i];
    auto const gpu_bits_from_cpu = gpu_decode_cpu_pk[i];
    auto const gpu_bits_roundtrp = gpu_roundtrip[i];

    if (cpu_bits != gpu_bits_from_cpu) { ++evidence.decode_from_cpu_mismatch_u64; }
    if (cpu_bits != gpu_bits_roundtrp) { ++evidence.roundtrip_mismatch_u64; }

    if (fastlanes::u64_bits_to_int64(cpu_bits) != fastlanes::u64_bits_to_int64(gpu_bits_from_cpu)) {
      ++evidence.decode_from_cpu_mismatch_i64;
    }

    if (fastlanes::u64_bits_to_int64(cpu_bits) != fastlanes::u64_bits_to_int64(gpu_bits_roundtrp)) {
      ++evidence.roundtrip_mismatch_i64;
    }
  }

  return evidence;
}

[[nodiscard]] std::array<uint64_t, 3> base_set_for_bw(uint8_t bw)
{
  auto const mask = bitwidth_mask(bw);

  auto const signed_hi_delta =
    (bw >= 64) ? int64_t{0} : static_cast<int64_t>(std::min<uint64_t>(mask, uint64_t{1} << 62));

  return {
    0ULL,
    fastlanes::int64_to_u64_bits(std::numeric_limits<int64_t>::min() + 1024),
    fastlanes::int64_to_u64_bits(std::numeric_limits<int64_t>::max() - signed_hi_delta),
  };
}

void assert_zero_mismatch(parity_evidence const& evidence)
{
  EXPECT_EQ(evidence.cpu_roundtrip_mismatch, 0U);
  EXPECT_EQ(evidence.gpu_pack_mismatch, 0U);
  EXPECT_EQ(evidence.decode_from_cpu_mismatch_u64, 0U);
  EXPECT_EQ(evidence.decode_from_cpu_mismatch_i64, 0U);
  EXPECT_EQ(evidence.roundtrip_mismatch_u64, 0U);
  EXPECT_EQ(evidence.roundtrip_mismatch_i64, 0U);
}

}  // namespace

TEST_F(ParquetFastLanesNative64GeneratedTest, AnchorMatrixParity)
{
  std::array<uint8_t, 8> constexpr anchors = {0, 1, 31, 32, 33, 37, 63, 64};
  std::array<uint64_t, 2> constexpr seeds = {0x00000000baddf00dULL, 0x0000000012345678ULL};
  std::array<data_pattern, 2> constexpr patterns = {
    data_pattern::randomized, data_pattern::adversarial};

  std::ostringstream report;
  report << "FL64-R4-M3 anchor:";

  for (auto const bw : anchors) {
    auto const bases = base_set_for_bw(bw);

    for (auto const seed : seeds) {
      for (auto const pattern : patterns) {
        for (auto const base : bases) {
          auto const evidence = run_bw_parity_case(bw, 2050, base, seed, pattern);
          assert_zero_mismatch(evidence);

          report << " [bw=" << static_cast<int>(bw) << ",seed=0x" << std::hex << seed << std::dec
                 << ",pattern=" << pattern_name(pattern) << ",base=0x" << std::hex << base
                 << std::dec << ",cpu_chk=0x" << std::hex << evidence.cpu_packed_checksum
                 << ",gpu_chk=0x" << evidence.gpu_packed_checksum << std::dec << "]";
        }
      }
    }
  }

  // std::cout << report.str() << std::endl;
}

TEST_F(ParquetFastLanesNative64GeneratedTest, FullSweepParityMatrix)
{
  std::array<uint32_t, 6> constexpr counts = {1023, 1024, 1025, 2047, 2048, 2049};

  std::ostringstream report;
  report << "FL64-R4-M3 sweep:";

  for (uint8_t bw = 0; bw <= 64; ++bw) {
    auto const bases = base_set_for_bw(bw);

    for (auto const count : counts) {
      for (auto const base : bases) {
        auto const seed = 0x5a50e4c34bd1c7f9ULL + count + static_cast<uint64_t>(bw) * 97ULL;
        auto const evidence = run_bw_parity_case(bw, count, base, seed, data_pattern::pathological);
        assert_zero_mismatch(evidence);

        report << " [bw=" << static_cast<int>(bw) << ",n=" << count << ",base=0x" << std::hex
               << base << std::dec << ",cpu_chk=0x" << std::hex << evidence.cpu_packed_checksum
               << ",gpu_chk=0x" << evidence.gpu_packed_checksum << std::dec << "]";
      }
    }

    if (bw == 64) { break; }
  }

  // std::cout << report.str() << std::endl;
}

TEST_F(ParquetFastLanesNative64GeneratedTest, StabilityRepeatsSelectedSeeds)
{
  std::array<uint8_t, 4> constexpr selected_bw = {1, 33, 37, 64};
  std::array<uint64_t, 3> constexpr seeds = {
    0x0000000000000011ULL, 0x000000000000001dULL, 0x000000000000002fULL};

  constexpr uint32_t repeats = 3;
  std::ostringstream report;
  report << "FL64-R4-M3 stability:";

  for (auto const bw : selected_bw) {
    auto const bases = base_set_for_bw(bw);
    auto const base  = bases[1];

    for (uint32_t rep = 0; rep < repeats; ++rep) {
      for (auto const seed : seeds) {
        auto const evidence = run_bw_parity_case(bw, 2049, base, seed, data_pattern::randomized);
        assert_zero_mismatch(evidence);

        report << " [bw=" << static_cast<int>(bw) << ",run=" << rep << ",seed=0x" << std::hex
               << seed << std::dec << ",cpu_chk=0x" << std::hex << evidence.cpu_packed_checksum
               << ",gpu_chk=0x" << evidence.gpu_packed_checksum << std::dec << "]";
      }
    }
  }

  // std::cout << report.str() << std::endl;
}
