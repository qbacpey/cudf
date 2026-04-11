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

class ParquetFastLanesNative64Bw37Test : public cudf::test::BaseFixture {};

constexpr uint32_t kVectorSize      = static_cast<uint32_t>(fastlanes::VECTOR_SIZE);
constexpr uint32_t kLanesPerVector  = 16;
constexpr uint32_t kValuesPerLane   = 64;
constexpr uint8_t kBw37             = 37;
constexpr uint64_t kFnvOffsetBasis  = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime        = 1099511628211ULL;
constexpr uint64_t kBw37Mask        = (uint64_t{1} << kBw37) - 1ULL;
constexpr uint32_t kWordsPerVector37 =
  static_cast<uint32_t>(fastlanes::encoded_size_bytes(kVectorSize, kBw37) / sizeof(uint64_t));
constexpr uint32_t kWordsPerLane37 = kWordsPerVector37 / kLanesPerVector;

static_assert(kWordsPerVector37 == 592, "bw37 vector word count contract changed unexpectedly");
static_assert(kWordsPerLane37 == 37, "bw37 lane word count contract changed unexpectedly");

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

[[nodiscard]] uint64_t make_bw37_delta(uint64_t index, uint64_t seed, data_pattern pattern)
{
  switch (pattern) {
    case data_pattern::randomized:
      return splitmix64(seed + index * 0x9e3779b97f4a7c15ULL) & kBw37Mask;
    case data_pattern::adversarial: {
      switch (index % 8ULL) {
        case 0: return 0ULL;
        case 1: return kBw37Mask;
        case 2: return 1ULL;
        case 3: return kBw37Mask - 1ULL;
        case 4: return uint64_t{1} << (index % kBw37);
        case 5: return kBw37Mask ^ (uint64_t{1} << (index % kBw37));
        case 6: return ((index / 8ULL) % 2ULL == 0ULL) ? 0ULL : kBw37Mask;
        default: return splitmix64(seed ^ (index * 0x517cc1b727220a95ULL)) & kBw37Mask;
      }
    }
    case data_pattern::pathological: {
      switch (index % 6ULL) {
        case 0: return (uint64_t{1} << (index % kBw37)) - 1ULL;
        case 1: return uint64_t{1} << (index % kBw37);
        case 2: return (0x155555555ULL >> (index % 7ULL)) & kBw37Mask;
        case 3: return (0x2aaaaaaaaULL >> (index % 5ULL)) & kBw37Mask;
        case 4: return (kBw37Mask - (index & 0x3ffULL)) & kBw37Mask;
        default: return splitmix64(seed + (index << 1U)) & kBw37Mask;
      }
    }
  }
  return 0ULL;
}

[[nodiscard]] std::vector<uint64_t> make_padded_bw37_deltas(uint32_t valid_count,
                                                             uint64_t seed,
                                                             data_pattern pattern)
{
  auto const padded = static_cast<uint32_t>(fastlanes::padded_count(valid_count));
  std::vector<uint64_t> deltas(padded, 0ULL);
  for (uint32_t i = 0; i < valid_count; ++i) {
    deltas[i] = make_bw37_delta(i, seed, pattern);
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

[[nodiscard]] std::vector<uint64_t> pack_one_vector_cpu(std::vector<uint64_t> const& values,
                                                         uint8_t bw)
{
  if (values.size() != kVectorSize) {
    throw std::invalid_argument("pack_one_vector_cpu expects exactly 1024 values");
  }

  auto const words = fastlanes::encoded_size_bytes(kVectorSize, bw) / sizeof(uint64_t);
  std::vector<uint64_t> packed(std::max<size_t>(words, 1), 0ULL);
  generated::pack::fallback::scalar::pack(values.data(), packed.data(), bw);
  packed.resize(words);
  return packed;
}

[[nodiscard]] std::vector<uint64_t> unpack_one_vector_cpu(std::vector<uint64_t> const& packed,
                                                           uint8_t bw)
{
  std::vector<uint64_t> unpacked(kVectorSize, 0ULL);
  uint64_t const dummy = 0ULL;
  auto const* in_ptr   = packed.empty() ? &dummy : packed.data();
  generated::unpack::fallback::scalar::unpack(in_ptr, unpacked.data(), bw);
  return unpacked;
}

[[nodiscard]] std::vector<uint64_t> pack_bw37_vectors_cpu(std::vector<uint64_t> const& padded_deltas)
{
  if (padded_deltas.size() % kVectorSize != 0) {
    throw std::invalid_argument("pack_bw37_vectors_cpu expects padded 1024-multiple input");
  }

  auto const num_vectors = padded_deltas.size() / kVectorSize;
  std::vector<uint64_t> packed(num_vectors * kWordsPerVector37, 0ULL);

  for (size_t v = 0; v < num_vectors; ++v) {
    generated::pack::fallback::scalar::pack(padded_deltas.data() + v * kVectorSize,
                                            packed.data() + v * kWordsPerVector37,
                                            kBw37);
  }
  return packed;
}

[[nodiscard]] std::vector<uint64_t> unpack_bw37_vectors_cpu(std::vector<uint64_t> const& packed,
                                                             size_t padded_count)
{
  if (padded_count % kVectorSize != 0) {
    throw std::invalid_argument("unpack_bw37_vectors_cpu expects padded_count multiple of 1024");
  }

  auto const num_vectors = padded_count / kVectorSize;
  if (packed.size() != num_vectors * kWordsPerVector37) {
    throw std::invalid_argument("unpack_bw37_vectors_cpu packed size mismatch");
  }

  std::vector<uint64_t> unpacked(padded_count, 0ULL);
  for (size_t v = 0; v < num_vectors; ++v) {
    generated::unpack::fallback::scalar::unpack(packed.data() + v * kWordsPerVector37,
                                                unpacked.data() + v * kVectorSize,
                                                kBw37);
  }
  return unpacked;
}

[[nodiscard]] std::string cuda_error(cudaError_t status, char const* operation)
{
  std::ostringstream oss;
  oss << operation << " failed: " << cudaGetErrorString(status);
  return oss.str();
}

#include <cudf/fastlanes/native64_bw37_cuda_kernels.inl>

[[nodiscard]] std::vector<uint64_t> encode_bw37_on_gpu(std::vector<uint64_t> const& values,
                                                        uint64_t base_bits,
                                                        uint32_t total_count)
{
  auto const stream       = cudf::get_default_stream();
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));

  if (values.size() != padded_count) {
    throw std::invalid_argument("encode_bw37_on_gpu values size mismatch");
  }

  std::vector<uint64_t> packed(static_cast<size_t>(num_vectors) * kWordsPerVector37, 0ULL);

  uint64_t* d_values = nullptr;
  uint64_t* d_packed = nullptr;

  auto const values_bytes = values.size() * sizeof(uint64_t);
  auto const packed_bytes = packed.size() * sizeof(uint64_t);

  if (values_bytes > 0) {
    auto const malloc_values_status = cudaMalloc(reinterpret_cast<void**>(&d_values), values_bytes);
    if (malloc_values_status != cudaSuccess) {
      throw std::runtime_error(cuda_error(malloc_values_status, "cudaMalloc(d_values)"));
    }
  }

  if (packed_bytes > 0) {
    auto const malloc_packed_status = cudaMalloc(reinterpret_cast<void**>(&d_packed), packed_bytes);
    if (malloc_packed_status != cudaSuccess) {
      if (d_values != nullptr) { cudaFree(d_values); }
      throw std::runtime_error(cuda_error(malloc_packed_status, "cudaMalloc(d_packed)"));
    }
  }

  auto cleanup = [&]() {
    if (d_values != nullptr) { cudaFree(d_values); }
    if (d_packed != nullptr) { cudaFree(d_packed); }
  };

  if (values_bytes > 0) {
    auto const h2d_status = cudaMemcpyAsync(
      d_values, values.data(), values_bytes, cudaMemcpyHostToDevice, stream.value());
    if (h2d_status != cudaSuccess) {
      auto err_msg = cuda_error(h2d_status, "cudaMemcpy H2D values");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  }

  if (packed_bytes > 0) {
    auto const memset_status = cudaMemsetAsync(d_packed, 0, packed_bytes, stream.value());
    if (memset_status != cudaSuccess) {
      auto err_msg = cuda_error(memset_status, "cudaMemset packed");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  }

  encode_bw37_gpu(d_values, d_packed, base_bits, total_count, stream.value());

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    auto err_msg = cuda_error(sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  if (packed_bytes > 0) {
    auto const d2h_status = cudaMemcpyAsync(
      packed.data(), d_packed, packed_bytes, cudaMemcpyDeviceToHost, stream.value());
    if (d2h_status != cudaSuccess) {
      auto err_msg = cuda_error(d2h_status, "cudaMemcpy D2H packed");
      cleanup();
      throw std::runtime_error(err_msg);
    }

    auto const d2h_sync_status = cudaStreamSynchronize(stream.value());
    if (d2h_sync_status != cudaSuccess) {
      auto err_msg = cuda_error(d2h_sync_status, "cudaStreamSynchronize");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  }

  cleanup();
  return packed;
}

[[nodiscard]] std::vector<uint64_t> decode_bw37_on_gpu(std::vector<uint64_t> const& packed,
                                                        uint64_t base_bits,
                                                        uint32_t total_count)
{
  auto const stream       = cudf::get_default_stream();
  auto const padded_count = static_cast<size_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<size_t>(fastlanes::num_vectors(total_count));
  auto const expected_packed_size = num_vectors * static_cast<size_t>(kWordsPerVector37);

  if (packed.size() != expected_packed_size) {
    throw std::invalid_argument("decode_bw37_on_gpu packed size mismatch");
  }

  std::vector<uint64_t> out_host(padded_count, 0ULL);

  auto const packed_bytes       = packed.size() * sizeof(uint64_t);
  auto const packed_alloc_words = std::max<size_t>(packed.size(), 1U);
  auto const packed_alloc_bytes = packed_alloc_words * sizeof(uint64_t);
  auto const decoded_bytes      = out_host.size() * sizeof(uint64_t);

  uint64_t* d_packed  = nullptr;
  uint64_t* d_decoded = nullptr;

  auto const malloc_packed_status = cudaMalloc(reinterpret_cast<void**>(&d_packed), packed_alloc_bytes);
  if (malloc_packed_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(malloc_packed_status, "cudaMalloc(d_packed)"));
  }

  if (decoded_bytes > 0) {
    auto const malloc_out_status = cudaMalloc(reinterpret_cast<void**>(&d_decoded), decoded_bytes);
    if (malloc_out_status != cudaSuccess) {
      cudaFree(d_packed);
      throw std::runtime_error(cuda_error(malloc_out_status, "cudaMalloc(d_decoded)"));
    }
  }

  auto cleanup = [&]() {
    if (d_packed != nullptr) { cudaFree(d_packed); }
    if (d_decoded != nullptr) { cudaFree(d_decoded); }
  };

  if (packed_bytes > 0) {
    auto const h2d_status = cudaMemcpyAsync(
      d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice, stream.value());
    if (h2d_status != cudaSuccess) {
      auto err_msg = cuda_error(h2d_status, "cudaMemcpy H2D packed");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  } else {
    auto const packed_init_status = cudaMemsetAsync(d_packed, 0, packed_alloc_bytes, stream.value());
    if (packed_init_status != cudaSuccess) {
      auto err_msg = cuda_error(packed_init_status, "cudaMemset packed sentinel");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  }

  if (decoded_bytes > 0) {
    auto const memset_status = cudaMemsetAsync(d_decoded, 0, decoded_bytes, stream.value());
    if (memset_status != cudaSuccess) {
      auto err_msg = cuda_error(memset_status, "cudaMemset decoded");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  }

  decode_bw37_gpu(d_packed, d_decoded, base_bits, total_count, stream.value());

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    auto err_msg = cuda_error(sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  if (decoded_bytes > 0) {
    auto const d2h_status = cudaMemcpyAsync(
      out_host.data(), d_decoded, decoded_bytes, cudaMemcpyDeviceToHost, stream.value());
    if (d2h_status != cudaSuccess) {
      auto err_msg = cuda_error(d2h_status, "cudaMemcpy D2H decoded");
      cleanup();
      throw std::runtime_error(err_msg);
    }

    auto const d2h_sync_status = cudaStreamSynchronize(stream.value());
    if (d2h_sync_status != cudaSuccess) {
      auto err_msg = cuda_error(d2h_sync_status, "cudaStreamSynchronize");
      cleanup();
      throw std::runtime_error(err_msg);
    }
  }

  cleanup();
  return out_host;
}

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

[[nodiscard]] parity_evidence run_bw37_parity_case(uint32_t total_count,
                                                   uint64_t base_bits,
                                                   uint64_t seed,
                                                   data_pattern pattern)
{
  auto const deltas = make_padded_bw37_deltas(total_count, seed, pattern);
  auto const values = add_base_to_deltas(deltas, base_bits);

  auto const cpu_packed = pack_bw37_vectors_cpu(deltas);
  auto const gpu_packed = encode_bw37_on_gpu(values, base_bits, total_count);

  auto const cpu_unpacked      = unpack_bw37_vectors_cpu(cpu_packed, deltas.size());
  auto const gpu_decode_cpu_pk = decode_bw37_on_gpu(cpu_packed, base_bits, total_count);
  auto const gpu_roundtrip     = decode_bw37_on_gpu(gpu_packed, base_bits, total_count);

  auto evidence = parity_evidence{.cpu_packed_checksum = checksum_words(cpu_packed),
                                  .gpu_packed_checksum = checksum_words(gpu_packed)};

  for (size_t i = 0; i < cpu_packed.size(); ++i) {
    if (cpu_packed[i] != gpu_packed[i]) { ++evidence.gpu_pack_mismatch; }
  }

  for (uint32_t i = 0; i < total_count; ++i) {
    if (cpu_unpacked[i] != deltas[i]) { ++evidence.cpu_roundtrip_mismatch; }

    auto const cpu_bits = base_bits + cpu_unpacked[i];
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

}  // namespace

TEST_F(ParquetFastLanesNative64Bw37Test, CpuOracleBoundaryFixturesDeterministic)
{
  std::array<uint8_t, 8> constexpr boundary_bitwidths = {0, 1, 31, 32, 33, 37, 63, 64};

  std::ostringstream report;
  report << "FL64-R1 boundary checksums:";

  for (auto const bw : boundary_bitwidths) {
    auto const mask = bitwidth_mask(bw);

    std::vector<uint64_t> values(kVectorSize, 0ULL);
    for (uint32_t i = 0; i < kVectorSize; ++i) {
      values[i] = splitmix64(0x41ab0f8d29c5d7e3ULL + i + bw * 131ULL) & mask;
    }

    auto const packed_first  = pack_one_vector_cpu(values, bw);
    auto const packed_second = pack_one_vector_cpu(values, bw);
    auto const unpacked      = unpack_one_vector_cpu(packed_first, bw);

    EXPECT_EQ(packed_first, packed_second);
    EXPECT_EQ(unpacked, values);

    report << " bw" << static_cast<int>(bw) << "=0x" << std::hex << checksum_words(packed_first)
           << std::dec;
  }

  // std::cout << report.str() << std::endl;
}

TEST_F(ParquetFastLanesNative64Bw37Test, CpuOracleBw37AdversarialFixtureDeterministic)
{
  std::vector<uint64_t> values(kVectorSize, 0ULL);
  for (uint32_t i = 0; i < kVectorSize; ++i) {
    values[i] = make_bw37_delta(i, 0x6d3a8f12b7c44a1fULL, data_pattern::adversarial);
  }

  auto const packed_first  = pack_one_vector_cpu(values, kBw37);
  auto const packed_second = pack_one_vector_cpu(values, kBw37);
  auto const unpacked      = unpack_one_vector_cpu(packed_first, kBw37);

  EXPECT_EQ(packed_first, packed_second);
  EXPECT_EQ(unpacked, values);

  std::cout << "FL64-R1 bw37 adversarial checksum: 0x" << std::hex << checksum_words(packed_first)
            << std::dec << std::endl;
}

TEST_F(ParquetFastLanesNative64Bw37Test, GpuBw37ParityRandomizedAndAdversarial)
{
  std::array<uint64_t, 3> constexpr seeds = {
    0x00000000baddf00dULL, 0x0000000012345678ULL, 0x00000000deadbeefULL};
  std::array<data_pattern, 2> constexpr patterns = {
    data_pattern::randomized, data_pattern::adversarial};
  std::array<uint64_t, 2> const base_bits = {
    0ULL,
    fastlanes::int64_to_u64_bits(std::numeric_limits<int64_t>::min() + (int64_t{1} << 40))};

  std::ostringstream report;
  report << "FL64-R2 parity:";

  for (auto const seed : seeds) {
    for (auto const pattern : patterns) {
      for (auto const base : base_bits) {
        auto const evidence = run_bw37_parity_case(2050, base, seed, pattern);

        EXPECT_EQ(evidence.cpu_roundtrip_mismatch, 0U);
        EXPECT_EQ(evidence.gpu_pack_mismatch, 0U);
        EXPECT_EQ(evidence.decode_from_cpu_mismatch_u64, 0U);
        EXPECT_EQ(evidence.decode_from_cpu_mismatch_i64, 0U);
        EXPECT_EQ(evidence.roundtrip_mismatch_u64, 0U);
        EXPECT_EQ(evidence.roundtrip_mismatch_i64, 0U);

        report << " [seed=0x" << std::hex << seed << std::dec
               << ",pattern=" << pattern_name(pattern) << ",base=0x" << std::hex << base
               << std::dec << ",pack=" << evidence.gpu_pack_mismatch
               << ",dec_u64=" << evidence.decode_from_cpu_mismatch_u64
               << ",dec_i64=" << evidence.decode_from_cpu_mismatch_i64
               << ",rt_u64=" << evidence.roundtrip_mismatch_u64
               << ",rt_i64=" << evidence.roundtrip_mismatch_i64 << ",cpu_chk=0x" << std::hex
               << evidence.cpu_packed_checksum << ",gpu_chk=0x" << evidence.gpu_packed_checksum
               << std::dec << "]";
      }
    }
  }

  // std::cout << report.str() << std::endl;
}

TEST_F(ParquetFastLanesNative64Bw37Test, GpuBw37ParityTailBoundariesExtremeAndPathological)
{
  std::array<uint32_t, 6> constexpr counts = {1023, 1024, 1025, 2047, 2048, 2049};
  std::array<uint64_t, 3> const bases = {
    fastlanes::int64_to_u64_bits(std::numeric_limits<int64_t>::min() + 1024),
    fastlanes::int64_to_u64_bits(std::numeric_limits<int64_t>::max() - static_cast<int64_t>(kBw37Mask)),
    std::numeric_limits<uint64_t>::max() - kBw37Mask};

  std::ostringstream report;
  report << "FL64-R3 boundary/extreme:";

  for (auto const count : counts) {
    for (auto const base : bases) {
      auto const evidence =
        run_bw37_parity_case(count, base, 0x5a50e4c34bd1c7f9ULL + count, data_pattern::pathological);

      EXPECT_EQ(evidence.cpu_roundtrip_mismatch, 0U);
      EXPECT_EQ(evidence.gpu_pack_mismatch, 0U);
      EXPECT_EQ(evidence.decode_from_cpu_mismatch_u64, 0U);
      EXPECT_EQ(evidence.decode_from_cpu_mismatch_i64, 0U);
      EXPECT_EQ(evidence.roundtrip_mismatch_u64, 0U);
      EXPECT_EQ(evidence.roundtrip_mismatch_i64, 0U);

      report << " [n=" << count << ",base=0x" << std::hex << base << std::dec
             << ",pack=" << evidence.gpu_pack_mismatch
             << ",dec_u64=" << evidence.decode_from_cpu_mismatch_u64
             << ",dec_i64=" << evidence.decode_from_cpu_mismatch_i64
             << ",rt_u64=" << evidence.roundtrip_mismatch_u64
             << ",rt_i64=" << evidence.roundtrip_mismatch_i64 << ",cpu_chk=0x" << std::hex
             << evidence.cpu_packed_checksum << ",gpu_chk=0x" << evidence.gpu_packed_checksum
             << std::dec << "]";
    }
  }

  // std::cout << report.str() << std::endl;
}

TEST_F(ParquetFastLanesNative64Bw37Test, GpuBw37ParityRepeatedRunStability)
{
  std::array<uint64_t, 3> constexpr seeds = {
    0x0000000000000011ULL, 0x000000000000001dULL, 0x000000000000002fULL};

  std::ostringstream report;
  report << "FL64-R3 stability:";

  constexpr uint32_t repeats = 3;
  for (uint32_t rep = 0; rep < repeats; ++rep) {
    for (auto const seed : seeds) {
      auto const evidence = run_bw37_parity_case(
        2049,
        fastlanes::int64_to_u64_bits(std::numeric_limits<int64_t>::min() + (int64_t{1} << 41)),
        seed,
        data_pattern::randomized);

      EXPECT_EQ(evidence.cpu_roundtrip_mismatch, 0U);
      EXPECT_EQ(evidence.gpu_pack_mismatch, 0U);
      EXPECT_EQ(evidence.decode_from_cpu_mismatch_u64, 0U);
      EXPECT_EQ(evidence.decode_from_cpu_mismatch_i64, 0U);
      EXPECT_EQ(evidence.roundtrip_mismatch_u64, 0U);
      EXPECT_EQ(evidence.roundtrip_mismatch_i64, 0U);

      report << " [run=" << rep << ",seed=0x" << std::hex << seed << std::dec
             << ",pack=" << evidence.gpu_pack_mismatch
             << ",dec_u64=" << evidence.decode_from_cpu_mismatch_u64
             << ",dec_i64=" << evidence.decode_from_cpu_mismatch_i64
             << ",rt_u64=" << evidence.roundtrip_mismatch_u64
             << ",rt_i64=" << evidence.roundtrip_mismatch_i64 << ",cpu_chk=0x" << std::hex
             << evidence.cpu_packed_checksum << ",gpu_chk=0x" << evidence.gpu_packed_checksum
             << std::dec << "]";
    }
  }

  // std::cout << report.str() << std::endl;
}
