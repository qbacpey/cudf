/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/fastlanes/common.cuh>
#include <cudf/fastlanes/fls_gen/pack/pack.hpp>
#include <cudf/fastlanes/fls_gen/unpack/unpack.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

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

namespace native64_generated {

struct derived_metadata_result {
  native64_encode_metadata metadata;
  uint64_t base_bits_from_device{};
};

[[nodiscard]] derived_metadata_result derive_metadata_gpu(std::vector<uint64_t> const& values,
                                                          uint32_t total_count)
{
  auto const stream       = cudf::get_default_stream();
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));

  if (values.size() != padded_count) {
    throw std::invalid_argument("derive_metadata_gpu values size mismatch");
  }

  uint64_t* d_values    = nullptr;
  uint64_t* d_base_bits = nullptr;

  auto cleanup = [&]() {
    if (d_values != nullptr) { cudaFree(d_values); }
    if (d_base_bits != nullptr) { cudaFree(d_base_bits); }
  };

  auto const values_bytes = values.size() * sizeof(uint64_t);

  auto const malloc_base_status = cudaMalloc(reinterpret_cast<void**>(&d_base_bits), sizeof(uint64_t));
  if (malloc_base_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(malloc_base_status, "cudaMalloc(d_base_bits)"));
  }

  if (values_bytes > 0) {
    auto const malloc_values_status = cudaMalloc(reinterpret_cast<void**>(&d_values), values_bytes);
    if (malloc_values_status != cudaSuccess) {
      cleanup();
      throw std::runtime_error(cuda_error(malloc_values_status, "cudaMalloc(d_values)"));
    }

    auto const h2d_status = cudaMemcpyAsync(
      d_values, values.data(), values_bytes, cudaMemcpyHostToDevice, stream.value());
    if (h2d_status != cudaSuccess) {
      cleanup();
      throw std::runtime_error(cuda_error(h2d_status, "cudaMemcpy H2D values"));
    }
  }

  native64_encode_metadata metadata{};
  derive_min_base_bits_metadata_to_device(
    d_values, total_count, d_base_bits, stream.value(), &metadata.base_bits);

  uint64_t base_bits_from_device{};
  auto const d2h_status = cudaMemcpyAsync(
    &base_bits_from_device, d_base_bits, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.value());
  if (d2h_status != cudaSuccess) {
    cleanup();
    throw std::runtime_error(cuda_error(d2h_status, "cudaMemcpy D2H base_bits"));
  }

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    cleanup();
    throw std::runtime_error(cuda_error(sync_status, "cudaStreamSynchronize"));
  }

  cleanup();
  return {metadata, base_bits_from_device};
}

[[nodiscard]] std::vector<uint64_t> encode_by_bw_gpu(uint8_t bw,
                                                     std::vector<uint64_t> const& values,
                                                     uint64_t base_bits,
                                                     uint32_t total_count)
{
  if (bw > 64) { throw std::invalid_argument("encode_by_bw_gpu requires bw in [0,64]"); }

  auto const stream       = cudf::get_default_stream();
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));
  auto const words_per_vector =
    static_cast<size_t>(fastlanes::encoded_size_bytes(kVectorSize, bw) / sizeof(uint64_t));

  if (values.size() != padded_count) {
    throw std::invalid_argument("encode_by_bw_gpu values size mismatch");
  }

  std::vector<uint64_t> packed(static_cast<size_t>(num_vectors) * words_per_vector, 0ULL);

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

  encode_by_bw_gpu_device_ptrs(
    bw, d_values, d_packed, base_bits, total_count, stream.value());

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

[[nodiscard]] std::vector<uint64_t> decode_by_bw_gpu(uint8_t bw,
                                                     std::vector<uint64_t> const& packed,
                                                     uint64_t base_bits,
                                                     uint32_t total_count)
{
  if (bw > 64) { throw std::invalid_argument("decode_by_bw_gpu requires bw in [0,64]"); }

  auto const stream       = cudf::get_default_stream();
  auto const padded_count = static_cast<size_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<size_t>(fastlanes::num_vectors(total_count));
  auto const words_per_vector =
    static_cast<size_t>(fastlanes::encoded_size_bytes(kVectorSize, bw) / sizeof(uint64_t));
  auto const expected_packed_size = num_vectors * words_per_vector;

  if (packed.size() != expected_packed_size) {
    throw std::invalid_argument("decode_by_bw_gpu packed size mismatch");
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

  decode_by_bw_gpu_device_ptrs(
    bw, d_packed, d_decoded, base_bits, total_count, stream.value());

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

}  // namespace native64_generated

struct min_oracle {
  uint64_t base_bits{};
};

[[nodiscard]] uint8_t bitwidth_from_max_delta_cpu(uint64_t max_delta)
{
  if (max_delta == 0ULL) { return 0; }

  uint8_t bits = 0;
  while (max_delta > 0ULL) {
    max_delta >>= 1;
    ++bits;
  }
  return bits;
}

[[nodiscard]] min_oracle derive_min_cpu(std::vector<uint64_t> const& values, uint32_t total_count)
{
  if (total_count == 0) { return min_oracle{}; }
  if (values.size() < total_count) {
    throw std::invalid_argument("derive_min_cpu values size mismatch");
  }

  auto const base = *std::min_element(values.begin(), values.begin() + total_count);
  return min_oracle{base};
}

[[nodiscard]] uint8_t derive_bitwidth_cpu(std::vector<uint64_t> const& values,
                                          uint32_t total_count,
                                          uint64_t base_bits)
{
  if (total_count == 0) { return 0; }
  if (values.size() < total_count) {
    throw std::invalid_argument("derive_bitwidth_cpu values size mismatch");
  }

  uint64_t max_delta{0};
  for (uint32_t i = 0; i < total_count; ++i) {
    max_delta = std::max(max_delta, values[i] - base_bits);
  }

  return bitwidth_from_max_delta_cpu(max_delta);
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

TEST_F(ParquetFastLanesNative64GeneratedTest, MetadataMinReductionMatchesCpuOracle)
{
  std::array<uint8_t, 7> constexpr selected_bw = {0, 1, 17, 33, 37, 63, 64};
  std::array<uint32_t, 4> constexpr counts = {1, 1025, 2047, 2049};
  std::array<data_pattern, 3> constexpr patterns = {
    data_pattern::randomized, data_pattern::adversarial, data_pattern::pathological};

  for (auto const bw : selected_bw) {
    auto const bases = base_set_for_bw(bw);

    for (auto const count : counts) {
      for (auto const pattern : patterns) {
        for (auto const base : bases) {
          auto const seed = 0x9c6f5d3b17e4a221ULL + static_cast<uint64_t>(count) * 131ULL +
                            static_cast<uint64_t>(bw) * 17ULL;
          auto const deltas = make_padded_deltas(count, seed, bw, pattern);
          auto const values = add_base_to_deltas(deltas, base);

          auto const expected = derive_min_cpu(values, count);
          auto const actual   = native64_generated::derive_metadata_gpu(values, count);

          EXPECT_EQ(actual.metadata.base_bits, expected.base_bits)
            << "bw=" << static_cast<int>(bw) << " count=" << count
            << " pattern=" << pattern_name(pattern);
          EXPECT_EQ(actual.base_bits_from_device, expected.base_bits)
            << "bw=" << static_cast<int>(bw) << " count=" << count
            << " pattern=" << pattern_name(pattern);
          EXPECT_EQ(actual.metadata.base_bits, actual.base_bits_from_device)
            << "bw=" << static_cast<int>(bw) << " count=" << count
            << " pattern=" << pattern_name(pattern);
        }
      }
    }
  }
}

TEST_F(ParquetFastLanesNative64GeneratedTest, DerivedMinEncodeMatchesExplicitPath)
{
  std::array<uint8_t, 5> constexpr selected_bw = {0, 1, 33, 37, 64};
  std::array<uint32_t, 3> constexpr counts = {1023, 1025, 2049};
  std::array<data_pattern, 2> constexpr patterns = {
    data_pattern::randomized, data_pattern::adversarial};

  for (auto const bw : selected_bw) {
    auto const bases = base_set_for_bw(bw);

    for (auto const count : counts) {
      for (auto const pattern : patterns) {
        for (auto const base : bases) {
          auto const seed = 0x5f2c19d4e301a77bULL + static_cast<uint64_t>(count) * 97ULL +
                            static_cast<uint64_t>(bw) * 31ULL;
          auto const deltas = make_padded_deltas(count, seed, bw, pattern);
          auto const values = add_base_to_deltas(deltas, base);

          auto const expected_min = derive_min_cpu(values, count);
          auto const explicit_bw = derive_bitwidth_cpu(values, count, expected_min.base_bits);
          auto const explicit_packed = native64_generated::encode_by_bw_gpu(
            explicit_bw, values, expected_min.base_bits, count);
          auto const derived_min = native64_generated::derive_metadata_gpu(values, count);

          EXPECT_EQ(derived_min.metadata.base_bits, expected_min.base_bits)
            << "bw=" << static_cast<int>(bw) << " count=" << count
            << " pattern=" << pattern_name(pattern);
          EXPECT_EQ(derived_min.base_bits_from_device, expected_min.base_bits)
            << "bw=" << static_cast<int>(bw) << " count=" << count
            << " pattern=" << pattern_name(pattern);

          auto const derived_min_packed = native64_generated::encode_by_bw_gpu(
            explicit_bw, values, derived_min.metadata.base_bits, count);
          EXPECT_EQ(derived_min_packed, explicit_packed)
            << "bw=" << static_cast<int>(bw) << " count=" << count
            << " pattern=" << pattern_name(pattern);

          auto const decoded = native64_generated::decode_by_bw_gpu(
            explicit_bw, derived_min_packed, derived_min.metadata.base_bits, count);
          for (uint32_t i = 0; i < count; ++i) {
            EXPECT_EQ(decoded[i], values[i])
              << "bw=" << static_cast<int>(bw) << " count=" << count
              << " pattern=" << pattern_name(pattern) << " index=" << i;
          }
        }
      }
    }
  }
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
