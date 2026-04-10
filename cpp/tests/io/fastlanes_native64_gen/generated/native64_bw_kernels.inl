/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * generated! Do not edit by hand.
 * Source template: cpp/tests/io/fastlanes_native64_gen/templates/native64_bw_kernels.inl.tmpl
 */

namespace native64_generated {

  // TODO只有这个文件需要留下来，其他的脚本文件都没什么用，开发完就可以删掉
  // 另一个文件里有一堆Row估计也没什么用，可以考虑删掉，或者至少不放在这个目录里
  // 还有就是可以建立一个Tag用来跟踪当前分支，因为最后发的文件肯定是要删掉的
  // 此外现在的似乎只是Pack？虽然现在的Delta需要自己提供最小值，可能是一个优化点，但如果FL64也需要提供最小值的话，感觉就没什么意义了，直接放在一起算了
  // 就是不知道为什么GPU编码也给我实现了，估计还得改

struct bw_row {
  uint8_t bw;
  uint32_t words_per_lane;
  uint32_t words_per_vector;
  uint8_t crossing_count;
};

inline constexpr std::array<bw_row, 65> kBwRows = {{
  bw_row{0u, 0u, 0u, 0u},
  bw_row{1u, 1u, 16u, 0u},
  bw_row{2u, 2u, 32u, 0u},
  bw_row{3u, 3u, 48u, 2u},
  bw_row{4u, 4u, 64u, 0u},
  bw_row{5u, 5u, 80u, 4u},
  bw_row{6u, 6u, 96u, 4u},
  bw_row{7u, 7u, 112u, 6u},
  bw_row{8u, 8u, 128u, 0u},
  bw_row{9u, 9u, 144u, 8u},
  bw_row{10u, 10u, 160u, 8u},
  bw_row{11u, 11u, 176u, 10u},
  bw_row{12u, 12u, 192u, 8u},
  bw_row{13u, 13u, 208u, 12u},
  bw_row{14u, 14u, 224u, 12u},
  bw_row{15u, 15u, 240u, 14u},
  bw_row{16u, 16u, 256u, 0u},
  bw_row{17u, 17u, 272u, 16u},
  bw_row{18u, 18u, 288u, 16u},
  bw_row{19u, 19u, 304u, 18u},
  bw_row{20u, 20u, 320u, 16u},
  bw_row{21u, 21u, 336u, 20u},
  bw_row{22u, 22u, 352u, 20u},
  bw_row{23u, 23u, 368u, 22u},
  bw_row{24u, 24u, 384u, 16u},
  bw_row{25u, 25u, 400u, 24u},
  bw_row{26u, 26u, 416u, 24u},
  bw_row{27u, 27u, 432u, 26u},
  bw_row{28u, 28u, 448u, 24u},
  bw_row{29u, 29u, 464u, 28u},
  bw_row{30u, 30u, 480u, 28u},
  bw_row{31u, 31u, 496u, 30u},
  bw_row{32u, 32u, 512u, 0u},
  bw_row{33u, 33u, 528u, 32u},
  bw_row{34u, 34u, 544u, 32u},
  bw_row{35u, 35u, 560u, 34u},
  bw_row{36u, 36u, 576u, 32u},
  bw_row{37u, 37u, 592u, 36u},
  bw_row{38u, 38u, 608u, 36u},
  bw_row{39u, 39u, 624u, 38u},
  bw_row{40u, 40u, 640u, 32u},
  bw_row{41u, 41u, 656u, 40u},
  bw_row{42u, 42u, 672u, 40u},
  bw_row{43u, 43u, 688u, 42u},
  bw_row{44u, 44u, 704u, 40u},
  bw_row{45u, 45u, 720u, 44u},
  bw_row{46u, 46u, 736u, 44u},
  bw_row{47u, 47u, 752u, 46u},
  bw_row{48u, 48u, 768u, 32u},
  bw_row{49u, 49u, 784u, 48u},
  bw_row{50u, 50u, 800u, 48u},
  bw_row{51u, 51u, 816u, 50u},
  bw_row{52u, 52u, 832u, 48u},
  bw_row{53u, 53u, 848u, 52u},
  bw_row{54u, 54u, 864u, 52u},
  bw_row{55u, 55u, 880u, 54u},
  bw_row{56u, 56u, 896u, 48u},
  bw_row{57u, 57u, 912u, 56u},
  bw_row{58u, 58u, 928u, 56u},
  bw_row{59u, 59u, 944u, 58u},
  bw_row{60u, 60u, 960u, 56u},
  bw_row{61u, 61u, 976u, 60u},
  bw_row{62u, 62u, 992u, 60u},
  bw_row{63u, 63u, 1008u, 62u},
  bw_row{64u, 64u, 1024u, 0u},
}};

template <uint8_t BW>
__host__ __device__ constexpr uint64_t mask_for_bw()
{
  if constexpr (BW == 0) {
    return 0ULL;
  } else if constexpr (BW >= 64) {
    return ~uint64_t{0};
  } else {
    return (uint64_t{1} << BW) - 1ULL;
  }
}

template <uint8_t BW>
__host__ __device__ constexpr uint32_t words_per_vector_for_bw()
{
  return static_cast<uint32_t>(fastlanes::encoded_size_bytes(kVectorSize, BW) / sizeof(uint64_t));
}

template <uint8_t BW>
__host__ __device__ constexpr uint32_t words_per_lane_for_bw()
{
  return words_per_vector_for_bw<BW>() / kLanesPerVector;
}

template <uint8_t BW>
__device__ __forceinline__ void encode_lane_native64(uint64_t const* __restrict vector_in,
                                                     uint64_t* __restrict vector_out,
                                                     uint32_t lane,
                                                     uint64_t base_bits)
{
  if constexpr (BW == 0) {
    return;
  } else {
    constexpr uint32_t kWordsPerLane = words_per_lane_for_bw<BW>();
    constexpr uint64_t kMask         = mask_for_bw<BW>();

    uint64_t words[kWordsPerLane] = {};

    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
      auto const in_idx    = static_cast<size_t>(i) * kLanesPerVector + lane;
      auto const delta     = (vector_in[in_idx] - base_bits) & kMask;
      auto const bit_index = i * BW;
      auto const word_idx  = bit_index >> 6;
      auto const bit_off   = bit_index & 63;

      words[word_idx] |= delta << bit_off;
      if (bit_off > (64 - BW)) {
        words[word_idx + 1] |= delta >> (64 - bit_off);
      }
    }

    for (uint32_t word_idx = 0; word_idx < kWordsPerLane; ++word_idx) {
      auto const out_idx = static_cast<size_t>(word_idx) * kLanesPerVector + lane;
      vector_out[out_idx] = words[word_idx];
    }
  }
}

template <uint8_t BW>
__device__ __forceinline__ void decode_lane_native64(uint64_t const* __restrict vector_in,
                                                     uint64_t* __restrict vector_out,
                                                     uint32_t lane,
                                                     uint64_t base_bits)
{
  if constexpr (BW == 0) {
    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
      auto const out_idx = i * kLanesPerVector + lane;
      vector_out[out_idx] = base_bits;
    }
  } else {
    constexpr uint64_t kMask = mask_for_bw<BW>();

    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
      auto const out_idx   = i * kLanesPerVector + lane;
      auto const bit_index = i * BW;
      auto const word_idx  = bit_index >> 6;
      auto const bit_off   = bit_index & 63;
      auto const lo_idx    = static_cast<size_t>(word_idx) * kLanesPerVector + lane;
      auto const lo_word   = vector_in[lo_idx];
      auto delta           = lo_word >> bit_off;

      if (bit_off > (64 - BW)) {
        auto const hi_idx = (static_cast<size_t>(word_idx) + 1) * kLanesPerVector + lane;
        auto const hi_word = vector_in[hi_idx];
        delta |= hi_word << (64 - bit_off);
      }

      vector_out[out_idx] = base_bits + (delta & kMask);
    }
  }
}

template <uint8_t BW>
__global__ void encode_native64_test_kernel(uint64_t const* __restrict values,
                                            uint64_t* __restrict packed,
                                            uint64_t base_bits,
                                            uint32_t padded_count)
{
  auto const lane      = static_cast<uint32_t>(threadIdx.x);
  auto const vector_id = static_cast<uint32_t>(blockIdx.x);

  if (lane >= kLanesPerVector) { return; }

  auto const vector_start = vector_id * kVectorSize;
  if (vector_start >= padded_count) { return; }

  auto const* vector_in = values + vector_start;
  auto* vector_out      = packed + static_cast<size_t>(vector_id) * words_per_vector_for_bw<BW>();
  encode_lane_native64<BW>(vector_in, vector_out, lane, base_bits);
}

template <uint8_t BW>
__global__ void decode_native64_test_kernel(uint64_t const* __restrict packed,
                                            uint64_t* __restrict decoded,
                                            uint64_t base_bits,
                                            uint32_t padded_count)
{
  auto const lane      = static_cast<uint32_t>(threadIdx.x);
  auto const vector_id = static_cast<uint32_t>(blockIdx.x);

  if (lane >= kLanesPerVector) { return; }

  auto const vector_start = vector_id * kVectorSize;
  if (vector_start >= padded_count) { return; }

  auto const* vector_in = packed + static_cast<size_t>(vector_id) * words_per_vector_for_bw<BW>();
  auto* vector_out      = decoded + vector_start;
  decode_lane_native64<BW>(vector_in, vector_out, lane, base_bits);
}

template <uint8_t BW>
[[nodiscard]] std::vector<uint64_t> encode_on_gpu(std::vector<uint64_t> const& values,
                                                  uint64_t base_bits,
                                                  uint32_t total_count)
{
  auto const stream      = cudf::get_default_stream();
  auto const padded_count = static_cast<uint32_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));

  if (values.size() != padded_count) {
    throw std::invalid_argument("encode_on_gpu values size mismatch");
  }

  constexpr auto kWordsPerVector = words_per_vector_for_bw<BW>();
  std::vector<uint64_t> packed(static_cast<size_t>(num_vectors) * kWordsPerVector, 0ULL);

  if constexpr (BW == 0) {
    return packed;
  }

  uint64_t* d_values = nullptr;
  uint64_t* d_packed = nullptr;

  auto const values_bytes = values.size() * sizeof(uint64_t);
  auto const packed_bytes = packed.size() * sizeof(uint64_t);

  auto const malloc_values_status = cudaMalloc(reinterpret_cast<void**>(&d_values), values_bytes);
  if (malloc_values_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(malloc_values_status, "cudaMalloc(d_values)"));
  }

  auto const malloc_packed_status = cudaMalloc(reinterpret_cast<void**>(&d_packed), packed_bytes);
  if (malloc_packed_status != cudaSuccess) {
    cudaFree(d_values);
    throw std::runtime_error(cuda_error(malloc_packed_status, "cudaMalloc(d_packed)"));
  }

  auto cleanup = [&]() {
    cudaFree(d_values);
    cudaFree(d_packed);
  };

  auto const h2d_status = cudaMemcpyAsync(
    d_values, values.data(), values_bytes, cudaMemcpyHostToDevice, stream.value());
  if (h2d_status != cudaSuccess) {
    auto err_msg = cuda_error(h2d_status, "cudaMemcpy H2D values");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const memset_status = cudaMemsetAsync(d_packed, 0, packed_bytes, stream.value());
  if (memset_status != cudaSuccess) {
    auto err_msg = cuda_error(memset_status, "cudaMemset packed");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  encode_native64_test_kernel<BW><<<num_vectors, kLanesPerVector, 0, stream.value()>>>(
    d_values, d_packed, base_bits, padded_count);

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    auto err_msg = cuda_error(launch_status, "native64 encode kernel launch");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    auto err_msg = cuda_error(sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

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

  cleanup();
  return packed;
}

template <uint8_t BW>
[[nodiscard]] std::vector<uint64_t> decode_on_gpu(std::vector<uint64_t> const& packed,
                                                  uint64_t base_bits,
                                                  uint32_t total_count)
{
  auto const stream      = cudf::get_default_stream();
  auto const padded_count = static_cast<size_t>(fastlanes::padded_count(total_count));
  auto const num_vectors  = static_cast<uint32_t>(fastlanes::num_vectors(total_count));

  constexpr auto kWordsPerVector = words_per_vector_for_bw<BW>();
  if (packed.size() != static_cast<size_t>(num_vectors) * kWordsPerVector) {
    throw std::invalid_argument("decode_on_gpu packed size mismatch");
  }

  std::vector<uint64_t> out_host(padded_count, 0ULL);

  if constexpr (BW == 0) {
    std::fill(out_host.begin(), out_host.end(), base_bits);
    return out_host;
  }

  uint64_t* d_packed  = nullptr;
  uint64_t* d_decoded = nullptr;

  auto const packed_bytes  = packed.size() * sizeof(uint64_t);
  auto const decoded_bytes = out_host.size() * sizeof(uint64_t);

  auto const malloc_packed_status = cudaMalloc(reinterpret_cast<void**>(&d_packed), packed_bytes);
  if (malloc_packed_status != cudaSuccess) {
    throw std::runtime_error(cuda_error(malloc_packed_status, "cudaMalloc(d_packed)"));
  }

  auto const malloc_out_status = cudaMalloc(reinterpret_cast<void**>(&d_decoded), decoded_bytes);
  if (malloc_out_status != cudaSuccess) {
    cudaFree(d_packed);
    throw std::runtime_error(cuda_error(malloc_out_status, "cudaMalloc(d_decoded)"));
  }

  auto cleanup = [&]() {
    cudaFree(d_packed);
    cudaFree(d_decoded);
  };

  auto const h2d_status = cudaMemcpyAsync(
    d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice, stream.value());
  if (h2d_status != cudaSuccess) {
    auto err_msg = cuda_error(h2d_status, "cudaMemcpy H2D packed");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const memset_status = cudaMemsetAsync(d_decoded, 0, decoded_bytes, stream.value());
  if (memset_status != cudaSuccess) {
    auto err_msg = cuda_error(memset_status, "cudaMemset decoded");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  decode_native64_test_kernel<BW><<<num_vectors, kLanesPerVector, 0, stream.value()>>>(
    d_packed, d_decoded, base_bits, static_cast<uint32_t>(padded_count));

  auto const launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    auto err_msg = cuda_error(launch_status, "native64 decode kernel launch");
    cleanup();
    throw std::runtime_error(err_msg);
  }

  auto const sync_status = cudaStreamSynchronize(stream.value());
  if (sync_status != cudaSuccess) {
    auto err_msg = cuda_error(sync_status, "cudaStreamSynchronize");
    cleanup();
    throw std::runtime_error(err_msg);
  }

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

  cleanup();
  return out_host;
}

using encode_dispatch_fn =
  std::vector<uint64_t> (*)(std::vector<uint64_t> const&, uint64_t, uint32_t);
using decode_dispatch_fn =
  std::vector<uint64_t> (*)(std::vector<uint64_t> const&, uint64_t, uint32_t);

template <uint8_t BW>
[[nodiscard]] std::vector<uint64_t> encode_dispatch_wrapper(std::vector<uint64_t> const& values,
                                                            uint64_t base_bits,
                                                            uint32_t total_count)
{
  return encode_on_gpu<BW>(values, base_bits, total_count);
}

template <uint8_t BW>
[[nodiscard]] std::vector<uint64_t> decode_dispatch_wrapper(std::vector<uint64_t> const& packed,
                                                            uint64_t base_bits,
                                                            uint32_t total_count)
{
  return decode_on_gpu<BW>(packed, base_bits, total_count);
}

inline constexpr std::array<encode_dispatch_fn, 65> kEncodeDispatch = {{
  &encode_dispatch_wrapper<0>,
  &encode_dispatch_wrapper<1>,
  &encode_dispatch_wrapper<2>,
  &encode_dispatch_wrapper<3>,
  &encode_dispatch_wrapper<4>,
  &encode_dispatch_wrapper<5>,
  &encode_dispatch_wrapper<6>,
  &encode_dispatch_wrapper<7>,
  &encode_dispatch_wrapper<8>,
  &encode_dispatch_wrapper<9>,
  &encode_dispatch_wrapper<10>,
  &encode_dispatch_wrapper<11>,
  &encode_dispatch_wrapper<12>,
  &encode_dispatch_wrapper<13>,
  &encode_dispatch_wrapper<14>,
  &encode_dispatch_wrapper<15>,
  &encode_dispatch_wrapper<16>,
  &encode_dispatch_wrapper<17>,
  &encode_dispatch_wrapper<18>,
  &encode_dispatch_wrapper<19>,
  &encode_dispatch_wrapper<20>,
  &encode_dispatch_wrapper<21>,
  &encode_dispatch_wrapper<22>,
  &encode_dispatch_wrapper<23>,
  &encode_dispatch_wrapper<24>,
  &encode_dispatch_wrapper<25>,
  &encode_dispatch_wrapper<26>,
  &encode_dispatch_wrapper<27>,
  &encode_dispatch_wrapper<28>,
  &encode_dispatch_wrapper<29>,
  &encode_dispatch_wrapper<30>,
  &encode_dispatch_wrapper<31>,
  &encode_dispatch_wrapper<32>,
  &encode_dispatch_wrapper<33>,
  &encode_dispatch_wrapper<34>,
  &encode_dispatch_wrapper<35>,
  &encode_dispatch_wrapper<36>,
  &encode_dispatch_wrapper<37>,
  &encode_dispatch_wrapper<38>,
  &encode_dispatch_wrapper<39>,
  &encode_dispatch_wrapper<40>,
  &encode_dispatch_wrapper<41>,
  &encode_dispatch_wrapper<42>,
  &encode_dispatch_wrapper<43>,
  &encode_dispatch_wrapper<44>,
  &encode_dispatch_wrapper<45>,
  &encode_dispatch_wrapper<46>,
  &encode_dispatch_wrapper<47>,
  &encode_dispatch_wrapper<48>,
  &encode_dispatch_wrapper<49>,
  &encode_dispatch_wrapper<50>,
  &encode_dispatch_wrapper<51>,
  &encode_dispatch_wrapper<52>,
  &encode_dispatch_wrapper<53>,
  &encode_dispatch_wrapper<54>,
  &encode_dispatch_wrapper<55>,
  &encode_dispatch_wrapper<56>,
  &encode_dispatch_wrapper<57>,
  &encode_dispatch_wrapper<58>,
  &encode_dispatch_wrapper<59>,
  &encode_dispatch_wrapper<60>,
  &encode_dispatch_wrapper<61>,
  &encode_dispatch_wrapper<62>,
  &encode_dispatch_wrapper<63>,
  &encode_dispatch_wrapper<64>,
}};

inline constexpr std::array<decode_dispatch_fn, 65> kDecodeDispatch = {{
  &decode_dispatch_wrapper<0>,
  &decode_dispatch_wrapper<1>,
  &decode_dispatch_wrapper<2>,
  &decode_dispatch_wrapper<3>,
  &decode_dispatch_wrapper<4>,
  &decode_dispatch_wrapper<5>,
  &decode_dispatch_wrapper<6>,
  &decode_dispatch_wrapper<7>,
  &decode_dispatch_wrapper<8>,
  &decode_dispatch_wrapper<9>,
  &decode_dispatch_wrapper<10>,
  &decode_dispatch_wrapper<11>,
  &decode_dispatch_wrapper<12>,
  &decode_dispatch_wrapper<13>,
  &decode_dispatch_wrapper<14>,
  &decode_dispatch_wrapper<15>,
  &decode_dispatch_wrapper<16>,
  &decode_dispatch_wrapper<17>,
  &decode_dispatch_wrapper<18>,
  &decode_dispatch_wrapper<19>,
  &decode_dispatch_wrapper<20>,
  &decode_dispatch_wrapper<21>,
  &decode_dispatch_wrapper<22>,
  &decode_dispatch_wrapper<23>,
  &decode_dispatch_wrapper<24>,
  &decode_dispatch_wrapper<25>,
  &decode_dispatch_wrapper<26>,
  &decode_dispatch_wrapper<27>,
  &decode_dispatch_wrapper<28>,
  &decode_dispatch_wrapper<29>,
  &decode_dispatch_wrapper<30>,
  &decode_dispatch_wrapper<31>,
  &decode_dispatch_wrapper<32>,
  &decode_dispatch_wrapper<33>,
  &decode_dispatch_wrapper<34>,
  &decode_dispatch_wrapper<35>,
  &decode_dispatch_wrapper<36>,
  &decode_dispatch_wrapper<37>,
  &decode_dispatch_wrapper<38>,
  &decode_dispatch_wrapper<39>,
  &decode_dispatch_wrapper<40>,
  &decode_dispatch_wrapper<41>,
  &decode_dispatch_wrapper<42>,
  &decode_dispatch_wrapper<43>,
  &decode_dispatch_wrapper<44>,
  &decode_dispatch_wrapper<45>,
  &decode_dispatch_wrapper<46>,
  &decode_dispatch_wrapper<47>,
  &decode_dispatch_wrapper<48>,
  &decode_dispatch_wrapper<49>,
  &decode_dispatch_wrapper<50>,
  &decode_dispatch_wrapper<51>,
  &decode_dispatch_wrapper<52>,
  &decode_dispatch_wrapper<53>,
  &decode_dispatch_wrapper<54>,
  &decode_dispatch_wrapper<55>,
  &decode_dispatch_wrapper<56>,
  &decode_dispatch_wrapper<57>,
  &decode_dispatch_wrapper<58>,
  &decode_dispatch_wrapper<59>,
  &decode_dispatch_wrapper<60>,
  &decode_dispatch_wrapper<61>,
  &decode_dispatch_wrapper<62>,
  &decode_dispatch_wrapper<63>,
  &decode_dispatch_wrapper<64>,
}};

[[nodiscard]] inline std::vector<uint64_t> encode_by_bw_gpu(uint8_t bw,
                                                            std::vector<uint64_t> const& values,
                                                            uint64_t base_bits,
                                                            uint32_t total_count)
{
  if (bw > 64) { throw std::invalid_argument("encode_by_bw_gpu requires bw in [0,64]"); }
  return kEncodeDispatch[bw](values, base_bits, total_count);
}

[[nodiscard]] inline std::vector<uint64_t> decode_by_bw_gpu(uint8_t bw,
                                                            std::vector<uint64_t> const& packed,
                                                            uint64_t base_bits,
                                                            uint32_t total_count)
{
  if (bw > 64) { throw std::invalid_argument("decode_by_bw_gpu requires bw in [0,64]"); }
  return kDecodeDispatch[bw](packed, base_bits, total_count);
}

[[nodiscard]] inline bw_row row_for_bw(uint8_t bw)
{
  if (bw > 64) { throw std::invalid_argument("row_for_bw requires bw in [0,64]"); }
  return kBwRows[bw];
}

}  // namespace native64_generated
