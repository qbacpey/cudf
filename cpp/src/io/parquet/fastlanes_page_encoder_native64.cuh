/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Note (FL-P4-R3): this header is intentionally source-included from
// `fastlanes_page_encoder.cu`, which is itself source-included from `page_enc.cu`. It
// inherits the enclosing `cudf::io::parquet::detail::` namespace plus the local templated
// kernels defined in `page_enc.cu` (e.g. `gpuGatherSinglePageTyped`, `gpuEncodePageLevels`,
// `gpuEncodeCpuPages`).
//
// NATIVE64 is a GPU-encoded FastLanes path: the encoder library uploads only the page
// header H->D, then launches `launch_native64_encode` to write the encoded payload directly
// into the device blob's payload region (see [cpp/src/fastlanes/encode_native64.cu] lines
// 99-102). The full encoded payload never makes a host-side round trip. The cudf-side flow
// below still pairs the gather kernel with the encoder's `encode_pages` call and runs the
// shared batched header validator (one D->H probe for headers), but no host-side blob
// staging is needed beyond the header probe that validate_category_headers_batched already
// performs.

#include "fastlanes_page_encoder_common.cuh"

namespace fastlanes_encode_stage {

namespace {

/**
 * @brief Non-template, NATIVE64-specialized batched encode for the int64 GPU path.
 *
 * Identical in shape to the (now-removed) `encode_category_batched<int64_t, FastLanesInt64NativeEncoder>`
 * template specialization. The control flow matches RAW32, but the underlying encoder writes
 * the encoded payload directly into device memory rather than copying the gathered input down
 * to host for CPU packing.
 */
inline void encode_native64_category_batched(
  device_span<EncPage> pages,
  std::vector<EncPage> const& host_pages,
  std::vector<fastlanes_page_category> const& category,
  parquet_fastlanes::FastLanesInt64NativeEncoder& encoder,
  fastlanes_cpu_upload_buffers& upload_buffers,
  cuda::stream_ref stream)
{
  size_t const n = category.size();
  if (n == 0) { return; }

  std::vector<rmm::device_uvector<uint64_t>> gather_buffers;
  gather_buffers.reserve(n);
  std::vector<int64_t*> gather_ptrs(n, nullptr);
  std::vector<uint32_t> gather_counts(n, 0);

  for (size_t i = 0; i < n; ++i) {
    auto const& cat = category[i];
    gather_buffers.emplace_back(cat.num_values, stream);
    gpuGatherSinglePageTyped<uint64_t, encode_block_size>
      <<<1, encode_block_size, 0, stream.get()>>>(
        pages, cat.page_idx, gather_buffers.back().data(), nullptr);
    CUDF_CUDA_TRY(cudaGetLastError());
    gather_ptrs[i]   = reinterpret_cast<int64_t*>(gather_buffers.back().data());
    gather_counts[i] = cat.num_values;
  }

  auto encoded = encoder.encode_pages(gather_ptrs, gather_counts, stream);
  auto& enc_buffers = std::get<0>(encoded);
  auto& enc_ptrs    = std::get<1>(encoded);
  auto& enc_sizes   = std::get<2>(encoded);

  validate_category_headers_batched(category, enc_ptrs, enc_sizes, stream);

  for (size_t i = 0; i < n; ++i) {
    auto const& cat = category[i];
    auto const sz   = enc_sizes[i];

    ensure_fastlanes_page_fits_reserved_size(cat.page_idx, host_pages[cat.page_idx], sz);

    upload_buffers.encoded_buffers.push_back(std::move(enc_buffers[i]));
    upload_buffers.host_upload_ptrs[cat.page_idx]  = enc_ptrs[i];
    upload_buffers.host_upload_sizes[cat.page_idx] = sz;
  }
}

}  // namespace

/**
 * @brief Top-level entry point for NATIVE64 (INT64 GPU-encoded) FastLanes pages.
 *
 * Categorizes the page set, encodes the NATIVE64 subset on GPU via
 * `FastLanesInt64NativeEncoder`, splices the per-page encoded device blobs into the upload
 * buffer, then dispatches the downstream `gpuEncodePageLevels` + `gpuEncodeCpuPages`
 * launchers with the hardcoded `encode_kernel_mask::FASTLANES_DELTA_BINARY` mask. If no
 * NATIVE64 pages are present, the function returns without launching the downstream
 * kernels (matches RAW32's early-exit semantics).
 */
inline void run_fastlanes_native64_encode(
  device_span<EncPage> pages,
  bool write_v2_headers,
  device_span<device_span<uint8_t const>> comp_in,
  device_span<device_span<uint8_t>> comp_out,
  device_span<codec_exec_result> comp_results,
  cuda::stream_ref stream)
{
  auto host_pages = copy_fastlanes_pages_to_host(pages, stream);
  fastlanes_cpu_upload_buffers upload_buffers(pages.size());

  auto const categorized = categorize_fastlanes_pages(host_pages);

  if (categorized.native64_pages.empty()) { return; }

  parquet_fastlanes::FastLanesInt64NativeEncoder encoder;
  encode_native64_category_batched(
    pages, host_pages, categorized.native64_pages, encoder, upload_buffers, stream);

  upload_fastlanes_results_and_launch(
    pages,
    write_v2_headers,
    comp_in,
    comp_out,
    comp_results,
    upload_buffers,
    static_cast<uint32_t>(encode_kernel_mask::FASTLANES_DELTA_BINARY),
    stream);
}

}  // namespace fastlanes_encode_stage
