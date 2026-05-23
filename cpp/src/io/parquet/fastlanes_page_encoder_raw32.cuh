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
// RAW32 is a CPU-encoded FastLanes path: the encoder library copies the gathered device input
// down to host, packs on the CPU, and uploads the encoded blob back to device for downstream
// consumption. The shape of the gather + encode + validate + register-upload loop is identical
// to the legacy `encode_category_batched<int32_t, FastLanesInt32Encoder>` template; this file
// pulls it out as a non-template helper so the two FastLanes paths (RAW32 / NATIVE64) can
// diverge independently in future packets without touching one another.

#include "fastlanes_page_encoder_common.cuh"

namespace fastlanes_encode_stage {

namespace {

/**
 * @brief Non-template, RAW32-specialized batched encode for the int32 CPU path.
 *
 * Identical in shape to the (now-removed) `encode_category_batched<int32_t, FastLanesInt32Encoder>`
 * template specialization, kept as an explicit non-template helper so the RAW32 and NATIVE64
 * flows can diverge.
 */
inline void encode_raw32_category_batched(
  device_span<EncPage> pages,
  std::vector<EncPage> const& host_pages,
  std::vector<fastlanes_page_category> const& category,
  parquet_fastlanes::FastLanesInt32Encoder& encoder,
  fastlanes_cpu_upload_buffers& upload_buffers,
  rmm::cuda_stream_view stream)
{
  size_t const n = category.size();
  if (n == 0) { return; }

  std::vector<rmm::device_uvector<uint32_t>> gather_buffers;
  gather_buffers.reserve(n);
  std::vector<int32_t*> gather_ptrs(n, nullptr);
  std::vector<uint32_t> gather_counts(n, 0);

  for (size_t i = 0; i < n; ++i) {
    auto const& cat = category[i];
    gather_buffers.emplace_back(cat.num_values, stream);
    gpuGatherSinglePageTyped<uint32_t, encode_block_size>
      <<<1, encode_block_size, 0, stream.value()>>>(
        pages, cat.page_idx, gather_buffers.back().data(), nullptr);
    gather_ptrs[i]   = reinterpret_cast<int32_t*>(gather_buffers.back().data());
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
 * @brief Top-level entry point for RAW32 (INT32 CPU-encoded) FastLanes pages.
 *
 * Categorizes the page set, encodes the RAW32 subset on CPU via `FastLanesInt32Encoder`,
 * splices the per-page encoded device blobs into the upload buffer, then dispatches the
 * downstream `gpuEncodePageLevels` + `gpuEncodeCpuPages` launchers with the hardcoded
 * `encode_kernel_mask::FASTLANE_BITPACK_RAW` mask. If no RAW32 pages are present, the
 * downstream launch is still issued so that any pre-existing page-state work on the stream
 * settles consistently (matches the legacy unified entry point's behavior when one of the
 * sub-paths was empty).
 */
inline void run_fastlanes_raw32_encode(device_span<EncPage> pages,
                                       bool write_v2_headers,
                                       device_span<device_span<uint8_t const>> comp_in,
                                       device_span<device_span<uint8_t>> comp_out,
                                       device_span<codec_exec_result> comp_results,
                                       rmm::cuda_stream_view stream)
{
  auto host_pages = copy_fastlanes_pages_to_host(pages, stream);
  fastlanes_cpu_upload_buffers upload_buffers(pages.size());

  auto const categorized = categorize_fastlanes_pages(host_pages);

  if (categorized.raw32_pages.empty()) { return; }

  parquet_fastlanes::FastLanesInt32Encoder encoder;
  encode_raw32_category_batched(
    pages, host_pages, categorized.raw32_pages, encoder, upload_buffers, stream);

  upload_fastlanes_results_and_launch(
    pages,
    write_v2_headers,
    comp_in,
    comp_out,
    comp_results,
    upload_buffers,
    static_cast<uint32_t>(encode_kernel_mask::FASTLANE_BITPACK_RAW),
    stream);
}

}  // namespace fastlanes_encode_stage
