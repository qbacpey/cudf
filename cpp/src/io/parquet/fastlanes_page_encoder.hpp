/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fastlanes_parquet_common.cuh"

namespace cudf::io::parquet::detail::fastlanes_encode_stage {

// FL-P4-R3 replaced the previous unified `run_fastlanes_cpu_encode` entry point with two
// dedicated entry points (`run_fastlanes_raw32_encode` and `run_fastlanes_native64_encode`).
// The actual definitions are inline in the per-path source-included headers
// (`fastlanes_page_encoder_raw32.cuh` and `fastlanes_page_encoder_native64.cuh`).

void run_fastlanes_raw32_encode(device_span<EncPage> pages,
                                bool write_v2_headers,
                                device_span<device_span<uint8_t const>> comp_in,
                                device_span<device_span<uint8_t>> comp_out,
                                device_span<cudf::io::detail::codec_exec_result> comp_results,
                                cuda::stream_ref stream);

void run_fastlanes_native64_encode(device_span<EncPage> pages,
                                   bool write_v2_headers,
                                   device_span<device_span<uint8_t const>> comp_in,
                                   device_span<device_span<uint8_t>> comp_out,
                                   device_span<cudf::io::detail::codec_exec_result> comp_results,
                                   cuda::stream_ref stream);

}  // namespace cudf::io::parquet::detail::fastlanes_encode_stage
