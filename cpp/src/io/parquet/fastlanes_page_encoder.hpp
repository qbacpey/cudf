/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fastlanes_parquet_common.cuh"

namespace cudf::io::parquet::detail::fastlanes_encode_stage {

void run_fastlanes_cpu_encode(device_span<EncPage> pages,
                              bool write_v2_headers,
                              device_span<device_span<uint8_t const>> comp_in,
                              device_span<device_span<uint8_t>> comp_out,
                              device_span<cudf::io::detail::codec_exec_result> comp_results,
                              uint32_t fastlanes_kernel_mask_bits,
                              rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet::detail::fastlanes_encode_stage
