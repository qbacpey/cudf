/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Note (FL-P3-R1 follow-up): this source file is intentionally source-included from
// `page_enc.cu` and therefore inherits the enclosing `cudf::io::parquet::detail::` namespace
// plus the local templated kernels defined there (e.g. `gpuGatherSinglePageTyped`,
// `gpuEncodePageLevels`, `gpuEncodeCpuPages`).
//
// FL-P4-R3 retired the unified `run_fastlanes_cpu_encode` entry point. This TU now exists
// solely to pull in the shared FastLanes encode-stage helpers and the two dedicated
// per-encoding entry points (`run_fastlanes_raw32_encode`, `run_fastlanes_native64_encode`),
// each defined inline in its own source-included header.

#include "fastlanes_page_encoder_common.cuh"
#include "fastlanes_page_encoder_raw32.cuh"
#include "fastlanes_page_encoder_native64.cuh"
