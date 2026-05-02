/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fastlanes/common.cuh>

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace native64_generated {

[[nodiscard]] std::string cuda_error(cudaError_t status, char const* operation);

}  // namespace native64_generated

#include <cudf/fastlanes/native64_cuda_kernels.inl>
