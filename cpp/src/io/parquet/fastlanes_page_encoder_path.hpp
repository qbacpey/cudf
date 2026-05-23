/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>

namespace cudf::io::parquet::detail::fastlanes_encode_stage {

/**
 * @brief Encode-path selector used by `run_fastlanes_cpu_encode`.
 *
 * The legacy path is the straight-line per-page loop introduced before the page-stage refactor.
 * The categorized path performs a single host-side categorization pass over the page metadata
 * and then dispatches per-page encoding grouped by FastLanes encoding mode.
 *
 * Both paths are required to produce byte-identical encoded output for the same inputs in
 * FL-P3-R2. They diverge in iteration order only; per-page encoded payloads and per-page
 * upload pointers/sizes (indexed by page_idx) are equal across paths.
 *
 * The selector is intended for the FL-P3 test-only A/B harness. Production behavior MUST remain
 * `legacy_per_page` until FL-P3-R3 lands batching changes.
 *
 * The declarations live in a slim non-CUDA header so plain-C++ test translation units can
 * include them without dragging in the parquet/CUDA stack.
 */
enum class fastlanes_encode_path : uint8_t {
  legacy_per_page      = 0,
  categorized_per_page = 1,
};

/**
 * @brief Returns the current FastLanes encode-path selection (per-thread).
 *
 * Defaults to `fastlanes_encode_path::legacy_per_page` so behavior matches the pre-refactor code.
 *
 * Exported with default visibility because the FL-P3 A/B harness lives in a test executable
 * that links against libcudf; without the export the symbol stays internal and the test fails
 * to link.
 */
[[nodiscard]] CUDF_EXPORT fastlanes_encode_path get_encode_path();

/**
 * @brief Overrides the active FastLanes encode-path for the calling thread.
 *
 * Test-only switch used by the FL-P3 A/B parity harness. Production code SHOULD NOT call this
 * directly; the default remains `legacy_per_page` for the lifetime of the program.
 *
 * See `get_encode_path` for visibility/export rationale.
 */
CUDF_EXPORT void set_encode_path(fastlanes_encode_path path);

/**
 * @brief Scoped RAII helper that restores the previous encode-path on destruction.
 *
 * Intended for narrow A/B test scopes so a failing test cannot leak a non-default selection
 * into subsequent tests.
 */
class scoped_encode_path {
 public:
  explicit scoped_encode_path(fastlanes_encode_path path) : _previous{get_encode_path()}
  {
    set_encode_path(path);
  }
  ~scoped_encode_path() { set_encode_path(_previous); }

  scoped_encode_path(scoped_encode_path const&)            = delete;
  scoped_encode_path& operator=(scoped_encode_path const&) = delete;
  scoped_encode_path(scoped_encode_path&&)                 = delete;
  scoped_encode_path& operator=(scoped_encode_path&&)      = delete;

 private:
  fastlanes_encode_path _previous;
};

}  // namespace cudf::io::parquet::detail::fastlanes_encode_stage
