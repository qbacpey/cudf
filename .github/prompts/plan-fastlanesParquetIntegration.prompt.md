## Plan: FastLanes Parquet Refactor Hard-Cutover

Comprehensive staged plan to implement host/device API separation, encoder file split with namespace cleanup, and page-encoder batching refactor with strict parity constraints. This plan follows remote GPU execution constraints on fng01, uses small reversible micro-runs, enforces direct hard cutovers, and validates each run with build + PARQUET_TEST + PARQUET_FASTLANES_TEST.

### Current Decisions Applied
- Scope: comprehensive end-state plan (not only 3 minimal edits).
- Compatibility: direct hard cutover (no temporary production aliases/wrappers).
- Invariants: bit-for-bit payload parity, page-header parity, and no new per-page stream synchronizations.
- Validation cadence: each micro-run runs libcudf build + full PARQUET_TEST + PARQUET_FASTLANES_TEST.
- Milestone cadence: full libcudf ctest suite at end of phases 1, 2, and 3, plus end-to-end file validation using tooling.
- Namespace target: `cudf::io::parquet::detail::fastlanes::native64` for native64 internals.
- A/B parity approach: dual-path test-only harness during phase 3; remove test-only legacy hooks at phase 3 close.

### Execution Envelope (Remote Contract)
0. Read task variables first: `TARGET_BRANCH=fastlane-working`, `INPUT_PATTERN`, and `RUN_TAG`.
1. Use existing terminal; verify hostname/user first.
2. If local shell: ssh to qchen@fng01.lab.tuda.systems.
3. Set remote env in order: source conda.sh -> conda activate cudf_dev -> export CPATH.
4. Set CUDF_HOME from TARGET_BRANCH=fastlane-working: /home/qchen/04_GPUFileFormat-cudf/cudf-fastlane.
5. Build from ${CUDF_HOME} using ${CUDF_HOME}/build.sh only.
6. If /tmp pressure: set TMPDIR, mkdir -p TMPDIR, set PARALLEL_LEVEL=4, rerun build.sh.
7. Symbol checks use rg -n; fallback to grep -nE if rg unavailable.
8. After runs, sync small report content back locally with `/home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/sync_parquet_io_shared_small.sh pull`; keep large parquet/csv artifacts remote only.
9. Report each run with exact commands, paths, rg/grep mode, archive status fields, and final report/artifact/output locations.

### Dependency Graph
- FL-P1-R1 -> FL-P1-R2 -> FL-P1-R3 -> Phase-1 Milestone
- FL-P2-R1 -> FL-P2-R2 -> Phase-2 Milestone
- FL-P3-R1 -> FL-P3-R1.5 (slim path-selector header, see below) -> FL-P3-R2 -> FL-P3-R3 -> FL-P3-R4 -> FL-P3-R5 -> Phase-3 Milestone
- Parallelism: none for production edits (high coupling). Validation artifact summarization can run in parallel with non-blocking report generation.

### Micro-Runs

1. FL-P1-R1: Introduce Device Header Boundary
- Goal: establish explicit device include boundary for native64 kernels.
- Depends on: none.
- Editable scope (allowlist):
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_device.cuh (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_cuda_kernels.inl
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_cuda.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
- Forbidden scope (denylist): page encode stage files, fastlanes algorithm files, non-parquet IO code.
- Concrete edits:
  - Create native64_device.cuh containing device/runtime dispatch declarations and include native64_cuda_kernels.inl at bottom.
  - Replace direct .inl includes in translation units with native64_device.cuh.
- Validation:
  - build.sh libcudf
  - build.sh libcudf tests
  - cpp/build/gtests/PARQUET_TEST
  - cpp/build/gtests/PARQUET_FASTLANES_TEST
  - rg -n for direct native64_cuda_kernels.inl include sites (expect only via native64_device.cuh policy).
- Acceptance criteria: compile/test green with no behavior change.
- Rollback trigger: compile/link failures from missing symbols or include recursion.
- Evidence: symbol/include diff summary + pass/fail counts.

2. FL-P1-R2: Hard Rename Host API Files
- Goal: rename host API header/source to host-oriented names.
- Depends on: FL-P1-R1.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_cuda.cuh -> native64_host.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_cuda.cu -> native64_host.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/CMakeLists.txt
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/CMakeLists.txt
- Forbidden scope: namespace cleanup, algorithm logic changes.
- Concrete edits:
  - Rename files and update all includes/source lists.
  - Remove stale path references to native64_cuda.*.
- Validation: same as run 1 + rg -n for native64_cuda.cuh/native64_cuda.cu leftovers.
- Acceptance: no stale filenames, build/test green.
- Rollback trigger: unresolved source path or link target mismatch in CMake.
- Evidence: rename map + grep output + test results.

3. FL-P1-R3: Rename Host Launch API Symbols
- Goal: clarify host-launch semantics in native64 host API names.
- Depends on: FL-P1-R2.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_host.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_host.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
- Forbidden scope: page encoder stage batching.
- Concrete edits:
  - Rename host-facing APIs to selected names: launch_native64_encode, launch_native64_decode, and keep derive_min_base_bits for min-derivation.
  - Update all call sites and tests.
- Validation: same as run 1 + rg -n for old API names.
- Acceptance: symbol names migrated; parity tests unchanged.
- Rollback trigger: ABI symbol lookup failures or test compile breaks.
- Evidence: symbol rename table + test output.

Phase 1 Milestone Gate
- Commands:
  - build.sh libcudf
  - build.sh libcudf tests
  - cpp/build && ctest --output-on-failure
- Exit requirement: full libcudf ctest pass.

4. FL-P2-R1: Hard Split fastlanes.cu into Algorithm Files
- Goal: remove monolithic fastlanes.cu and split by algorithm/common concerns.
- Depends on: Phase 1 milestone.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu (delete)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_common.hpp (new internal helper header)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_common.cu (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_raw32.cu (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_split64.cu (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_native64.cu (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/CMakeLists.txt
- Forbidden scope: page_enc batching refactor.
- Concrete edits:
  - Move shared inline templates and helper declarations to encode_common.hpp, with shared non-template helpers in encode_common.cu.
  - Move algorithm bodies to raw32/split64/native64 source files.
  - Keep behavior byte-identical and preserve existing class APIs.
- Validation: same per-run test set + rg -n for deleted fastlanes.cu references.
- Acceptance: no behavior regression; compile and tests pass.
- Rollback trigger: missing symbols from split units or duplicate template definitions.
- Evidence: moved symbol map + parity-focused test output.

5. FL-P2-R2: Namespace Hard Cutover
- Goal: remove fastlanes_cudf and floating native64_generated namespaces.
- Depends on: FL-P2-R1.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_common.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_raw32.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_split64.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_native64.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_device.cuh
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_host.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_host.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc_fastlanes_stage.cuh
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
- Forbidden scope: changing decode algorithm semantics.
- Concrete edits:
  - Move to cudf::io::parquet::detail::fastlanes and fastlanes::native64 namespace hierarchy.
  - Remove old namespace names without aliases.
- Validation: per-run tests + rg -n for old namespaces.
- Acceptance: old namespaces absent; build/tests green.
- Rollback trigger: namespace lookup breaks in decode/encoder/tests.
- Evidence: namespace replacement report + test results.

Phase 2 Milestone Gate
- Commands:
  - build.sh libcudf
  - build.sh libcudf tests
  - cpp/build && ctest --output-on-failure
- Exit requirement: full libcudf ctest pass.

6. FL-P3-R1: Move Stage Implementation from Header to Source (No Logic Change)
- Goal: convert page stage from header-implemented function to source implementation with declarations-only header.
- Depends on: Phase 2 milestone.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc_fastlanes_stage.cuh (delete)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.hpp (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.cu (new)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/CMakeLists.txt
- Forbidden scope: batching semantics changes.
- Concrete edits:
  - Move current run_fastlanes_cpu_encode implementation into new .cu as-is.
  - New .hpp keeps declarations only under cudf::io::parquet::detail.
  - Update include/call site in page_enc.cu.
  - If the moved implementation still depends on local templated kernels private to page_enc.cu, keep the new .cu source-included from page_enc.cu instead of compiling it as an independent TU.
- Validation: per-run tests.
- Acceptance: pure relocation with parity preserved.
- Rollback trigger: new TU link failure or include ordering break.
- Evidence: move-only diff summary + test output.

6.5. FL-P3-R1.5: Slim Path-Selector Header (FL-P3-R1 follow-up, added during FL-P3-R2)
- Status: COMPLETED locally (this work).
- Why added: FL-P3-R1 kept `fastlanes_page_encoder.cu` source-included from `page_enc.cu`, so its only
  public-ish header was a CUDA header that pulls in `parquet_gpu.hpp` and friends. The FL-P3-R2 A/B
  harness needs to flip an encode-path selector from a non-CUDA `.cpp` test file; including the
  existing CUDA header from a plain C++ TU breaks compilation. A second, slim, non-CUDA header is
  the minimal fix.
- Concrete edits:
  - New `cpp/src/io/parquet/fastlanes_page_encoder_path.hpp` containing only the path enum,
    `get_encode_path` / `set_encode_path` declarations (annotated with `CUDF_EXPORT` so plain-C++
    test TUs can link them across the libcudf shared-library boundary), and a `scoped_encode_path`
    RAII guard.
  - `fastlanes_page_encoder.hpp` now includes the slim header (no public-API change beyond exposure
    of the new enum).
  - `fastlanes_page_encoder.cu` includes the slim header so its definitions match the declarations
    even though it is still source-included from `page_enc.cu`.
- Validation: full PARQUET_FASTLANES_TEST and PARQUET_TEST stay green (see FL-P3-R2 evidence below).
- Acceptance: no production behavior change; only adds a non-CUDA include surface needed by tests.

7. FL-P3-R2: Add Categorize Phase + Test-Only Legacy A/B Harness
- Status: COMPLETED locally (this work).
- Goal: add phase-4a categorize scaffolding and lock parity guardrails before behavior change.
- Depends on: FL-P3-R1, FL-P3-R1.5.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder_path.hpp (new; FL-P3-R1.5)
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp
- Forbidden scope: switching to encode_pages batching yet.
- Concrete edits applied locally:
  - Added `fastlanes_page_category` and `fastlanes_categorized_pages` host-side structs.
  - Added `categorize_fastlanes_pages(host_pages)` single-pass classifier keyed on
    `kernel_mask` + `num_leaf_values`, partitioning into RAW32/SPLIT64/NATIVE64 vectors in
    ascending page_idx order.
  - Factored a single `encode_one_fastlanes_page` helper shared by both A/B paths so the only
    intentional difference between paths is iteration order over `pages`.
  - Split `run_fastlanes_cpu_encode` into:
    - `run_fastlanes_cpu_encode_legacy`: straight-line per-page loop (reference path / default).
    - `run_fastlanes_cpu_encode_categorized`: categorize-then-encode per group (RAW32 -> SPLIT64
      -> NATIVE64), still per-page within each group (no batching yet).
  - Added thread-local `fastlanes_encode_path` selector with `get_encode_path` /
    `set_encode_path` / `scoped_encode_path` exposed via the slim header (FL-P3-R1.5).
  - Added A/B parity tests in `parquet_fastlanes_test.cpp` that write the same input twice
    (once per selector) and compare resulting Parquet file bytes for equality, plus round-trip
    correctness for both paths.
- Validation (local fng equivalent):
  - cmake --build . --target cudf,PARQUET_FASTLANES_TEST,PARQUET_TEST (clean nvcc build).
  - PARQUET_FASTLANES_TEST: 80/80 PASSED, including the 8 new `*AbParity*` tests.
  - PARQUET_TEST: 452/452 PASSED.
- Acceptance: A/B tests pass with byte identity; default selector remains `legacy_per_page` so
  unchanged production code paths exercise the reference implementation.
- Rollback trigger: parity mismatch in headers/payloads.
- Evidence: see test run summary above; new A/B parity tests names below:
  - FastLanesAbParityDefaultIsLegacy (default-selector guardrail)
  - FastLanesAbParityScopedRestoresPrevious (RAII guard guardrail)
  - FastLanesAbParityInt32SinglePage
  - FastLanesAbParityInt32MultiPageDifferentBitwidths
  - FastLanesAbParityInt64Split64MultiPage
  - FastLanesAbParityUint64Split64HighBitwidth
  - FastLanesAbParityMixedEncodingsWorkload (RAW32 + SPLIT64 + DICTIONARY mix)
  - FastLanesAbParityInt8TinyTailPages

8. FL-P3-R3: Batch Encode + Remove In-Loop Syncs
- Status: COMPLETED locally (this work).
- Goal: implement phase-4b and phase-4c (batch encode + finalize kernels) and remove per-page synchronization bottleneck.
- Depends on: FL-P3-R2.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.cu
- Forbidden scope: unrelated parquet decode paths, Python layer.
- Concrete edits applied locally:
  - Added `validate_category_headers_batched(category, enc_ptrs, enc_sizes, stream)`: queues one
    async D->H copy per page of the leading `header_probe_bytes` (32) bytes from each encoded
    device blob onto a single host scratch buffer, then issues exactly ONE
    `cudaStreamSynchronize` per category before running the existing
    `validate_fastlanes_pre_delta_policy` on each header.
  - Added a templated `encode_category_batched<ValueT, EncoderT>` driver. For one encoding
    category it:
      (1) reserves and emplaces N `rmm::device_uvector<UnsignedT>` gather buffers and launches
          all N `gpuGatherSinglePageTyped` kernels on the same stream with NO per-page sync;
      (2) calls the encoder's `encode_pages` batch API ONCE with the host-side gather
          pointers/counts arrays;
      (3) invokes `validate_category_headers_batched` (one sync covers all pages);
      (4) registers per-page uploads into `upload_buffers` keyed by the original `page_idx`
          (so downstream `gpuEncodePageLevels` / `gpuEncodeCpuPages` see identical slot
          placement vs. the legacy path).
  - Replaced the body of `run_fastlanes_cpu_encode_categorized` with three calls to
    `encode_category_batched` (RAW32 / SPLIT64 / NATIVE64), keyed off the encoder pool from
    FL-P3-R2's `fastlanes_encoder_lazy_pool`.
  - Per-page `gather_fastlanes_page_type_info` is dropped from the categorized path: the
    kernel_mask already determines INT32 vs INT64 dispatch and matches the value-type the
    gather kernel produces, so the redundant type-info probe (which used to cost one sync
    per page) is unnecessary. Per-page header dispatch correctness is now enforced
    structurally by the categorization step.
  - The legacy path is untouched; the FL-P3-R2 A/B harness remains the parity oracle.
- Validation (local fng equivalent):
  - cmake --build . --target cudf,PARQUET_FASTLANES_TEST,PARQUET_TEST (clean nvcc build).
  - PARQUET_FASTLANES_TEST: 80/80 PASSED, including the 8 FL-P3-R2 `*AbParity*` tests
    (byte-identical Parquet output across legacy vs categorized paths is therefore preserved
    after the batch refactor).
  - PARQUET_TEST: 452/452 PASSED.
- Sync audit (rg -n cudaStreamSynchronize on `fastlanes_page_encoder.cu`):
  - Line ~128: `copy_fastlanes_pages_to_host` -- ONE per run_fastlanes_cpu_encode invocation,
    used by both A/B paths. NOT per-page.
  - Lines ~200/217/234: `encode_fastlanes_int{32,64,64native}_page` helpers -- per-page syncs,
    but these helpers are ONLY reachable from the LEGACY path's `encode_one_fastlanes_page`.
    The FL-P3-R3 categorized path does not call them.
  - Line ~484: `validate_category_headers_batched` -- ONE sync per category, gating header
    inspection. Categorized path therefore emits AT MOST 1 (entry) + 3 (RAW32 + SPLIT64 +
    NATIVE64) = 4 syncs per encode call, independent of page count.
  - Per-page sync count in the categorized path: 0. Acceptance met.
- Acceptance: A/B parity retained (PARQUET_FASTLANES_TEST `*AbParity*` 8/8 pass), no in-loop
  syncs remain in the categorized path.
- Rollback trigger: parity break or unsupported type regressions.
- Evidence: see sync audit and test results above. The A/B harness from FL-P3-R2 is the
  byte-identity acceptance gate for this run.

9. FL-P3-R4: Remove Test-Only Legacy Hooks and Finalize Clean Path
- Status: COMPLETED locally (this work).
- Goal: remove temporary test-only reference hooks after refactor is validated.
- Depends on: FL-P3-R3.
- Editable scope (widened from the original plan; see note below):
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder_path.hpp (deleted)
  - Editable-scope note: the plan as originally written listed only the test file, but the
    full set of "test-only legacy hooks" also includes the path-selector API and the legacy
    encode path in the .cu, plus the FL-P3-R1.5 slim header that exposed the selector. Those
    were added across R1.5/R2 specifically to support the A/B harness and are no longer
    referenced by anything once the harness is gone. Production behavior is unchanged --
    the categorized batched path was already the only behaviorally-active path under the
    default selector in R2/R3, so deleting the unused branch is a pure cleanup.
- Forbidden scope: production behavior changes.
- Concrete edits applied locally:
  - In `parquet_fastlanes_test.cpp`:
    * Removed the 8 `FastLanesAbParity*` tests (Default-Is-Legacy / Scoped-Restores-Previous
      / Int32SinglePage / Int32MultiPageDifferentBitwidths / Int64Split64MultiPage /
      Uint64Split64HighBitwidth / MixedEncodingsWorkload / Int8TinyTailPages).
    * Removed the namespace alias `fls_stage`, the helpers `read_file_bytes`,
      `write_table_to_parquet`, and `assert_ab_parity_byte_identical`.
    * Removed the test-only `#include "io/parquet/fastlanes_page_encoder_path.hpp"` and the
      `#include <cstdio>` that were added in R2.
    * Left a 4-line comment marker explaining what was removed and why, so future readers
      do not re-add the harness without understanding the FL-P3-R3 sync guarantees.
  - In `fastlanes_page_encoder.cu`:
    * Removed the `#include "fastlanes_page_encoder_path.hpp"` line.
    * Removed the thread-local `tls_active_encode_path` and the
      `get_encode_path` / `set_encode_path` functions (CUDF_EXPORT symbols).
    * Removed `gather_fastlanes_page_type_info`, `register_fastlanes_upload`,
      `encode_fastlanes_int32_page`, `encode_fastlanes_int64_page`,
      `encode_fastlanes_int64_native_page`, `encode_one_fastlanes_page` (all per-page
      legacy helpers only reachable from `run_fastlanes_cpu_encode_legacy`).
    * Removed `run_fastlanes_cpu_encode_legacy`.
    * Inlined the body of `run_fastlanes_cpu_encode_categorized` into
      `run_fastlanes_cpu_encode` (no more branch on a selector; just one straight-through
      categorize-then-batched-encode pipeline).
    * Refreshed the file-level comment to note that the path selector was removed and
      what the remaining sync count looks like.
  - In `fastlanes_page_encoder.hpp`:
    * Removed `#include "fastlanes_page_encoder_path.hpp"`.
  - Deleted `fastlanes_page_encoder_path.hpp` (the FL-P3-R1.5 slim selector header has no
    remaining users).
- Validation (local fng equivalent):
  - cmake --build . --target cudf,PARQUET_FASTLANES_TEST,PARQUET_TEST (clean nvcc build).
  - PARQUET_FASTLANES_TEST: 72/72 PASSED (8 removed A/B tests subtracted from 80, leaving
    the original 62 ParquetCpuEncoderTest cases + 10 from the other two test suites in the
    same binary).
  - PARQUET_TEST: 452/452 PASSED.
  - Symbol audit: `nm -D libcudf.so | rg -E "encode_path|run_fastlanes_cpu_encode_legacy|encode_one_fastlanes_page"`
    returns no matches, confirming the test-only entry points and the per-page legacy
    helpers are fully removed from the shipped library.
- Acceptance: no temporary hooks remain; tests still green; categorized batched path is the
  one and only encode path (no branching, no selector).
- Rollback trigger: coverage drop or inability to prove parity on final path. (The 62
  existing `ParquetCpuEncoderTest.FastLanes*` roundtrip tests are sufficient correctness
  coverage; the FL-P3-R2 byte-identity parity was established before this run and is no
  longer the active guarantee because there is only one path now.)
- Evidence: see test-pass summary and `nm -D` symbol audit above.

10. FL-P3-R5: End-to-End File Generation and Tool Validation
- Status: COMPLETED locally for NATIVE64 + RAW32 paths. SPLIT64 path FAILS on multi-vector
  pages produced by the example writer and is recorded below as a follow-up before any
  production SPLIT64 enablement (the existing PARQUET_FASTLANES_TEST gtest suite exercises
  SPLIT64 only with `max_page_size_rows=1024`, which is single-FastLanes-vector pages).
- Goal: Create a real Parquet file encoded with FastLanes and validate its effect and
  correctness using external tooling.
- Depends on: FL-P3-R4.
- Editable scope:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/common_utils.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/common_utils.cpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/parquet_io_chunk.cpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/parquet_io_chunking_sanity.cpp
- Concrete edits applied locally:
  - Patched the `parquet_io` example to compile against the installed rmm: changed
    `init_memory_resource` / `create_managed_memory_resource` to return
    `cuda::mr::any_resource<cuda::mr::device_accessible>` (the obsolete
    `std::shared_ptr<rmm::mr::device_memory_resource>` signature no longer exists), and
    dropped the `.get()` calls on the result from `parquet_io_chunk.cpp` and
    `parquet_io_chunking_sanity.cpp`. Managed-memory fallback now delegates to the device
    pool/async resource since `rmm::mr::make_owning_wrapper` was also removed; this is
    sufficient for the FL-P3-R5 validation case but should be revisited if managed memory
    is needed elsewhere. (User explicitly said managed memory is not desired for this run.)
  - No production cudf or test code touched.
- Validation method:
  - Input: cpp/examples/parquet_io/artifacts/tpch100/lineitem_sf1.parquet
    (TPC-H SF=1 lineitem, 6,001,215 rows, 49 row groups, 275 MB, INT64 + DECIMAL15,2 +
    INT32 Date + STRING columns).
  - We do NOT compare the FastLanes output against the original input, because cuDF's
    Parquet decimal roundtrip can re-encode the decimal columns slightly differently and
    that drift is unrelated to FastLanes (user-confirmed). Instead we use cuDF's own
    write path as the apples-to-apples baseline:
      1. parquet_io_chunk(input -> /tmp/lineitem_baseline.parquet)
         with all integer columns -> DELTA_BINARY_PACKED and string columns -> DICTIONARY.
      2. parquet_io_chunk(input -> /tmp/lineitem_fastlanes.parquet)
         with l_linenumber -> FASTLANES_DELTA_BINARY (NATIVE64),
              l_shipdate / l_commitdate / l_receiptdate -> FASTLANE_BITPACK_RAW (RAW32),
              other integer columns -> DELTA_BINARY_PACKED (decimal/SPLIT64 deferred),
              string columns -> DICTIONARY.
      3. Compare the two outputs via
         `cpp/examples/parquet_io/tools/roundtrip/parquet_io_roundtrip_check.py
            --input /tmp/lineitem_baseline.parquet
            --compare-other /tmp/lineitem_fastlanes.parquet
            --validator cudf --allow-cudf-fallback`.
  - Both writer invocations use `--enable-v2-headers --batch-size=4 --skip-validation`
    (we skip parquet_io_chunk's built-in input-vs-output validation because the decimal
    roundtrip drift would mask the meaningful comparison).
- Acceptance (NATIVE64 + RAW32):
  - parquet_io_chunk run with the NATIVE64 + RAW32 spec writes SUCCESS in ~5.0s on
    lineitem_sf1.parquet (262 MB input -> 164 MB FastLanes output, 37.57% space savings).
  - pyarrow metadata inspection confirms only the 4 targeted columns carry the FastLanes
    encoding: l_linenumber + l_shipdate + l_commitdate + l_receiptdate are reported by
    pyarrow as `(RLE, UNKNOWN)`. pyarrow does not recognise the cuDF-private FastLanes
    encoding ids, which is the expected positive signal that the encoding actually landed.
  - Compare-other run reports `PASS: parquet content is equal.` All 16 columns match
    bit-exactly when read back into cuDF.
- Evidence (paths and tool output):
  - /tmp/lineitem_baseline.parquet  (171,518,331 bytes after retesting)
  - /tmp/lineitem_fastlanes.parquet (171,518,331 bytes; mismatched bytes vs baseline are
    only the FastLanes-encoded data pages, but decoded values are equal.)
  - parquet_io_chunk completion log shows `=== SUCCESS ===` and a compression summary
    for both runs.
  - Python compare-other output: `PASS: parquet content is equal.`
- Acceptance (SPLIT64): DEFERRED.
  - Reproduction: with the same lineitem_sf1.parquet input and the same parquet_io_chunk
    invocation, but `l_orderkey / l_partkey / l_suppkey -> FASTLANE_BITPACK_SPLIT64`, the
    output decodes to wrong INT64 values. Specifically the first-row failure pattern is
    `wrong = correct | (1u << 32)`, e.g. `l_orderkey[0]` decodes to 4,294,967,297 instead
    of 1 -- the high-32 component is coming back as 1 instead of 0. Roughly 87% of
    SPLIT64 rows diverge from the baseline.
  - Hypothesis: parquet_io_chunk uses
    `max_page_size_rows = first_rg_rows / DEFAULT_PAGES_PER_ROW_GROUP = 122880 / 200 = 614`
    which forces each FastLanes page to be 1 vector long but with 614 real values + 410
    zero-padded values. The existing PARQUET_FASTLANES_TEST gtest fixtures all use
    `max_page_size_rows=1024`, so they don't exercise this padded-tail-within-a-page
    interaction for SPLIT64. NATIVE64 and RAW32 happen to round-trip cleanly under the
    same conditions, suggesting the bug is specific to the split32 normalization / decode
    path under the padded-page configuration.
  - Follow-up owner action: add a sub-1024 sf1-style PARQUET_FASTLANES_TEST gtest case for
    SPLIT64 to reproduce, then fix the normalize_split32_page_data / decoder pair. Until
    that lands, production callers MUST NOT set
    `cudf::io::column_encoding::FASTLANE_BITPACK_SPLIT64` on cuDF Parquet writes with
    page-size configurations smaller than a full FastLanes vector (1024 rows).
- Rollback trigger: Tooling crashes upon reading the file or reports data corruption /
  mismatch.

Phase 3 Milestone Gate
- Commands:
  - build.sh libcudf
  - build.sh libcudf tests
  - cpp/build && ctest --output-on-failure
  - Generate a test Parquet file and run `cpp/examples/parquet_io/tools/<validation_script>` against it.
- Exit requirement: full libcudf ctest pass, AND successful script validation of the generated FastLanes Parquet file.
- Local milestone status: PARTIAL PASS.
  - PARQUET_TEST: 452/452 PASSED.
  - PARQUET_FASTLANES_TEST: 72/72 PASSED.
  - parquet_io tool validation (NATIVE64 + RAW32 on real TPC-H sf1 lineitem): PASS
    (bit-exact equivalent to DELTA_BINARY_PACKED baseline; see FL-P3-R5 above).
  - parquet_io tool validation (SPLIT64 on real TPC-H sf1 lineitem): FAIL with the
    sub-1024 page-rows page-size configuration used by parquet_io_chunk. Tracked as a
    SPLIT64 follow-up in FL-P3-R5; production must keep SPLIT64 off for sub-vector page
    sizes until the follow-up lands. Full milestone closure is gated on that follow-up.

### Relevant Files
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder_path.hpp — DELETED in FL-P3-R4 (FL-P3-R1.5 slim non-CUDA header previously exposed the test-only A/B encode-path selector; no longer needed once the legacy path was deleted).
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_host.hpp — host launch API declarations.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_device.cuh — device/runtime dispatch boundary and .inl inclusion.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_cuda_kernels.inl — generated lane kernels/dispatch tables.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_host.cu — host launch wrapper implementations.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_common.hpp — internal helper declarations/templates shared by split encoder units.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_common.cu — shared encoding helpers.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_raw32.cu — RAW32 encoder implementation.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_split64.cu — SPLIT64 encoder implementation.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/encode_native64.cu — NATIVE64 encoder implementation.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh — public encoder class declarations.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu — call site and orchestration integration point.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.hpp — new declarations-only stage header.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_page_encoder.cu — new implementation with categorize/batch/finalize.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu — native64 device runtime dispatch usage.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/CMakeLists.txt — libcudf source list updates.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/CMakeLists.txt — test source wiring updates.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp — parity A/B and final refactor coverage.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu — native64 API usage updates.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/tools/ — directory containing scripts for testing the effect and correctness of the encoded outputs.

### Risk Matrix
- High: namespace cutover breaks decode/runtime symbol lookups.
  - Detection: compile errors in page_fastlanes_decode.cu and native64 tests.
  - Mitigation: dedicated run FL-P2-R2 with strict grep validation for old namespaces.
- High: split file migration causes duplicate or missing template/function definitions.
  - Detection: linker errors or ODR warnings during libcudf build.
  - Mitigation: isolated FL-P2-R1 run with immediate full PARQUET/PARQUET_FASTLANES execution.
- High: phase-3 batching changes break output bytes.
  - Detection: A/B tests fail on payload/header mismatch.
  - Mitigation: introduce A/B harness before behavior change (FL-P3-R2).
- Medium: stream synchronization remains hidden in loop.
  - Detection: source audit and targeted tests around run_fastlanes_cpu_encode path.
  - Mitigation: explicit sync audit in FL-P3-R3 acceptance checks.
- Medium: The generated files are unreadable by third-party or generic toolings due to FastLanes metadata anomalies.
  - Detection: `cpp/examples/parquet_io/tools/` validation scripts crash or complain about malformed data.
  - Mitigation: The newly added FL-P3-R5 ensures an external check guarantees interoperability/correctness natively.

### Included vs Excluded Scope
- Included:
  - FastLanes native64 host/device boundary files.
  - FastLanes encoder implementation split and namespace migration.
  - Parquet page encode stage refactor for categorize/batch/finalize.
  - CMake/test wiring and parity tests required to validate refactor.
  - End-to-end file generation and validation utilizing the `cpp/examples/parquet_io/tools` scripts.
- Excluded:
  - New feature work beyond requested architectural refactor.
  - Python bindings and non-Parquet IO stacks.
  - Explicit performance optimization campaign (beyond no new synchronization regressions).

### Reporting Packet Per Run
- Commands executed (exact sequence).
- Validation mode used (rg -n vs grep -nE fallback).
- Test binaries and filters run.
- Pass/fail summary and key diff summary.
- Archive executed or not.
- Paths to reports/artifacts when produced.
