## Plan: FastLanes Native64 TODO Refactor

Refactor scope is TODO/FIXME plus nearby awkward/debug code in the same touched functions. Strategy is a hard-cut to per-encoding decode masks, separate public decode entrypoints, and a two-step removal of ExternalPageMode to minimize behavioral risk while eliminating redundancy.

**Steps**
1. R0 Baseline lock and TODO status map. Build a source-of-truth list of active TODO/FIXME and adjacent debt in touched areas, and mark stale TODO anchors from older drafts as resolved/deleted. Depends on none.
2. R1 Hard-cut decode-mask wiring to per-encoding bits. Replace legacy FASTLANES_BINARY usage with explicit FASTLANE_BITPACK_RAW, FASTLANE_BITPACK_SPLIT64, FASTLANES_DELTA_BINARY in page classification and launch gating. Update kernel_mask_for_page and all decode gate checks to avoid aggregate alias. Depends on 1.
3. R2 Split public decode APIs by encoding and remove combined facade usage. Introduce explicit raw32/split64/native64 public decode launch functions and call them directly from reader dispatch; remove direct reader dependence on combined decode facade. Depends on 2.
4. R3 ExternalPageMode replacement (phase 1, compatibility-safe internals). Introduce encoding+physical-type validation helpers in header validation paths and migrate call sites to them while keeping behavior equivalent. Depends on 2; parallel with 5 where no overlap.
5. R4 ExternalPageMode removal (phase 2). Delete ExternalPageMode enum and wrappers, remove mode-parameter APIs, and fully migrate empty-page/header validation helpers to encoding-driven contracts. Depends on 4.
6. R5 Encoder cleanup from active TODOs and nearby debt. Deduplicate bitwidth helper variants, simplify empty-page creation APIs to encoding-driven calls, and normalize debug_print behavior across scalar32/split64/native64 encode helpers. Depends on 4; parallel with 6 only where files do not overlap.
7. R6 Debug boundary consolidation (keep capability, reduce duplication). Centralize debug gates/logging and keep one explicit decode debug boundary; remove duplicated ad-hoc logging paths while preserving runtime toggles. Depends on 3 and 5.
8. R7 Final integration verification and cleanup. Run focused tests each run, full ctest at milestone gates, and final grep-based invariants for removed abstractions/symbols. Depends on 2-7.

**Dependency graph**
- Critical path: R0 -> R1 -> R2 -> R3 -> R4 -> R7.
- Secondary path: R3 -> R5 -> R6 -> R7.
- Parallel window: R5 may proceed during/after R4 if file ownership is split cleanly; otherwise serialize after R4.

**Per-run scope boundaries and validation**
1. R0
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc_fastlanes_stage.cuh, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl.cpp, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh.
- Denylist: non-parquet IO modules, unrelated cudf algorithms, Python packages.
- Validation: grep inventory for TODO/FIXME and symbol baseline; no behavior change.
2. R1
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl.cpp, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu.
- Denylist: encode-path algorithms and common header semantics.
- Validation: focused decode tests + compile; grep invariant that decode_kernel_mask::FASTLANES_BINARY no longer exists.
3. R2
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl.cpp.
- Denylist: page header serialization internals and encode helpers.
- Validation: PARQUET_FASTLANES_TEST focused filters; milestone full ctest gate #1.
4. R3
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc_fastlanes_stage.cuh.
- Denylist: reader orchestration and mask mapping files (already stabilized in R1/R2).
- Validation: header metadata tests in PARQUET_FASTLANES_TEST and targeted writer encoding tests.
5. R4
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp.
- Denylist: unrelated parquet tests and non-fastlanes readers.
- Validation: no remaining ExternalPageMode references; milestone full ctest gate #2.
6. R5
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp.
- Denylist: decode routing and reader orchestration.
- Validation: focused encode/decode parity tests and writer user-requested-encoding test.
7. R6
- Allowlist: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc_fastlanes_stage.cuh, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl.cpp, /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp.
- Denylist: core header encoding semantics.
- Validation: debug-off behavior equivalence and debug-on smoke checks.
8. R7
- Allowlist: touched files only for final hygiene.
- Denylist: feature expansions (sentinel dtype fix, broader INT64/UINT64 activation).
- Validation: focused tests each run, final milestone full ctest gate #3, grep invariants.

**Relevant files**
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu — active TODOs at compute_bitwidth and create_empty_result; native64 encode helper and debug_print parity cleanup.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh — ExternalPageMode removal path and encoding-driven header validation/pre-delta defaults.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp — decode_kernel_mask hard-cut and new public per-encoding decode entrypoints.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu — kernel_mask_for_page mapping to per-encoding mask bits.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl.cpp — direct per-encoding dispatch orchestration and debug boundary call-site.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu — split launch APIs, setup validation migration off mode enum, debug kernel boundary.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc_fastlanes_stage.cuh — centralized encode staging helpers and debug/logging dedup.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu — top-level encode orchestration entrypoint using staging helper.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp — header validity and pre-delta contract tests currently tied to mode wrappers.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_writer_test.cpp — UserRequestedEncodings regression checks.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/CMakeLists.txt — PARQUET_FASTLANES_TEST composition and affected native64 test sources.

**Verification**
1. Focused per-run checks: PARQUET_FASTLANES_TEST native64 suites and PARQUET_TEST writer filter for UserRequestedEncodings.
2. Milestone gates after R2, R4, and R7: full ctest from cpp/build with output-on-failure.
3. Build policy: use build.sh from CUDF_HOME (libcudf, then libcudf tests) in remote GPU environment.
4. Invariant checks:
- no decode_kernel_mask::FASTLANES_BINARY symbol remains after R1;
- no ExternalPageMode symbol remains after R4;
- no behavior regressions in pre-delta/header validity tests and native64 decode parity tests.

**Decisions**
- Scope: TODO/FIXME plus adjacent awkward/debug code in touched functions only.
- Remove ExternalPageMode and rely on Encoding + Type, but in two phases for safer migration.
- Decode mask transition is hard-cut to per-encoding bits (no temporary aggregate alias).
- Public decode API should be separated per encoding; reader calls per-encoding entrypoints directly.
- Debug capabilities remain, but hooks are centralized/de-duplicated.
- Validation cadence is focused tests per run with milestone full ctest gates.
- Sentinel dtype-mismatch and broader INT64/UINT64 activation stay out of scope.

**Further Considerations**
1. If R1 reveals broad compile churn from hard-cut mask removal, enforce strictly ordered landing: page_hdr.cu then reader_impl.cpp then page_fastlanes_decode.cu in one run to avoid intermediate breakage.
2. For R4, keep test updates in the same run as ExternalPageMode deletion to avoid transient red tests tied to deprecated bool wrapper overloads.
3. If debug consolidation alters logging format, treat text output format as non-contractual and validate only gating behavior and absence of decode side effects.