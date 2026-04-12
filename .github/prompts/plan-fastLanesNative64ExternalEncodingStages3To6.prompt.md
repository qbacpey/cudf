## Plan: FastLanes Native64 External-Encoding Stages 3-6

Refine the original staged roadmap to match actual Commit 1-2 outcomes and your constraints: keep GPU-first pointer APIs from Commit 2, represent native64 layout using external parquet encoding FASTLANES_DELTA_BINARY (not a new internal fastlanes payload enum), reserve one component stream for native64, and keep Commit 6 optional optimization-only.

**Steps**
1. Freeze Commit 1-2 as baseline contracts (depends on completed work)
- Keep include-level API boundary as device-pointer host launch APIs plus device runtime lane dispatch helpers.
- Keep host-vector wrappers in test files only.
- Keep backward-compatible alias names while steering new callsites to explicit device_ptr names.

2. Commit 3: GPU Min/Base_Bits Reduction and Metadata Extraction (depends on 1)
- Scope:
  - Move INT64 min/base_bits derivation from host loops to GPU reduction path.
  - Keep split32/split64 wire behavior unchanged in this commit.
- Main edits:
  - page encode path computes min/base_bits metadata on GPU and copies only compact metadata to host.
  - split32 page header serialization continues unchanged semantically.
  - tests updated/extended to validate parity between GPU-derived metadata and previous behavior.
- Safety boundary:
  - No FASTLANES_DELTA_BINARY writer selection yet.
  - No reader dispatch changes yet.

3. Commit 4: Guarded Native64 Writer Path Using FASTLANES_DELTA_BINARY (depends on 2)
- Scope:
  - Introduce guarded native64 writer selection for INT64 pages.
  - Native64 pages are represented by external Encoding FASTLANES_DELTA_BINARY.
  - Split32/split64 fallback remains default-safe path.
- Main edits:
  - writer kernel-mask and external parquet encoding mapping recognizes FASTLANES_DELTA_BINARY for native64 output.
  - native64 guard check decides native64 encode vs split fallback.
  - native64 stream reservation uses one component stream.
- Safety boundary:
  - Do not introduce new internal fastlanes payload-mode enum/type.
  - Preserve existing split64 and raw encoding behavior.

4. Commit 5: Reader Classification and Native64 Decode Dispatch (depends on 4)
- Scope:
  - Reader classifies FASTLANES_DELTA_BINARY as native64 layout and dispatches native64 decode path.
  - Existing split64 and raw decode paths remain unchanged for existing files.
- Main edits:
  - page header/classification maps FASTLANES_DELTA_BINARY to fastlanes decode routing.
  - decode validation enforces INT64-only constraints for native64 path.
  - native64 decode kernel path selected where encoding indicates native64.
- Safety boundary:
  - Maintain full backward compatibility for split64 historical files.

5. Commit 6 (optional): Native64 Fusion and Performance Polish (depends on 5)
- Scope:
  - Optimization-only changes after correctness is green.
- Main edits:
  - kernel fusion and decode/encode pipeline polish limited to native64 codepath.
  - no format changes, no enum changes, no behavior changes.
- Safety boundary:
  - drop commit entirely if perf gain is not measurable or adds instability.

6. Verification contract after each commit (parallelizable with report collection, but build/test order blocks)
- Remote setup sequence:
  - local hostname/whoami check, SSH to fng01 if local.
  - source conda.sh, conda activate cudf_dev, export CPATH, export CUDF_HOME for fastlane-working.
- Build sequence:
  - build.sh libcudf, then build.sh libcudf tests.
  - if /tmp pressure: export TMPDIR, mkdir -p TMPDIR, export PARALLEL_LEVEL=4, rerun.
- Core gates every commit:
  - PARQUET_FASTLANES_TEST generated suite.
  - PARQUET_FASTLANES_TEST bw37 suite.
  - PARQUET_FASTLANES_TEST split32/raw compatibility filter.
- Commit-specific gates:
  - Commit 4 adds writer filter using resolved writer gtest binary rule.
  - Commit 5 and Commit 6 add PARQUET_TEST reader filter.
- Reporting contract:
  - print exact commands, final input path, final report/artifact paths, archive flag, rg/grep fallback, and writer gtest binary when used.

7. Updated prompt packet (depends on 2-6 plan finalization)
- Reframe objective around external-encoding native64 rollout:
  - GPU-native pointer APIs already in place.
  - commit roadmap now focuses on metadata reduction, guarded writer integration, reader routing, and optional perf.
- Require explicit include/deny scope per commit and rollback trigger per commit.
- Require test evidence per commit with remote contract reporting fields.

**Relevant files**
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/fastlanes_encode.cuh — commit 3 metadata reduction, commit 4 writer guard/native64 encode path.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu — writer kernel-mask selection, external encoding mapping, stream reservation, page encoding flow.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu — writer integration and runtime support checks.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp — encode/decode mask enums and support helpers for FASTLANES_DELTA_BINARY routing.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/io/parquet_schema.hpp — external parquet encoding enum usage consistency.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/io/types.hpp — column encoding API alignment for FASTLANES_DELTA_BINARY.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu — commit 5 page classification to reader decode masks.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu — commit 5 native64 decode dispatch and validation gates.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl_preprocess_utils.cu — reader preprocess integration touchpoint.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp — split32/raw compatibility and writer/reader regression tests.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu — parity suites and commit 6 optimization guard checks.
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu — bw37 guard suite.

**Verification**
1. Commit 3
- build.sh libcudf
- build.sh libcudf tests
- PARQUET_FASTLANES_TEST with native64 generated + bw37 + split32/raw compatibility filters.
- Acceptance: parity unchanged and split32/raw semantics unchanged.
- Rollback trigger: signed-edge metadata mismatch or roundtrip mismatch.

2. Commit 4
- Commit 3 gate plus writer tests using resolved writer binary (PARQUET_WRITER_TEST if present else PARQUET_TEST).
- Acceptance: native64 writer path guarded and split fallback correct.
- Rollback trigger: native64 pages unreadable or fallback bypass.

3. Commit 5
- Commit 4 gate plus PARQUET_TEST reader filter.
- Acceptance: native64 decode green, split64 compatibility preserved.
- Rollback trigger: layout misclassification or regressions on existing split64 files.

4. Commit 6 optional
- Commit 5 full correctness rerun plus benchmark evidence.
- Acceptance: measurable gain with zero correctness regression.
- Rollback trigger: neutral/regressive perf or instability.

**Decisions**
- Use external parquet encoding FASTLANES_DELTA_BINARY to represent native64 layout; do not introduce internal fastlanes layout-type enum.
- Native64 stream reservation: one stream.
- Commit 6 remains optional optimization-only.
- Commit 1-2 adjusted APIs remain baseline: explicit device pointer host APIs and device runtime lane dispatch helpers.

**Further Considerations**
1. FASTLANES_DELTA_BINARY is currently partially wired in writer/reader paths in this repo; Commit 4 must complete writer mapping and Commit 5 must complete reader classification to avoid falling back to raw/general paths.
2. Keep backward-compatible aliases in include APIs until Commit 5 lands, then evaluate alias retirement in a later cleanup-only commit.
3. Avoid mixing format/wire changes and performance work in the same commit to maintain rollback clarity.