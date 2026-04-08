```text
BEGIN AGENT REMOTE CONTRACT

I am working from a local machine without GPU. You can assume `lsyncd` is running. You must execute GPU build/test on remote server.

Task variables you must read first:
- TARGET_BRANCH: fastlane-working
- INPUT_PATTERN: (example: (base) qchen@fng01:~/04_GPUFileFormat-cudf/large_input$ l
0003-cudf-DELTA.parquet tpch1-SNAPPY-cudf.parquet  tpch1-nocomp-cudf.parquet)
- RUN_TAG: short tag for this run (example: rt_20260404_a)

Execution policy:
1. Use my current terminal whenever possible. Do not open extra terminals unless blocked.
2. First verify whether you are already in SSH environment by checking hostname/user.
3. If not on remote, connect with: ssh `qchen@fng01.lab.tuda.systems`
4. In remote shell, set environment in this order:
   `source ~/miniconda3/etc/profile.d/conda.sh`
   `conda activate cudf_dev`
   `export CPATH="$CONDA_PREFIX/include/rapids:$CONDA_PREFIX/include"`
5. Do not run set -u before conda activate.
6. For validation text search, use `rg -n` when available. If `rg` is unavailable, use
   `grep -nE` with the same pattern and file list.
7. For long remote commands with nested single quotes, prefer
   `cat <<'EOF' | ssh ... bash -s` to avoid quote-break issues.

Branch mapping:
1. If TARGET_BRANCH=main:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/GPUFileFormat-cudf`
2. If TARGET_BRANCH=fastlane:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane`
3. If TARGET_BRANCH=nvcomp:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-nvcomp`
4. If TARGET_BRANCH=fastlane-working:
   `export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane`

Roundtrip data roots:
- `export ROUNDTRIP_INPUT_ROOT=/home/qchen/04_GPUFileFormat-cudf/large_input`
- `export PARQUET_IO_SHARED_ROOT=/home/qchen/04_GPUFileFormat-cudf/parquet_io_shared`

Build rule:
1. Always build from ${CUDF_HOME}.
2. Always use ${CUDF_HOME}/build.sh (never direct cmake).
3. If build fails due `/tmp` pressure on fng01, set:
   `export TMPDIR=/home/qchen/04_GPUFileFormat-cudf/tmpbuild`
   `mkdir -p "$TMPDIR"`
   `export PARALLEL_LEVEL=4`
   then rerun `${CUDF_HOME}/build.sh` from `${CUDF_HOME}`.

Writer test binary rule:
1. Resolve writer gtest binary before writer filters:
   `if [ -x "${CUDF_HOME}/cpp/build/gtests/PARQUET_WRITER_TEST" ]; then export WRITER_GTEST_BIN="${CUDF_HOME}/cpp/build/gtests/PARQUET_WRITER_TEST"; else export WRITER_GTEST_BIN="${CUDF_HOME}/cpp/build/gtests/PARQUET_TEST"; fi`
2. Use `${WRITER_GTEST_BIN}` for `ParquetWriterTest.*` filters.

Roundtrip run rule:
1. Read inputs from ${ROUNDTRIP_INPUT_ROOT}/${INPUT_PATTERN}.
2. Write active outputs to shared area under:
   - ${PARQUET_IO_SHARED_ROOT}/reports/$(basename ${CUDF_HOME})
   - ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})
3. Do not rely on symlinks for runtime outputs.

Post-run sync rule:
1. Sync small report content back to local:
   /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/sync_parquet_io_shared_small.sh pull
2. Keep large parquet/csv files remote only.

Reporting rule:
1. Print exact commands run.
2. Print final input path used.
3. Print final report/artifact/output locations.
4. Print whether archive step was executed.
5. Print whether `rg -n` or `grep -nE` fallback was used.
6. If writer tests were run, print resolved `${WRITER_GTEST_BIN}`.

END AGENT REMOTE CONTRACT
```


## Dependency Graph
1. R01 -> R02, R03
2. R03 -> R04
3. R04 -> R05, R06
4. R02, R05, R06 -> R07
5. R07 -> R08 -> R09 -> R10
6. R07, R10 -> R11 -> R12 -> R13 -> R14

## Scope Aliases
1. S1 Enum surface:
- cpp/include/cudf/io/parquet_schema.hpp
- cpp/include/cudf/io/types.hpp
- cpp/src/io/parquet/reader_impl_preprocess_utils.cu
- cpp/examples/parquet_io/common_utils.cpp
2. S2 Reader routing:
- cpp/src/io/parquet/parquet_gpu.hpp
- cpp/src/io/parquet/page_hdr.cu
3. S3 Writer guards:
- cpp/src/io/parquet/writer_impl.cu
- cpp/tests/io/parquet_writer_test.cpp
4. S4 Runtime selector:
- cpp/src/io/parquet/page_enc.cu
- cpp/src/io/parquet/parquet_gpu.hpp
5. S5 Page encoding assignment:
- cpp/src/io/parquet/page_enc.cu
- cpp/src/io/parquet/parquet_gpu.hpp
6. S6 Reservation math:
- cpp/src/io/parquet/page_enc.cu
7. S7 Header refactor:
- cpp/include/cudf/fastlanes/common.cuh
- cpp/src/io/parquet/page_fastlanes_decode.cu
- cpp/include/cudf/fastlanes/debug.hpp
- cpp/src/io/parquet/page_enc.cu
- cpp/tests/io/parquet_fastlanes_test.cpp
8. S8 Rename surface alias phase: same files as S1
9. S9 Rename core pipeline:
- cpp/src/io/parquet/writer_impl.cu
- cpp/src/io/parquet/page_enc.cu
- cpp/src/io/parquet/page_hdr.cu
- cpp/src/io/parquet/parquet_gpu.hpp
- cpp/src/io/parquet/reader_impl_preprocess_utils.cu
10. S10 Finalize rename and count safety:
- cpp/include/cudf/io/parquet_schema.hpp
- cpp/src/io/parquet/writer_impl.cu
- cpp/src/io/parquet/reader_impl_preprocess_utils.cu
11. S11 PRE_DELTA header field:
- cpp/include/cudf/fastlanes/common.cuh
- cpp/include/cudf/fastlanes/debug.hpp
- cpp/tests/io/parquet_fastlanes_test.cpp
12. S12 Encode PRE_DELTA policy:
- cpp/include/cudf/fastlanes/fastlanes_encode.cuh
- cpp/src/io/parquet/page_enc.cu
- cpp/tests/io/parquet_fastlanes_test.cpp
13. S13 Decode PRE_DELTA policy:
- cpp/src/io/parquet/page_fastlanes_decode.cu
- cpp/include/cudf/fastlanes/common.cuh
- cpp/tests/io/parquet_fastlanes_test.cpp
14. S14 Final matrix and tooling:
- cpp/tests/io/parquet_fastlanes_test.cppaaads
- cpp/tests/io/parquet_writer_test.cpp
- cpp/examples/parquet_io/common_utils.cpp
- cpp/src/io/parquet/reader_impl_preprocess_utils.cu

## Global Rules For Every Run
- Use the AGENT REMOTE CONTRACT exactly as provided.
- Read TARGET_BRANCH, INPUT_PATTERN, RUN_TAG first.
- If TARGET_BRANCH=fastlane-working, set CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane.
- For validation text search, use `rg -n` when available; otherwise use `grep -nE` with the same regex and file list.
- Before any `ParquetWriterTest` invocation, resolve and use `${WRITER_GTEST_BIN}` from the contract rule.
- If fng01 build fails from `/tmp` pressure, apply `TMPDIR` and reduced `PARALLEL_LEVEL` fallback, then rerun.
- Forbidden files for each run: everything outside that run's editable scope.
- Stop after validation and report. Do not merge next-run work.

## Prompt R01
Run ID: R01_enum_surface_split64
Suggested RUN_TAG: fl_r01_enum_surface
Dependency: none
Editable scope: S1 only
Task:
- Introduce FASTLANE_BITPACK_SPLIT64 enum/name surface only.
- No writer/reader behavior changes.
Validation required:
- rg -n "FASTLANE_BITPACK_SPLIT64" cpp/include/cudf/io/parquet_schema.hpp cpp/include/cudf/io/types.hpp cpp/src/io/parquet/reader_impl_preprocess_utils.cu cpp/examples/parquet_io/common_utils.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf
Acceptance:
- New symbol appears in intended surface files only.
- Build succeeds.

## Prompt R02
Run ID: R02_reader_accept_route_split64
Suggested RUN_TAG: fl_r02_reader_split64
Dependency: R01 complete
Editable scope: S2 only
Task:
- Reader accepts split64 encoding and routes it to FastLanes decode path.
Validation required:
- rg -n "FASTLANE_BITPACK_SPLIT64" cpp/src/io/parquet/parquet_gpu.hpp cpp/src/io/parquet/page_hdr.cu
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf
Acceptance:
- split64 is accepted and routed.
- Existing encodings unchanged.

## Prompt R03
Run ID: R03_writer_guard_matrix
Suggested RUN_TAG: fl_r03_writer_guard
Dependency: R01 complete
Editable scope: S3 only
Task:
- FASTLANES/FASTLANE_BITPACK rejects INT64 physical.
- FASTLANE_BITPACK_SPLIT64 accepts INT64 only.
- Add targeted writer tests in scope.
Validation required:
- rg -n "FASTLANE_BITPACK_SPLIT64|FASTLANES_BITPACK|INT64" cpp/src/io/parquet/writer_impl.cu cpp/tests/io/parquet_writer_test.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- Resolve `${WRITER_GTEST_BIN}` using contract rule
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings
Acceptance:
- Guard matrix enforced.
- Targeted writer test passes.

## Prompt R04
Run ID: R04_runtime_selector_split
Suggested RUN_TAG: fl_r04_runtime_selector
Dependency: R03 complete
Editable scope: S4 only
Task:
- Update `data_encoding_for_col` selector to distinguish FASTLANES_BITPACK (raw mode) vs
   FASTLANE_BITPACK_SPLIT64 (split64 mode), each with explicit fallback behavior.
- Ensure selector logic in S4 does not depend on `NUM_ENCODINGS` sentinel ordering and has no stale old-name assumptions.
Validation required:
- rg -n "data_encoding_for_col|is_fastlanes_bitpack_raw_runtime_supported|is_fastlanes_bitpack_split64_runtime_supported|FASTLANE_BITPACK_SPLIT64|FASTLANES_BITPACK|FASTLANE_BITPACK_SPLIT64 = \(1 << 8\)" cpp/src/io/parquet/page_enc.cu cpp/src/io/parquet/parquet_gpu.hpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf
Acceptance:
- Selector resolves per request mode with explicit fallbacks.

## Prompt R05
Run ID: R05_page_encoding_assignment
Suggested RUN_TAG: fl_r05_page_tagging
Dependency: R04 complete
Editable scope: S5 only
Task:
- Set `page.encoding` per page according to selected fastlanes mode.
- Raw mode pages must tag `FASTLANES_BITPACK`; split64 mode pages must tag `FASTLANE_BITPACK_SPLIT64`.
- Remove hardcoded single fastlanes page encoding assignment.
- Ensure for all the pages in a column chunk, the encoding is consistent and correctly reflects the mode, to align with the fastlane convention.
Validation required:
- rg -n "page\.encoding|fastlanes_encoding_for_mask|is_fastlanes_bitpack_mask|FASTLANE_BITPACK_SPLIT64|FASTLANES_BITPACK" cpp/src/io/parquet/page_enc.cu
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf
Acceptance:
- Per-page encoding tags reflect selected mode.

## Prompt R06
Run ID: R06_reservation_math_modes
Suggested RUN_TAG: fl_r06_reservation
Dependency: R04 complete
Editable scope: S6 only
Task:
- Implement mode-specific `max_data_size` reservation for raw vs split64 paths.
- Include split64 component-stream reservation math and keep under-allocation impossible.
Validation required:
- rg -n "max_data_size|reserved|component_streams|fastlanes_body_size|fastlanes_component_streams_for_mask|fastlanes_bitpack_kernel_masks" cpp/src/io/parquet/page_enc.cu
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf
Acceptance:
- Safe reservations for both modes.

R04-R06 current implementation baseline (fastlane-working):
- Dedicated split64 kernel bit exists in parquet gpu mask enum (`FASTLANE_BITPACK_SPLIT64 = (1 << 8)`).
- Selector split is implemented via `is_fastlanes_bitpack_raw_runtime_supported` and
   `is_fastlanes_bitpack_split64_runtime_supported` with explicit PLAIN fallback.
- FastLanes page tagging is mode-derived via `fastlanes_encoding_for_mask(...)` (not hardcoded).
- Reservation uses `fastlanes_component_streams_for_mask(...)` and treats both FastLanes masks via
   `is_fastlanes_bitpack_mask(...)`.
- `EncodePages` keeps one shared FastLanes execution block for either mode using
   `fastlanes_bitpack_kernel_masks` stream accounting.

## Prompt R07
Run ID: R07_header_refactor_no_layout_mode
Suggested RUN_TAG: fl_r07_header_refactor
Dependency: R02, R05, R06 complete
Editable scope: S7 only
Task:
- Remove layout signaling from FastLanes header mode contract.
- Validate against external page encoding mode.
- Update debug and in-scope tests.
Validation required:
- rg -n "is_valid_for_external_mode|is_valid_for_physical|layout_mode|bitwidth_mode|legacy_layout_mode|legacy_bitwidth_mode" cpp/include/cudf/fastlanes/common.cuh cpp/src/io/parquet/page_fastlanes_decode.cu cpp/include/cudf/fastlanes/debug.hpp cpp/tests/io/parquet_fastlanes_test.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesHeaderScalar32MetadataRoundTrip:ParquetCpuEncoderTest.FastLanesHeaderSplit32MetadataRoundTrip:ParquetCpuEncoderTest.FastLanesHeaderRejectsMalformedSplit32Metadata
Acceptance:
- Header no longer uses internal layout flags for mode identity.

## Prompt R08
Run ID: R08_rename_surface_raw_alias
Suggested RUN_TAG: fl_r08_surface_raw
Dependency: R07 complete
Editable scope: S8 only
Task:
- Introduce FASTLANE_BITPACK_RAW naming at API surface.
- Keep temporary compatibility alias.
Validation required:
- rg -n "FASTLANE_BITPACK_RAW|FASTLANES_BITPACK" cpp/include/cudf/io/parquet_schema.hpp cpp/include/cudf/io/types.hpp cpp/src/io/parquet/reader_impl_preprocess_utils.cu cpp/examples/parquet_io/common_utils.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf
Acceptance:
- RAW symbol available and compiles.

## Prompt R09
Run ID: R09_rename_core_pipeline
Suggested RUN_TAG: fl_r09_core_rename
Dependency: R08 complete
Editable scope: S9 only
Task:
- Mechanical rename from BITPACK naming to RAW naming in core parquet pipeline scope.
Validation required:
- rg -n "FASTLANE_BITPACK_RAW|FASTLANES_BITPACK" cpp/src/io/parquet/writer_impl.cu cpp/src/io/parquet/page_enc.cu cpp/src/io/parquet/page_hdr.cu cpp/src/io/parquet/parquet_gpu.hpp cpp/src/io/parquet/reader_impl_preprocess_utils.cu
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- Resolve `${WRITER_GTEST_BIN}` using contract rule
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings
Acceptance:
- Core pipeline rename compiles and writer smoke test passes.

R07-R09 current implementation baseline (fastlane-working):
- `is_valid_for_external_mode(...)` is the primary FastLanes header mode validator.
- `is_valid_for_physical(...)` compatibility wrapper has been removed.
- Debug now reports only effective mode fields from external page encoding (legacy debug header flags removed).
- Core parquet pipeline now uses RAW naming (`FASTLANE_BITPACK_RAW`) in writer/encoder/decoder routing paths.
- API surface no longer keeps `FASTLANES_BITPACK` compatibility aliases.

## Prompt R10
Run ID: R10_finalize_rename_num_enc
Suggested RUN_TAG: fl_r10_finalize_rename
Dependency: R09 complete
Editable scope: S10 only
Task:
- Finalize rename transition. Remove all the code for keeping compatibility for legacy FASTLANE encodings.
- Ensure NUM_ENCODINGS-related iteration remains correct.
Validation required:
- rg -n "NUM_ENCODINGS|update_chunk_encodings|FASTLANE_BITPACK_RAW|FASTLANES_BITPACK" cpp/include/cudf/io/parquet_schema.hpp cpp/src/io/parquet/writer_impl.cu cpp/src/io/parquet/reader_impl_preprocess_utils.cu
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- Resolve `${WRITER_GTEST_BIN}` using contract rule
- Test all Parquet Related tests
Acceptance:
- No stale old-name dependency in core paths.

R10 focused follow-up status (fastlane-working):
- `PARQUET_FASTLANES_TEST` INT64/UINT64 forced-bitpack coverage now uses split64 semantics (`FASTLANE_BITPACK_SPLIT64`) for requested/expected 64-bit fastlanes mode.
- Typed fastlanes test helpers now derive requested writer encoding from expected parquet encoding (RAW vs SPLIT64), keeping assertions aligned with R03/R10 guard matrix.
- Remote validation result: `${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST` passed (`58/58`).

R10 legacy cleanup follow-up status (fastlane-working):
- Removed legacy debug-only fields (`legacy_layout_mode`, `legacy_bitwidth_mode`) from `PageDebugInfo` and decode debug population.
- Removed obsolete fastlanes compatibility APIs (`PageHeader::read`, `PageHeader::get_payload_ptr`, and `is_valid_for_physical(...)`).
- Removed old encoding alias from public API surface: `column_encoding::FASTLANES_BITPACK`.
- Updated tests/examples/tooling to use explicit `FASTLANE_BITPACK_RAW` (INT32) and `FASTLANE_BITPACK_SPLIT64` (INT64/UINT64) requests.
- Remote validation result: full `ctest --output-on-failure` passed (`114/114`), including parquet targets (`PARQUET_TEST`, `PARQUET_FASTLANES_TEST`, `PARQUET_DELETION_VECTORS_TEST`, `STREAM_IO_PARQUET_TEST`).

## Prompt R11
Run ID: R11_header_add_pre_delta
Suggested RUN_TAG: fl_r11_pre_delta_header
Dependency: R07 and R10 complete
Editable scope: S11 only
Task:
- Add PRE_DELTA field to header serialization/deserialization and debug/test support.
- No policy behavior changes yet.
Validation required:
- rg -n "PRE_DELTA|pre_delta" cpp/include/cudf/fastlanes/common.cuh cpp/include/cudf/fastlanes/debug.hpp cpp/tests/io/parquet_fastlanes_test.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesHeaderScalar32MetadataRoundTrip:ParquetCpuEncoderTest.FastLanesHeaderSplit32MetadataRoundTrip
Acceptance:
- PRE_DELTA header field roundtrips.

## Prompt R12
Run ID: R12_encode_pre_delta_policy
Suggested RUN_TAG: fl_r12_pre_delta_encode
Dependency: R11 complete
Editable scope: S12 only
Task:
- Encode PRE_DELTA policy:
  - RAW/SPLIT64 default PRE_DELTA=true.
  - DELTA forces PRE_DELTA=false.
  - DELTA+PRE_DELTA=true rejected.
- Add tests:
  - ParquetCpuEncoderTest.FastLanesDeltaRejectsPreDelta
  - ParquetCpuEncoderTest.FastLanesRawSplit64DefaultPreDelta
Validation required:
- rg -n "PRE_DELTA|pre_delta" cpp/include/cudf/fastlanes/fastlanes_encode.cuh cpp/src/io/parquet/page_enc.cu cpp/tests/io/parquet_fastlanes_test.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesDeltaRejectsPreDelta:ParquetCpuEncoderTest.FastLanesRawSplit64DefaultPreDelta
Acceptance:
- Encode policy enforced and tests pass.

## Prompt R13
Run ID: R13_decode_pre_delta_enforcement
Suggested RUN_TAG: fl_r13_pre_delta_decode
Dependency: R11 and R12 complete
Editable scope: S13 only
Task:
- Decode enforces mode+PRE_DELTA legality.
- Reject DELTA+PRE_DELTA invalid combination early.
- Add tests:
  - ParquetCpuEncoderTest.FastLanesDecodeRejectsDeltaPreDeltaCombo
  - ParquetCpuEncoderTest.FastLanesSplit64DecodeHonorsPreDelta
Validation required:
- rg -n "PRE_DELTA|pre_delta|INVALID_DATA_TYPE|UNSUPPORTED_ENCODING" cpp/src/io/parquet/page_fastlanes_decode.cu cpp/include/cudf/fastlanes/common.cuh cpp/tests/io/parquet_fastlanes_test.cpp
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesDecodeRejectsDeltaPreDeltaCombo:ParquetCpuEncoderTest.FastLanesSplit64DecodeHonorsPreDelta
Acceptance:
- Decode legality checks enforced and tests pass.

## Prompt R14
Run ID: R14_final_matrix_and_tooling
Suggested RUN_TAG: fl_r14_final_matrix
Dependency: R13 complete
Editable scope: S14 only
Task:
- Finalize matrix and tooling:
  - DELTA+PRE_DELTA rejection coverage.
  - Updated writer request/fallback matrix coverage.
  - Updated visible encoding-name helpers.
Validation required:
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST
- Resolve `${WRITER_GTEST_BIN}` using contract rule
- ${WRITER_GTEST_BIN}
- Optional roundtrip check if INPUT_PATTERN resolves to at least one file.
Acceptance:
- Both gtest targets pass.
- Final naming/policy expectations reflected in tests/tool output.

## Reporting Template (Required For Every Run)
- Commands run (exact order).
- Files changed.
- Validation results per command.
- Final input path used (or N/A).
- Final report/artifact/output paths.
- Archive step executed: yes/no.

I will build this as a planning-only simplification review: first I am going to inspect the current FastLanes header/encode/decode/test surfaces, then produce phased micro-runs R15-R18 with remote-compatible validation commands and rollback points. I will start with a quick context sweep and symbol inventory before drafting the final plan.I am checking existing memory notes first so the new plan can reuse prior remote workflow constraints and avoid duplicate session notes.

Read memory

Read memory

Read memory

I have the memory index; next I’ll read the repository-specific FastLanes notes to align this simplification plan with prior guardrail decisions before I inspect code symbols.

Read memory [](file:///home/qba/.vscode-server/data/User/workspaceStorage/7938f0ef122d35883c8cb443df714727/GitHub.copilot-chat/memory-tool/memories/repo/fastlanes_guard_matrix_notes.md)

Read memory [](file:///home/qba/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/cudf-remote-workflow.md)

I’m moving into discovery now by running parallel codebase exploration: one pass for header/mode symbols in FastLanes core and one pass for encode/decode/test call sites, so the removal plan is evidence-based and safely phased.

Header symbol inventory

Encode decode test mapping

Searched for files matching `**/cpp/include/cudf/fastlanes/*.cuh`, 3 matches

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#1-1), lines 1 to 500

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#1-1), lines 1 to 100

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#1-1), lines 1 to 100

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#1-1), lines 1 to 100

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#500-500), lines 500 to 611

Searched for regex `SPLIT64|PRE_DELTA|RAW` (`**/cpp/src/io/parquet/page_enc.cu`), 48 results

Searched for regex `SPLIT64|PRE_DELTA|RAW` (`**/cpp/src/io/parquet/page_fastlanes_decode.cu`), 20 results

Searched for regex `SPLIT64|PRE_DELTA|RAW` (`**/cpp/tests/io/parquet_fastlanes_test.cpp`), 50 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#500-500), lines 500 to 650

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#3970-3970), lines 3970 to 4070

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#110-110), lines 110 to 250

Searched for regex `PageLayoutMode|SCALAR32|SPLIT32|NATIVE64` (`**/cpp/src/**`), 18 results

Searched for regex `PageLayoutMode|SCALAR32|SPLIT32|NATIVE64` (`**/cpp/tests/**`), 20 results

Searched for regex `BitwidthMode|BitwidthMode::SINGLE|SPLIT_COMPONENTS` (`**/cpp/src/**`), 3 results

Searched for regex `BitwidthMode|BitwidthMode::SINGLE|SPLIT_COMPONENTS` (`**/cpp/tests/**`), 9 results

Searched for regex `ExternalPageMode|RAW|DELTA` (`**/cpp/src/**`), 20 results

Searched for regex `ExternalPageMode|RAW|DELTA` (`**/cpp/tests/**`), 20 results

Searched for regex `TEST_F.*FastLanes|test.*split64|test.*pre_delta` (`**/cpp/tests/io/parquet_fastlanes_test.cpp`), 58 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#670-670), lines 670 to 850

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#850-850), lines 850 to 1290

Searched for regex `TypeCastMode|SIGNED_SAFE|SIGNED_REINTERPRET` (`**/cpp/src/**`), no results

Searched for regex `TypeCastMode|SIGNED_SAFE|SIGNED_REINTERPRET` (`**/cpp/tests/**`), no results

Searched for regex `pre_delta|pre-delta` (`**/cpp/src/**`), 14 results

Searched for regex `pre_delta|pre-delta` (`**/cpp/tests/**`), 14 results

Searched for regex `serialize_scalar32|serialize_split32` (`**/cpp/src/**`), no results

Searched for regex `serialize_scalar32|serialize_split32` (`**/cpp/tests/**`), 9 results

Searched for files matching `**/fastlanes/common.cuh`, 1 match

Searched for files matching `**/fastlanes/debug.hpp`, 1 match

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#1-1), lines 1 to 150

Searched for regex `is_valid_cast_mode|is_valid_bitwidth|is_valid_layout_mode` (`**/cpp/src/**`), no results

Searched for regex `is_valid_cast_mode|is_valid_bitwidth|is_valid_layout_mode` (`**/cpp/tests/**`), no results

Searched for regex `is_scalar32_layout|is_split32_layout|has_valid_raw_mode_metadata` (`**/cpp/src/**`), no results

Searched for regex `is_scalar32_layout|is_split32_layout|has_valid_raw_mode_metadata` (`**/cpp/tests/**`), 4 results

Searched for regex `component_bitwidth_low|component_bitwidth_high|min_value_low_bits` (`**/cpp/src/**`), 15 results

Searched for regex `component_bitwidth_low|component_bitwidth_high|min_value_low_bits` (`**/cpp/tests/**`), 7 results

Searched for regex `PageHeader|ExternalPageMode|PageLayoutMode|BitwidthMode` (`**/cpp/include/cudf/fastlanes/common.cuh`), 46 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#190-190), lines 190 to 611

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#570-570), lines 570 to 610

Searched for regex `int32_to_u32_bits|u32_bits_to_int32|int64_to_u64_bits|u64_bits_to_int64` (`**/cpp/src/**`), 2 results

Searched for regex `int32_to_u32_bits|u32_bits_to_int32|int64_to_u64_bits|u64_bits_to_int64` (`**/cpp/tests/**`), 2 results

Searched for regex `is_valid_for_external_mode|deserialize|payload_ptr` (`**/cpp/src/**`), 16 results

Searched for regex `is_valid_for_external_mode|deserialize|payload_ptr` (`**/cpp/tests/**`), 20 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#250-250), lines 250 to 400

Searched for regex `default_pre_delta_for_mode|expected_body_size_bytes|min_value_bits` (`**/cpp/src/**`), 4 results

Searched for regex `default_pre_delta_for_mode|expected_body_size_bytes|min_value_bits` (`**/cpp/tests/**`), 3 results

Searched for regex `has_valid_common_metadata|has_valid_split64` (`**/cpp/src/**`), no results

Searched for regex `has_valid_common_metadata|has_valid_split64` (`**/cpp/tests/**`), no results

Searched for regex `common\.cuh|fastlanes` (`**/cpp/include/**`), 20 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#1-1), lines 1 to 150

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#200-200), lines 200 to 250

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#250-250), lines 250 to 280

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#3980-3980), lines 3980 to 4080

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#4040-4040), lines 4040 to 4080

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#670-670), lines 670 to 750

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#750-750), lines 750 to 850

Searched for regex `header_size|padded_size|encoded_size_bytes|max_bitwidth` (`**/cpp/src/**`), 20 results

Searched for regex `header_size|padded_size|encoded_size_bytes|max_bitwidth` (`**/cpp/tests/**`), 14 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#1-1), lines 1 to 50

Read memory

Created memory file [](file:///home/qba/.vscode-server/data/User/workspaceStorage/7938f0ef122d35883c8cb443df714727/GitHub.copilot-chat/memory-tool/memories/Y2FkNzYyNTItMjBiYS00OTZkLThjZDgtMjFhZmQ0ZGE0NDFk/fastlanes_inspection_findings.md)

Created memory file [](file:///home/qba/.vscode-server/data/User/workspaceStorage/7938f0ef122d35883c8cb443df714727/GitHub.copilot-chat/memory-tool/memories/Y2FkNzYyNTItMjBiYS00OTZkLThjZDgtMjFhZmQ0ZGE0NDFk/fastlanes_header_analysis.md)

Parallel discovery completed and already confirms the core direction: external Parquet encoding is authoritative, while several internal mode tags are legacy. I still need to read the first subagent’s full symbol inventory file, then I’ll synthesize the final R15–R18 simplification plan.

Read [](file:///home/qba/.vscode-server/data/User/workspaceStorage/7938f0ef122d35883c8cb443df714727/GitHub.copilot-chat/chat-session-resources/cad76252-20ba-496d-8cd8-21afd4da441d/call_kephp6zszzd4bo1t0Bl2pgia__vscode-1775455800002/content.txt)

I’m now verifying the subagent findings directly in the anchored files to ensure the plan references only symbols that truly exist in your current tree and reflects the latest changes.

Searched for regex `enum class (TypeCastMode|PageLayoutMode|BitwidthMode|ExternalPageMode)|is_valid_for_external_mode|is_pre_delta_valid_for_mode|default_pre_delta_for_mode|serialize_scalar32|serialize_split32|is_scalar32_layout|is_split32_layout|has_valid_common_metadata|expected_body_size_bytes` (`**/cpp/include/cudf/fastlanes/common.cuh`), 20 results

Searched for regex `ExternalPageMode|is_valid_for_external_mode|is_pre_delta_valid_for_mode|FASTLANE_BITPACK_RAW|FASTLANE_BITPACK_SPLIT64|page.encoding|physical_type` (`**/cpp/src/io/parquet/page_fastlanes_decode.cu`), 20 results

Searched for regex `PageHeader::deserialize|is_pre_delta_valid_for_mode|ExternalPageMode|FASTLANE_BITPACK_RAW|FASTLANE_BITPACK_SPLIT64|component_bitwidth|layout_mode|bitwidth_mode|pre_delta` (`**/cpp/src/io/parquet/page_enc.cu`), 20 results

Searched for regex `default_pre_delta_for_mode|is_valid_for_external_mode|OFFSET_PRE_DELTA|layout_mode|bitwidth_mode|serialize_scalar32|serialize_split32|SPLIT64|INT64|UINT64` (`**/cpp/tests/io/parquet_fastlanes_test.cpp`), 20 results

I’ve confirmed the key symbol footprints in the current tree; next I’ll inspect the active prompt file’s run structure so the new R15–R18 plan drops in consistently with your existing micro-run style.

Searched for regex `^## R1[0-9]|^### R1[0-9]|R11|R12|R13|R14|Acceptance|Validation|Editable scope` (`**/.github/prompts/plan-fastlanesExplicitParquetEncodingsMicroRuns.prompt.md`), 69 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/.github/prompts/plan-fastlanesExplicitParquetEncodingsMicroRuns.prompt.md#120-120), lines 120 to 428

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#150-150), lines 150 to 611

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh#1-1), lines 1 to 149

I’ve confirmed the current header shape in detail; now I’m reading decode and encode hot paths to lock down which wire fields are truly consumed at runtime versus only asserted in tests.

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#60-60), lines 60 to 280

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#360-360), lines 360 to 470

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#540-540), lines 540 to 700

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu#3950-3950), lines 3950 to 4095

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#620-620), lines 620 to 910

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#1120-1120), lines 1120 to 1435

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp#1500-1500), lines 1500 to 1675

Searched for regex `TypeCastMode|is_valid_cast_mode|PageLayoutMode::NATIVE64|OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE|layout_mode|bitwidth_mode|is_scalar32_layout|is_split32_layout|padded_size\(|serialize\(uint8_t bw,\s*TypeCastMode` (`**/cpp/**`), 101 results

I found an important dependency: `TypeCastMode` is also used in FastLanes encode internals, so I’m quickly verifying that usage before proposing any removals to avoid breaking debug/host encoder flows.

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh)

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/debug.hpp#1-1), lines 1 to 460

Searched for regex `layout_mode|bitwidth_mode|component_bitwidth_low|component_bitwidth_high|pre_delta|is_valid_for_external_mode|OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE` (`**/cpp/src/io/parquet/page_enc.cu`), 12 results

Searched for regex `header\.layout_mode|header\.bitwidth_mode|OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE|PageLayoutMode|BitwidthMode` (`**/cpp/src/io/parquet/page_fastlanes_decode.cu`), 4 results

Read [](file:///home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu#280-280), lines 280 to 360

I’ve validated the current call sites and there is one key design fork that affects what “remove” means for header fields: full wire-layout compaction versus semantic removal with reserved bytes. I’m asking one short compatibility question so R15–R18 can be unambiguous and revertable.

Analyzing your answers...

I now have enough direct evidence to draft the simplification design and the R15–R18 sequence; the key delta is decoupling PRE_DELTA from Parquet DELTA and fully collapsing internal mode bytes in the wire header. Next I’ll persist the full plan in session memory and then present it in the exact format you requested.

Created memory file [](file:///home/qba/.vscode-server/data/User/workspaceStorage/7938f0ef122d35883c8cb443df714727/GitHub.copilot-chat/memory-tool/memories/Y2FkNzYyNTItMjBiYS00OTZkLThjZDgtMjFhZmQ0ZGE0NDFk/plan.md)

Plan decision applied: fully repurpose/remove legacy mode bytes now (not keep as reserved compatibility bytes).

**1. Target simplified design**
Final minimal FastLanes header fields (still 128-byte header, payload alignment unchanged):
1. Byte 0: component_bitwidth_low
2. Byte 1: component_bitwidth_high
3. Byte 2: pre_delta
4. Byte 3: reserved_flags (must be 0 for now)
5. Bytes 4-7: original_count
6. Bytes 8-11: padded_count
7. Bytes 12-15: body_size
8. Bytes 16-19: min_value_low_bits
9. Bytes 20-23: min_value_high_bits
10. Bytes 24-127: reserved/padding

Remain (and why):
1. Enum: TypeCastMode in common.cuh because it is still used by encoder/debug metadata flow (not for wire decode mode selection).
2. Header fields: component_bitwidth_low, component_bitwidth_high, pre_delta, original_count, padded_count, body_size, min_value_low_bits, min_value_high_bits in common.cuh because these are sufficient for RAW/SPLIT64 decode correctness and size checks.
3. Functions: serialize_scalar32, serialize_split32, deserialize, payload_ptr, has_valid_common_metadata, has_valid_raw_mode_metadata, has_valid_split64_mode_metadata, expected_body_size_bytes, and bool-based external validation path in common.cuh, common.cuh, common.cuh, common.cuh, common.cuh.
4. Decode guard behavior in page_fastlanes_decode.cu because INT64 and UINT64 split64 constraints must remain unchanged.

Remove (and why):
1. Enums: PageLayoutMode, BitwidthMode, ExternalPageMode in common.cuh since external Parquet encoding plus physical type is already authoritative.
2. Header fields and offsets tied to internal mode indirection: bitwidth, layout_mode, bitwidth_mode, OFFSET_LAYOUT_MODE, OFFSET_BITWIDTH_MODE in common.cuh because decode correctness does not require them once low/high component widths are authoritative.
3. DELTA-coupled PRE_DELTA policy helpers in common.cuh and DELTA pre_delta gate in page_fastlanes_decode.cu because PRE_DELTA must be FastLanes-only.
4. Mode validator/helpers: is_valid_layout_mode, is_valid_bitwidth_mode, is_scalar32_layout, is_split32_layout in common.cuh, common.cuh, common.cuh since they only support removed mode abstractions.
5. Dead compatibility wrappers: padded_size, serialize(TypeCastMode...), is_valid_cast_mode overloads in common.cuh, common.cuh, common.cuh.

**2. Dead-code and complexity removal table**

| symbol | current purpose | why redundant | dependencies | removal risk | migration note |
|---|---|---|---|---|---|
| PageLayoutMode | Internal layout identity | External encoding already identifies RAW vs SPLIT64 | common.cuh, parquet_fastlanes_test.cpp | Medium | Replace with split64_mode bool from page.encoding |
| BitwidthMode | Internal stream interpretation tag | Duplicates information from mode + widths | common.cuh, parquet_fastlanes_test.cpp | Medium | Remove enum and related assertions |
| ExternalPageMode | Internal mode enum | Adds indirection and DELTA coupling | common.cuh, page_enc.cu | Medium | Use bool split64_mode and FastLanes-only pre_delta check |
| ExternalPageMode::DELTA path | DELTA-specific PRE_DELTA checks | PRE_DELTA should not follow DELTA family semantics | page_fastlanes_decode.cu, parquet_fastlanes_test.cpp | Low | Remove DELTA-focused FastLanes tests |
| PageHeader.layout_mode | Wire mode byte | Not needed for decode correctness | common.cuh | High | Repurpose byte 1 as component_bitwidth_high |
| PageHeader.bitwidth_mode | Wire mode byte | Not needed for decode correctness | common.cuh | High | Remove from struct and deserialization |
| PageHeader.bitwidth | Raw width duplicate | component_bitwidth_low can be single source | common.cuh, page_fastlanes_decode.cu | Medium | Raw decode reads component_bitwidth_low |
| is_valid_layout_mode / is_valid_bitwidth_mode | Legacy validators | No mode enums after simplification | common.cuh | Low | Delete together with enums |
| is_scalar32_layout / is_split32_layout | Test helpers on removed fields | No longer meaningful once internal mode fields removed | common.cuh, parquet_fastlanes_test.cpp | Low | Replace tests with external-mode assertions |
| default_pre_delta_for_mode / is_pre_delta_valid_for_mode(mode,...) | Mode-coupled PRE_DELTA policy | Violates FastLanes-only PRE_DELTA intent | common.cuh | Medium | Replace with FastLanes-only policy helper |
| padded_size | Alias of header_size | Pure redundancy | common.cuh | Low | Delete |
| serialize(uint8_t, TypeCastMode,...) | Legacy wrapper | No active value path | common.cuh | Low | Delete |
| is_valid_cast_mode overloads | Cast byte validator | No runtime call sites | common.cuh | Low | Delete; keep TypeCastMode enum where actively used |

**3. Incremental run plan (micro-runs)**

R15
1. Run id: R15_pre_delta_fastlanes_only
2. Editable scope files: common.cuh, page_enc.cu, page_fastlanes_decode.cu, parquet_fastlanes_test.cpp
3. Concrete edits: remove DELTA-coupled PRE_DELTA checks and keep RAW/SPLIT64 PRE_DELTA behavior identical.
4. Validation commands:
- if command -v rg >/dev/null 2>&1; then rg -n "ExternalPageMode::DELTA|DELTA_BINARY_PACKED.*pre_delta|is_pre_delta_valid_for_mode" common.cuh page_enc.cu page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; else grep -nE "ExternalPageMode::DELTA|DELTA_BINARY_PACKED.*pre_delta|is_pre_delta_valid_for_mode" common.cuh page_enc.cu page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; fi
- cd ${CUDF_HOME}
- ${CUDF_HOME}build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesRawSplit64DefaultPreDelta:ParquetCpuEncoderTest.FastLanesSplit64DecodeHonorsPreDelta
5. Acceptance criteria: PRE_DELTA policy is FastLanes-only; focused PRE_DELTA tests pass.

R16
1. Run id: R16_header_compact_repurpose_bytes
2. Editable scope files: [cpp/include/cudf/fastlanes/common.cuh](cpp/include/cudf/fastlanes/common.cuh), [cpp/src/io/parquet/page_enc.cu](cpp/src/io/parquet/page_enc.cu), [cpp/src/io/parquet/page_fastlanes_decode.cu](cpp/src/io/parquet/page_fastlanes_decode.cu), [cpp/tests/io/parquet_fastlanes_test.cpp](cpp/tests/io/parquet_fastlanes_test.cpp)
3. Concrete edits: repurpose legacy mode bytes into compact bitwidth/pre_delta fields; remove layout_mode/bitwidth_mode parsing; make component_bitwidth_low authoritative for RAW decode width.
4. Validation commands:
- if command -v rg >/dev/null 2>&1; then rg -n "OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE|layout_mode|bitwidth_mode" cpp/include/cudf/fastlanes/common.cuh cpp/src/io/parquet/page_enc.cu cpp/src/io/parquet/page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; else grep -nE "OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE|layout_mode|bitwidth_mode" cpp/include/cudf/fastlanes/common.cuh cpp/src/io/parquet/page_enc.cu cpp/src/io/parquet/page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; fi
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesHeaderScalar32MetadataRoundTrip:ParquetCpuEncoderTest.FastLanesHeaderSplit32MetadataRoundTrip:ParquetCpuEncoderTest.FastLanesHeaderRejectsMalformedSplit32Metadata
5. Acceptance criteria: header roundtrip/malformed tests pass without internal mode bytes.

R17
1. Run id: R17_mode_enum_and_validator_removal
2. Editable scope files: common.cuh, page_enc.cu, page_fastlanes_decode.cu, parquet_fastlanes_test.cpp
3. Concrete edits: remove PageLayoutMode, BitwidthMode, ExternalPageMode and their validator helpers; keep physical-type split64 guard unchanged.
4. Validation commands:
- if command -v rg >/dev/null 2>&1; then rg -n "PageLayoutMode|BitwidthMode|ExternalPageMode|is_valid_layout_mode|is_valid_bitwidth_mode" common.cuh page_enc.cu page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; else grep -nE "PageLayoutMode|BitwidthMode|ExternalPageMode|is_valid_layout_mode|is_valid_bitwidth_mode" common.cuh page_enc.cu page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; fi
- cd ${CUDF_HOME}
- ${CUDF_HOME}build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesInt64SinglePageForcedBitpack:ParquetCpuEncoderTest.FastLanesUInt64SinglePageForcedBitpack:ParquetCpuEncoderTest.FastLanesInt64PhysicalLogicalTypeSupportMatrix
5. Acceptance criteria: no internal mode enums remain; INT64 and UINT64 split64 behavior unchanged.

R18
1. Run id: R18_dead_api_cleanup_and_full_fastlanes_matrix
2. Editable scope files: [cpp/include/cudf/fastlanes/common.cuh](cpp/include/cudf/fastlanes/common.cuh), [cpp/src/io/parquet/page_enc.cu](cpp/src/io/parquet/page_enc.cu), [cpp/src/io/parquet/page_fastlanes_decode.cu](cpp/src/io/parquet/page_fastlanes_decode.cu), [cpp/tests/io/parquet_fastlanes_test.cpp](cpp/tests/io/parquet_fastlanes_test.cpp)
3. Concrete edits: remove dead wrappers/helpers and remaining stale compatibility text.
4. Validation commands:
- if command -v rg >/dev/null 2>&1; then rg -n "padded_size\\(|is_valid_cast_mode|serialize\\(uint8_t bw,\\s*TypeCastMode|OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE|PageLayoutMode|BitwidthMode|ExternalPageMode" cpp/include/cudf/fastlanes/common.cuh cpp/src/io/parquet/page_enc.cu cpp/src/io/parquet/page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; else grep -nE "padded_size\\(|is_valid_cast_mode|serialize\\(uint8_t bw,\\s*TypeCastMode|OFFSET_LAYOUT_MODE|OFFSET_BITWIDTH_MODE|PageLayoutMode|BitwidthMode|ExternalPageMode" cpp/include/cudf/fastlanes/common.cuh cpp/src/io/parquet/page_enc.cu cpp/src/io/parquet/page_fastlanes_decode.cu cpp/tests/io/parquet_fastlanes_test.cpp; fi
- cd ${CUDF_HOME}
- ${CUDF_HOME}/build.sh libcudf tests
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST
5. Acceptance criteria: dead compatibility API removed and full PARQUET_FASTLANES_TEST green.

**4. Validation matrix**
Remote-compatible preflight (every run):
1. hostname
2. whoami
3. If not remote: ssh qchen@fng01.lab.tuda.systems
4. source ~/miniconda3/etc/profile.d/conda.sh
5. conda activate cudf_dev
6. export CPATH="$CONDA_PREFIX/include/rapids:$CONDA_PREFIX/include"
7. export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane
8. Optional tmp fallback on failure: export TMPDIR=/home/qchen/04_GPUFileFormat-cudf/tmpbuild; mkdir -p "$TMPDIR"; export PARALLEL_LEVEL=4

Phase validation:
1. R15: search for DELTA-coupled PRE_DELTA references, then build/tests, then focused PRE_DELTA filters.
2. R16: search for mode-field offsets/usages removed, then build/tests, then header roundtrip filters.
3. R17: search for enum/validator removals, then build/tests, then INT64 and UINT64 split64 guard filters.
4. R18: search for dead helper removal, then build/tests, then full PARQUET_FASTLANES_TEST.

Required search fallback form in every phase:
1. if command -v rg >/dev/null 2>&1; then rg -n "<pattern>" <files>; else grep -nE "<pattern>" <files>; fi

Required reporting per phase (from your contract):
1. Exact commands run
2. Final input path used
3. Final report/artifact/output locations
4. Archive step executed yes/no
5. Whether rg -n or grep -nE fallback was used
6. Writer gtest binary only if writer tests are run

**5. Risks and rollback**
Top 5 regression risks:
1. Header byte repurpose misaligns decode field reads.
2. RAW decode width migration (bitwidth to component_bitwidth_low) changes unpack behavior.
3. SPLIT64 low/high field mapping inversion corrupts INT64 and UINT64 reconstruction.
4. PRE_DELTA decoupling accidentally changes non-FastLanes DELTA handling.
5. Removing dead wrappers breaks hidden call sites at compile time.

Rollback trigger points:
1. R16 rollback trigger: header roundtrip or malformed metadata filters fail.
2. R17 rollback trigger: any INT64 or UINT64 split64 forced-bitpack test fails.
3. R18 rollback trigger: full PARQUET_FASTLANES_TEST regression.
4. Any run rollback trigger: libcudf tests build failure due missing removed symbol.
5. Any run rollback trigger: RAW/SPLIT64 behavior mismatch against existing expectations.

Rollback strategy:
1. One commit per run.
2. Revert only the failing run commit.
3. Keep previously accepted runs intact.

The full plan has also been persisted to session plan memory for handoff continuity.