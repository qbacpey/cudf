## Plan: FastLanes Three-Path Refactor Packet

Refactor parquet FastLanes encode/decode from template-coupled INT32/INT64 logic into three explicit paths with strict encoding-to-kernel mapping: INT32 RAW CPU, INT64 SPLIT64 CPU, INT64 NATIVE64 GPU. Preserve explicit SPLIT64 support, activate FASTLANES_DELTA_BINARY as preferred INT64/UINT64 FastLanes path, and enforce remote full-ctest validation on every run.

### Current Understanding Snapshot

Goal summary:
- Split encoder public API and implementation into explicit classes/helpers for RAW32, SPLIT64, and NATIVE64.
- Implement GPU-native INT64 encode path using native64 APIs without host download of full input values.
- Separate decode kernels so each FastLanes encoding has a dedicated kernel and strict route by page.encoding.
- Deliver staged, reversible runs with remote validation and evidence collection per run.

Confirmed constraints:
- Default INT64/UINT64 FastLanes direction: prefer FASTLANES_DELTA_BINARY when eligible.
- API rename policy: hard rename (no compatibility alias for FastLanesInt64Encoder).
- Decode routing policy: key off page.encoding == FASTLANES_DELTA_BINARY for native64.
- Scope: include tooling/docs in the same planning packet (not deferred).
- Validation cadence: full remote ctest checkpoint on every run.
- Remote contract: use current terminal, SSH to fng01 if needed, source conda first, build via ${CUDF_HOME}/build.sh only, report command/evidence locations. Skill vscode-remote://wsl+ubuntu/home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/.github/skills/remote-working-contract/SKILL.md

Open decisions (runtime only):
- INPUT_PATTERN concrete filename(s) for sentinel roundtrip input.
- RUN_TAG concrete value for this execution cycle.

Decisions from alignment:
- Keep explicit FASTLANE_BITPACK_SPLIT64 requests valid and non-degraded.
- Native64 decode routing remains strict to encoding enum, not inferred header heuristics.

### Dependency Graph

FLN-R1 -> FLN-R2 -> FLN-R3 -> FLN-R4 -> FLN-R5 -> FLN-R6

Notes:
- FLN-R3 depends on FLN-R2 policy plumbing.
- FLN-R4 depends on FLN-R3 for writer-emitted native64 pages to decode.
- FLN-R6 depends on FLN-R5 validated behavior before docs/tooling freeze.

### Global Validation Contract (applies to every run)

Pre-run remote setup:
1. Verify remote context: whoami + hostname.
2. If local shell, connect via ssh qchen@fng01.lab.tuda.systems.
3. source ~/miniconda3/etc/profile.d/conda.sh
4. conda activate cudf_dev
5. export CPATH=${CONDA_PREFIX}/include/rapids:${CONDA_PREFIX}/include
6. export CUDF_HOME=/home/qchen/04_GPUFileFormat-cudf/cudf-fastlane
7. export ROUNDTRIP_INPUT_ROOT=/home/qchen/04_GPUFileFormat-cudf/large_input
8. export PARQUET_IO_SHARED_ROOT=/home/qchen/04_GPUFileFormat-cudf/parquet_io_shared
9. If /tmp pressure occurs: export TMPDIR=/home/qchen/04_GPUFileFormat-cudf/tmpbuild; mkdir -p ${TMPDIR}; export PARALLEL_LEVEL=4

Per-run mandatory validation:
1. Build from ${CUDF_HOME}: ./build.sh libcudf
2. Build tests from ${CUDF_HOME}: ./build.sh libcudf tests
3. Resolve writer gtest binary:
   - If ${CUDF_HOME}/cpp/build/gtests/PARQUET_WRITER_TEST exists, use it.
   - Else use ${CUDF_HOME}/cpp/build/gtests/PARQUET_TEST.
4. Run targeted gtests for changed scope.
5. Run full ctest checkpoint: ctest --output-on-failure --no-tests=error
6. Run sentinel roundtrip using ${ROUNDTRIP_INPUT_ROOT}/${INPUT_PATTERN} and write outputs under:
   - ${PARQUET_IO_SHARED_ROOT}/reports/$(basename ${CUDF_HOME})
   - ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})
7. Pull small reports locally via /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/sync_parquet_io_shared_small.sh pull

Per-run report must include:
- Exact commands executed.
- Final input path used.
- Final reports/artifacts/output locations.
- Whether archive step executed.
- Whether rg -n or grep -nE fallback was used.
- Resolved WRITER_GTEST_BIN if writer tests executed.

### Run Packet Details

#### FLN-R1: Encoder API Split + De-Template (Behavior-Preserving)

1. Goal:
- Remove template-funneled encode entry points and expose explicit encoder classes while preserving existing RAW32/SPLIT64 behavior.

2. Dependencies:
- None.

3. Editable scope (allowlist):
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu

4. Forbidden scope (denylist):
- page_fastlanes_decode.cu decode kernels
- native64_cuda.cu/.cuh algorithm behavior
- writer policy defaults

5. Concrete edits:
- Rename FastLanesInt64Encoder to FastLanesInt64Split32Encoder in public header and implementation.
- Add FastLanesInt64NativeEncoder class with same public method signatures.
- Remove/replace template entry points in fastlanes.cu:
  - encode_page_impl<T>
  - encode_pages_impl<T>
  - create_empty_result<T>
- Introduce explicit helper functions per path:
  - encode_scalar32_page_helper + encode_scalar32_pages_helper
  - encode_split32_page_helper + encode_split32_pages_helper
  - encode_native64_page_helper + encode_native64_pages_helper (stubbed to explicit not-yet-enabled path if needed)
- Update page_enc.cu references to renamed split32 class while preserving existing routing.

6. Validation commands:
- rg -n "encode_page_impl<|encode_pages_impl<|if constexpr" cpp/src/fastlanes/fastlanes.cu
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesInt32*:
  ParquetCpuEncoderTest.FastLanesInt64*:
  ParquetCpuEncoderTest.FastLanesUInt64*
- ctest --output-on-failure --no-tests=error
- Sentinel roundtrip command on ${ROUNDTRIP_INPUT_ROOT}/${INPUT_PATTERN}

7. Acceptance criteria:
- No template dispatch remains for encode entry points in fastlanes.cu.
- Existing FASTLANE_BITPACK_RAW and FASTLANE_BITPACK_SPLIT64 test expectations unchanged.
- Full ctest passes.

8. Rollback trigger:
- Any regression in existing split64/raw encoding outputs or class API compile break.

9. Evidence to collect:
- grep/rg output proving template entry points removed.
- Targeted gtest summaries.
- Full ctest summary.
- Sentinel report location.

#### FLN-R2: Policy Plumbing for Third Encoding Mode (Non-Activation Stage)

1. Goal:
- Wire metadata/mask recognition for FASTLANES_DELTA_BINARY safely without yet relying on native64 encode completion.

2. Dependencies:
- Depends on FLN-R1.

3. Editable scope (allowlist):
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl_preprocess_utils.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_decode.cuh

4. Forbidden scope (denylist):
- Native64 writer pack/unpack body generation
- Native64 decode kernel body

5. Concrete edits:
- Add writer metadata validation for column_encoding::FASTLANES_DELTA_BINARY (INT64/UINT64 flat, supported logical constraints).
- Restrict/clarify runtime support helpers so FASTLANES_DELTA_BINARY eligibility is explicit for INT64 path.
- Extend FastLanes kernel-mask helper coverage in page_enc.cu to include FASTLANES_DELTA_BINARY while preserving current behavior gate.
- Update fastlanes_encoding_for_mask to explicit 3-way mapping.
- Add FASTLANES_DELTA_BINARY to supported decode-encoding recognition:
  - is_supported_encoding in parquet_gpu.hpp
  - encoding_to_string in reader_impl_preprocess_utils.cu
  - kernel_mask_for_page in page_hdr.cu
  - setup-local page acceptance path in page_decode.cuh where required
- Keep activation guard so path does not silently mis-encode before R3 native writer lands.

6. Validation commands:
- rg -n "FASTLANES_DELTA_BINARY" on edited parquet source files.
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesHeader*:
  ParquetCpuEncoderTest.FastLanesRawSplit64DefaultPreDelta
- ctest --output-on-failure --no-tests=error
- Sentinel roundtrip

7. Acceptance criteria:
- FASTLANES_DELTA_BINARY is recognized in preprocess/mask plumbing.
- No unsupported-encoding regressions for RAW/SPLIT64 files.
- Full ctest passes.

8. Rollback trigger:
- Any path that accepts FASTLANES_DELTA_BINARY but routes to invalid encoder payload generation.

9. Evidence to collect:
- grep evidence for enum/mask wiring.
- Writer/fastlanes test logs.
- ctest summary.

#### FLN-R3: Native64 GPU Writer + INT64 Default Activation

1. Goal:
- Implement NATIVE64 GPU encode path and make it the preferred FastLanes path for eligible INT64/UINT64 unless split64 explicitly requested.

2. Dependencies:
- Depends on FLN-R2.

3. Editable scope (allowlist):
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_cuda.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_cuda.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh

4. Forbidden scope (denylist):
- Native64 decode kernel integration (R4)

5. Concrete edits:
- Implement encode_native64_page_helper in fastlanes.cu with GPU-only value path:
  - derive base/min via native64_generated::derive_min_base_bits
  - directly use min as the base for GPU encode without host delta add, refer to /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu as example for API usage and expected behavior 
  - allocate device buffer for [128-byte header][packed body]
  - call native64_generated::encode_by_bw_gpu_device_ptrs for payload region
  - produce header bytes on host and copy to device header region
- Ensure EncodedPageResult accounting remains correct (total_size/body_size/min_value/bitwidth).
- Route page_enc INT64 path by selected kernel mask:
  - explicit FASTLANE_BITPACK_SPLIT64 -> FastLanesInt64Split32Encoder
  - FASTLANES_DELTA_BINARY -> FastLanesInt64NativeEncoder
- Activate default INT64/UINT64 FastLanes preference to FASTLANES_DELTA_BINARY where eligible.
- Preserve explicit split64 request behavior unchanged.

6. Validation commands:
- rg -n "cudaMemcpy.*DeviceToHost" cpp/src/fastlanes/fastlanes.cu cpp/src/fastlanes/native64_cuda.cu (verify no full-input D2H in native path)
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetFastLanesNative64GeneratedTest.*:
  ParquetFastLanesNative64Bw37Test.*
- ctest --output-on-failure --no-tests=error
- Sentinel roundtrip + one focused writer metadata inspection for INT64 default files

7. Acceptance criteria:
- INT64/UINT64 default FastLanes writes produce FASTLANES_DELTA_BINARY metadata.
- Explicit split64 requests still produce FASTLANE_BITPACK_SPLIT64.
- Native64 generated parity tests pass.
- Full ctest passes.

8. Rollback trigger:
- Default INT64 path regresses to wrong encoding.
- Native64 parity mismatch or stream-usage failures.

9. Evidence to collect:
- Footer encoding evidence for default and explicit split64 cases.
- Native64 parity logs.
- ctest summary.

#### FLN-R4: Decoder Three-Kernel Separation + Native64 Read Path

1. Goal:
- Refactor decode path into one kernel per FastLanes encoding and implement native64 decode using native64_generated API.

2. Dependencies:
- Depends on FLN-R3.

3. Editable scope (allowlist):
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_decode.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh

4. Forbidden scope (denylist):
- Writer selection policy logic

5. Concrete edits:
- Refactor setup_and_validate_fastlanes_page to recognize three encodings:
  - FASTLANE_BITPACK_RAW
  - FASTLANE_BITPACK_SPLIT64
  - FASTLANES_DELTA_BINARY
- Replace current two-kernel naming with explicit three-kernel layout:
  - decode_fastlanes_raw32_kernel
  - decode_fastlanes_split64_kernel
  - decode_fastlanes_native64_kernel
- Implement native64 decode kernel using native64_generated decode API and ensure no external delta add is applied.
- Update decode_fastlanes_binary host launch to launch all three kernels per level type; rely on encoding check + early-exit.
- Update debug metadata/log mode labels so native64 pages are traceable.

6. Validation commands:
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --gtest_filter=ParquetCpuEncoderTest.FastLanesInt64*:
  ParquetCpuEncoderTest.FastLanesUInt64*:
  ParquetFastLanesNative64GeneratedTest.*
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings
- ctest --output-on-failure --no-tests=error
- Sentinel roundtrip on ${INPUT_PATTERN}

7. Acceptance criteria:
- Reader successfully decodes native64-written pages.
- Existing RAW/SPLIT64 decode tests remain green.
- Full ctest passes.

8. Rollback trigger:
- decode_error INVALID_DATA_TYPE increase for legacy split64/raw fixtures.
- Native64 read path mismatches roundtrip correctness.

9. Evidence to collect:
- Focused decode test logs.
- Roundtrip evidence for native64 + legacy split64/raw.
- ctest summary.

#### FLN-R5: Coverage Expansion + Regression Matrix Hardening

1. Goal:
- Expand tests to lock three-path behavior and prevent regressions in writer/read routing, metadata, and parity boundaries.

2. Dependencies:
- Depends on FLN-R4.

3. Editable scope (allowlist):
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_writer_test.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/CMakeLists.txt

4. Forbidden scope (denylist):
- Production encode/decode algorithm semantics unless required for test unblocking

5. Concrete edits:
- Add writer tests for explicit FASTLANES_DELTA_BINARY request acceptance.
- Add writer tests proving default INT64/UINT64 FastLanes emits FASTLANES_DELTA_BINARY.
- Add tests proving explicit split64 still emits FASTLANE_BITPACK_SPLIT64.
- Add roundtrip tests asserting decode correctness across all three path encodings.
- Add metadata/header tests for native64 routing assumptions and malformed-mode rejection.
- Ensure PARQUET_FASTLANES_TEST linkage remains correct for native64 symbols.

6. Validation commands:
- ${WRITER_GTEST_BIN} --gtest_filter=ParquetWriterTest.UserRequestedEncodings:ParquetWriterTest.*FastLane*
- ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST
- python3 ${CUDF_HOME}/cpp/examples/parquet_io/tools/tests/run_parity_matrix.py --gtest-binary ${CUDF_HOME}/cpp/build/gtests/PARQUET_FASTLANES_TEST --run-tag ${RUN_TAG} --report-json ${PARQUET_IO_SHARED_ROOT}/reports/$(basename ${CUDF_HOME})/${RUN_TAG}/r4_matrix_summary.json --phase all --with-stability --with-bw37-guard --with-ctest
- ctest --output-on-failure --no-tests=error
- Sentinel roundtrip

7. Acceptance criteria:
- Three-path encode/decode behavior is test-locked.
- Parity matrix and bw37 guard pass.
- Full ctest passes.

8. Rollback trigger:
- New tests expose unstable/native64 nondeterministic behavior.

9. Evidence to collect:
- New gtest pass output.
- r4_matrix_summary.json with path_policy_ok=true.
- ctest summary.

#### FLN-R6: Tooling, Docs, and Remote Reporting Finalization

1. Goal:
- Align tooling/docs/reporting with implemented three-path model and remote contract output expectations.

2. Dependencies:
- Depends on FLN-R5.

3. Editable scope (allowlist):
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/tools/tests/run_parity_matrix.py
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/tools/roundtrip/parquet_io_roundtrip_check.py
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/docs/**
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/README.md
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/sync_parquet_io_shared_small.sh (if path/report sync updates needed)

4. Forbidden scope (denylist):
- Core encode/decode algorithms except urgent bugfix from R5 validation

5. Concrete edits:
- Update docs to describe:
  - RAW32 CPU path
  - SPLIT64 CPU path
  - NATIVE64 GPU path
  - default INT64/UINT64 policy and explicit split64 override
- Update tooling/report templates to print required remote-contract summary fields consistently.
- Confirm report/artifact path conventions for shared-root no-symlink workflow.
- Add/refresh operator checklist for remote run reporting and archive markers.

6. Validation commands:
- run_parity_matrix.py --dry-run with report path contract
- one live parity run with ${RUN_TAG}
- one sentinel roundtrip run from ${INPUT_PATTERN}
- ctest --output-on-failure --no-tests=error

7. Acceptance criteria:
- Documentation reflects actual three-path runtime behavior.
- Tooling outputs contract-compliant reporting fields.
- Full ctest passes.

8. Rollback trigger:
- Tool/docs changes introduce path/reporting regressions or mismatch implemented behavior.

9. Evidence to collect:
- Dry-run and live-run tool outputs.
- Report/artifact path samples.
- ctest summary.

### Relevant Files (Primary Architecture)

- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/fastlanes_encode.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/fastlanes.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_cuda.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/fastlanes/native64_cuda.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_decode.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/reader_impl_preprocess_utils.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_writer_test.cpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/examples/parquet_io/tools/tests/run_parity_matrix.py

### Risk Matrix

1. Risk: INT64 default policy change silently alters downstream expectations.
- Severity: High
- Detection: Writer metadata tests and mixed-encoding roundtrip comparisons.
- Mitigation: Explicit split64 override preserved + dedicated regression tests before activation gate.

2. Risk: Native64 header/body contract mismatch causes decode corruption.
- Severity: High
- Detection: Native64 generated parity + roundtrip decode tests.
- Mitigation: Enforce strict encoding-based kernel routing; keep header validation mode-specific.

3. Risk: Stream misuse under ctest stream-identification mode.
- Severity: Medium
- Detection: PARQUET_FASTLANES_TEST under ctest checkpoint.
- Mitigation: Use stream-bound APIs and avoid implicit default-stream operations.

4. Risk: Build instability on fng01 due /tmp pressure.
- Severity: Medium
- Detection: build.sh failures with temp-space errors.
- Mitigation: TMPDIR and PARALLEL_LEVEL fallback in remote contract.

5. Risk: Writer gtest binary path mismatch across environments.
- Severity: Medium
- Detection: missing binary/runtime invocation failure.
- Mitigation: explicit WRITER_GTEST_BIN resolution logic each run.

6. Risk: Partial activation (encoding policy before implementation) causes broken pages.
- Severity: High
- Detection: R2 targeted tests + R3 activation gate.
- Mitigation: staged activation design: plumbing first, native writer next, decode after.

### Milestone Gates

- Gate M1 (after FLN-R2): plumbing complete, no behavior change regressions, full ctest green.
- Gate M2 (after FLN-R4): end-to-end read/write across RAW32, SPLIT64, NATIVE64 validated.
- Gate M3 (after FLN-R6): docs/tooling/reporting aligned, full validation packet archived.

### Execution Order Recommendation

1. Execute FLN-R1 and FLN-R2 as stabilization foundation.
2. Execute FLN-R3 to activate native64 writer/defaults only after R2 passes.
3. Execute FLN-R4 immediately after R3 for reader parity completion.
4. Execute FLN-R5 for coverage hardening before declaring architecture complete.
5. Execute FLN-R6 to finalize tooling/docs/reporting once behavior is stable.

### Ready-to-Run Packet (Next Run Only: FLN-R1)

Run ID: FLN-R1
Title: Encoder API Split + De-Template (Behavior-Preserving)

Command packet summary:
1. Verify remote shell (whoami, hostname), SSH if needed.
2. Apply remote env contract (conda + CPATH + CUDF_HOME).
3. Build: ${CUDF_HOME}/build.sh libcudf
4. Build tests: ${CUDF_HOME}/build.sh libcudf tests
5. Resolve WRITER_GTEST_BIN path.
6. Run targeted writer + fastlanes gtests for RAW/SPLIT64 parity.
7. Run full ctest checkpoint.
8. Execute sentinel roundtrip on ${ROUNDTRIP_INPUT_ROOT}/${INPUT_PATTERN}.
9. Sync small reports locally.
10. Emit run report fields required by remote contract.

Expected evidence paths:
- ${PARQUET_IO_SHARED_ROOT}/reports/$(basename ${CUDF_HOME})/${RUN_TAG}/
- ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/${RUN_TAG}/

### Delta Log

Changed since previous concise packet:
- Added run-by-run 10-part execution format with allowlist/denylist/concrete edits/rollback/evidence.
- Added explicit remote-contract command and reporting requirements as global invariant.
- Added dependency-safe activation strategy (policy plumbing before native writer, decode after writer).
- Added dedicated risk matrix, milestone gates, and next-run packet.
- Incorporated confirmed decisions: default INT64/UINT64 native64 preference, no API alias, strict encoding-based routing, tooling/docs included, full ctest every run.

Still unresolved:
- Runtime values only: INPUT_PATTERN selection and RUN_TAG concrete value for execution.