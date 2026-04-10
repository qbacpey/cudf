## Plan: INT64 FastLanes GPU Encode/Decode Staging

Integrate INT64 FastLanes GPU encoding and decoding with small reversible runs, while preserving existing SPLIT64 behavior and keeping INT32 on current fallback. External on-file policy follows user decision: FASTLANES_DELTA_BINARY is the INT64 GPU path and is restricted to INT64 only. The plan leaves detailed kernel body logic for later refinement after lane-kernel implementation is finalized.

Confirmed inputs from alignment:
- Primary goal: writer INT64 GPU path first.
- Keep external encoding enums unchanged as enum set, but route INT64 GPU path through existing FASTLANES_DELTA_BINARY value.
- Scope now: INT64 GPU writer integration, INT32 CPU fallback unchanged, reader dispatch seam plus stub native64 entrypoint.
- Feature gate: no temporary gate; always-on once merged.
- Validation cadence: full checkpoint each run.
- Reporting detail: command list, key outcomes, paths.

Conflict resolution interpreted from user answers:
- FASTLANES_DELTA_BINARY semantics: encoding/decoding kernel performs FOR internally.
- Restrict FASTLANES_DELTA_BINARY usage to INT64 only in this stage because INT32 does not yet have native FOR support.

Dependency graph:
- FL64-GPU-R1 -> FL64-GPU-R2
- FL64-GPU-R1 -> FL64-GPU-R3
- FL64-GPU-R2 + FL64-GPU-R3 -> FL64-GPU-R4
- FL64-GPU-R4 -> FL64-GPU-R5
- FL64-GPU-R5 -> FL64-GPU-R6

Micro-runs (provisional)

1) FL64-GPU-R1: Contract and Routing Baseline
- Goal:
  - Lock encoding contract and route selection for INT64 GPU path using FASTLANES_DELTA_BINARY.
  - Enforce INT64-only acceptance for this encoding request.
- Dependencies: none.
- Editable scope allowlist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- Forbidden scope denylist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_bw_kernels.inl
- Concrete edit intent:
  - Update schema/encoding validation in writer path so FASTLANES_DELTA_BINARY is accepted only for INT64 leafs.
  - Keep existing FASTLANE_BITPACK_SPLIT64 behavior unchanged.
  - Ensure encode kernel mask selection for requested FASTLANES_DELTA_BINARY is explicit and non-ambiguous.
- Validation approach:
  - Full build/test checkpoint.
  - Writer encoding validation tests showing INT64 accepted and INT32 rejected/fallback behavior explicit.
- Acceptance:
  - Compile success and no regression in existing SPLIT64 tests.
- Rollback trigger:
  - Any change in existing SPLIT64 expected page encoding metadata.

2) FL64-GPU-R2: Writer Kernel Scaffold and Launch Wiring
- Goal:
  - Add dedicated GPU launch branch in EncodePages for FASTLANES_DELTA_BINARY with two-call orchestration (levels then data).
- Dependencies: FL64-GPU-R1.
- Editable scope allowlist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- Forbidden scope denylist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_bw_kernels.inl
- Concrete edit intent:
  - Introduce gpuEncodeFastLanesInt64Pages kernel entrypoint (scaffold).
  - Wire stream selection and launch ordering consistent with existing encode pattern.
  - Keep INT32 fallback branch untouched.
- Validation approach:
  - Full build/test checkpoint.
  - Kernel launch smoke tests and writer targeted tests.
- Acceptance:
  - FASTLANES_DELTA_BINARY pages execute dedicated launch path without affecting other masks.
- Rollback trigger:
  - Stream join or kernel-mask dispatch regressions in non-fastlanes encodings.

3) FL64-GPU-R3: Reader Dispatch Seam and Native64 Stub Entry
- Goal:
  - Prepare decode dispatch for INT64 FASTLANES_DELTA_BINARY pages with a stub native64 kernel entrypoint.
- Dependencies: FL64-GPU-R1.
- Editable scope allowlist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- Forbidden scope denylist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/native64_bw_kernels.inl
- Concrete edit intent:
  - Route FASTLANES_DELTA_BINARY pages to fastlanes decode kernel mask.
  - Add layout-based dispatch seam for INT64 path and a native64 stub launch point.
  - Preserve existing RAW/SPLIT64 decode behavior unchanged.
- Validation approach:
  - Full build/test checkpoint.
  - Existing RAW/SPLIT64 decode tests must pass unchanged.
  - New targeted tests confirm dispatch branch selection for FASTLANES_DELTA_BINARY pages.
- Acceptance:
  - No behavioral change on existing files; new encoding path reaches intended dispatch seam.
- Rollback trigger:
  - Any regression in existing RAW/SPLIT64 decode tests.

4) FL64-GPU-R4: Header/Metadata Validation for Native64-FOR Contract
- Goal:
  - Align page header validation and metadata assumptions for INT64 native64 FOR path.
- Dependencies: FL64-GPU-R2 and FL64-GPU-R3.
- Editable scope allowlist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/debug.hpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- Forbidden scope denylist:
  - Broad parquet enum additions/removals.
- Concrete edit intent:
  - Add/adjust validation helper paths for INT64 native64 FOR metadata expectations.
  - Keep external encoding enum set unchanged.
  - Ensure mismatch paths fail fast rather than silently decode incorrectly.
- Validation approach:
  - Full build/test checkpoint.
  - Header validation unit tests for valid/invalid native64 metadata.
- Acceptance:
  - Validation logic deterministic and backward-safe for existing page types.
- Rollback trigger:
  - Inability to distinguish legacy split behavior from native64-for path safely.

5) FL64-GPU-R5: End-to-End INT64 GPU Path Activation
- Goal:
  - Activate INT64 GPU encode/decode path end-to-end using the assumed lane-level encode/decode APIs.
- Dependencies: FL64-GPU-R4.
- Editable scope allowlist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/debug.hpp
- Forbidden scope denylist:
  - INT32 fastlanes encode/decode logic changes.
- Concrete edit intent:
  - Integrate kernel calls around encode_lane_native64/decode_lane_native64 style API.
  - Ensure finish_page_encode/compression wiring remains standard.
  - Ensure page sizing and overflow checks enforce max_data_size safety.
- Validation approach:
  - Full build/test checkpoint.
  - Targeted roundtrip tests for INT64/UINT64 across page boundaries.
  - Existing SPLIT64 tests remain green.
- Acceptance:
  - INT64 FASTLANES_DELTA_BINARY path roundtrips correctly and does not alter legacy behavior.
- Rollback trigger:
  - Any data mismatch in roundtrip matrix or existing split64 regressions.

6) FL64-GPU-R6: Hardening, Matrix, and Remote-ready Packet
- Goal:
  - Final hardening pass, evidence packaging, and remote execution packet definition.
- Dependencies: FL64-GPU-R5.
- Editable scope allowlist:
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_test.cpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_writer_test.cpp
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_generated_test.cu
  - /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/tests/io/parquet_fastlanes_native64_bw37_test.cu
- Forbidden scope denylist:
  - Core encode/decode algorithm changes (stability run only).
- Concrete edit intent:
  - Add/adjust assertions for new encoding contract and INT64-only enforcement.
  - Finalize remote-ready validation command packet and evidence checklist.
- Validation approach:
  - Full build/test checkpoint.
  - Targeted native64 plus writer tests.
- Acceptance:
  - Reproducible verification packet for remote execution with explicit outputs.
- Rollback trigger:
  - Any non-deterministic failures across repeated runs.

Relevant file anchors to reuse:
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_enc.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_fastlanes_decode.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/page_hdr.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/parquet_gpu.hpp
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/src/io/parquet/writer_impl.cu
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/common.cuh
- /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/cudf-fastlane/cpp/include/cudf/fastlanes/debug.hpp

Remote-ready validation policy (commands prepared, not executed in planning stage):
- Follow remote contract order:
  - ssh to qchen@fng01.lab.tuda.systems
  - source conda activation script
  - conda activate cudf_dev
  - export CPATH
  - export CUDF_HOME for fastlane-working branch
  - run builds with ${CUDF_HOME}/build.sh only
- Full checkpoint each run as requested:
  - ${CUDF_HOME}/build.sh libcudf
  - ${CUDF_HOME}/build.sh libcudf tests
  - ctest --output-on-failure --no-tests=error
- Targeted checks each run:
  - writer gtests via resolved WRITER_GTEST_BIN
  - fastlanes gtests (PARQUET_FASTLANES_TEST filters)

Open items for next refinement:
- Detailed lane-kernel insertion design once final encode_lane_native64/decode_lane_native64 production APIs are available.
- Exact metadata fields for native64-for mode in header validation path.
- Whether FASTLANES_DELTA_BINARY repurpose requires migration notes for any existing datasets using prior semantics.
