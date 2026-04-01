# FastLanes Integration Handover Report (Updated 2026-04-02)

## 1. Scope and Purpose

This report documents the current FastLanes integration status in cuDF Parquet after the phase-2 split32 work. It focuses on:

- current INT32 and INT64/UINT64 behavior,
- where encode/decode and gating logic live,
- how test coverage is organized,
- what to modify when native 64-bit FastLanes payload is introduced.

## 2. Current Encoding Strategy

### 2.1 INT32 physical columns

INT32 physical columns use scalar32 FastLanes layout:

- one packed stream,
- one bitwidth per page,
- page-local min value stored in header.

### 2.2 INT64 and UINT64 logical data

INT64/UINT64 currently use split32 layout:

- each page keeps the same logical row count,
- values are split into low and high 32-bit components,
- low and high components are encoded as two separate streams,
- each stream has its own bitwidth and min component base.

So your understanding is correct: current INT64/UINT64 handling is split into low/high vectors and encoded independently.

## 3. Decode/Encode Architecture (Current)

### 3.1 Decode path

File:
- cpp/src/io/parquet/page_fastlanes_decode.cu

Current structure:
- shared page setup and validation helper,
- dedicated INT32 kernel path,
- dedicated INT64 split32 kernel path,
- host launch wrapper dispatches both type-specific kernels; each kernel exits early for non-matching physical pages.

### 3.2 Encode path

File:
- cpp/include/cudf/fastlanes/fastlanes_encode.cuh

Current structure:
- encode_page delegates to dedicated helpers:
- scalar32 helper for INT32,
- split32 helper for INT64,
- shared upload helper for host->device encoded blob transfer.

### 3.3 Writer/runtime gating

Files:
- cpp/src/io/parquet/writer_impl.cu
- cpp/src/io/parquet/page_enc.cu

Current behavior:
- schema-side and runtime-side checks are explicit and readable,
- logical allowlist/fallback behavior remains unchanged,
- unsupported logical types still fall back from requested FastLanes to safe alternatives.

## 4. Header and Metadata Contract

File:
- cpp/include/cudf/fastlanes/common.cuh

Important layout enums:
- SCALAR32
- SPLIT32
- NATIVE64 (reserved)

Current validation for INT64 physical pages accepts SPLIT32 only.

## 5. Test Organization (Updated)

File:
- cpp/tests/io/parquet_fastlanes_test.cpp

The test file now has explicit grouping comments to make intent clear:

1. INT32 baseline and edge-path roundtrip
2. INT32 diagnostics (high bitwidth, tail pages)
3. Header metadata validity (scalar32 and split32)
4. INT32 logical-type force-enable coverage
5. INT64/UINT64 split32 data-path coverage
6. INT32 logical support/fallback matrix
7. Mixed-encoding workload-pattern regression

Naming hints:
- ForcedBitpack: FastLanes was explicitly requested
- Fallback: FastLanes request should be rejected and fallback used
- BoundarySizes: vector boundary and tail-page coverage

## 6. Native64 Future Work: Exact Modification Points

When introducing a real native 64-bit FastLanes path, start at these locations:

### 6.1 Header validation and size accounting

File:
- cpp/include/cudf/fastlanes/common.cuh

Update:
- is_valid_for_physical(...): allow and validate NATIVE64 for INT64 physical pages.
- expected_body_size_bytes(): add NATIVE64 body-size computation.

### 6.2 Encoder dispatch

File:
- cpp/include/cudf/fastlanes/fastlanes_encode.cuh

Update:
- encode_page(...): add branch to encode_native64_page(...).
- keep encode_split32_page(...) for backward compatibility or gated fallback.
- add native64 serialization metadata in header.

### 6.3 Decoder dispatch and kernel

File:
- cpp/src/io/parquet/page_fastlanes_decode.cu

Update:
- add native64 decode kernel path,
- dispatch INT64 pages by header layout mode (split32 vs native64),
- keep split32 path for existing data files.

### 6.4 Page-size reservation logic

File:
- cpp/src/io/parquet/page_enc.cu

Update:
- revise reservation from two-component split assumption to layout-based stream accounting,
- preserve conservative bounds during transition.

## 7. Build and Validation Workflow (Remote)

Required environment in remote shell:

- CUDF_HOME=/home/qchen/GPUFileFormat-cudf
- conda activate cudf_dev
- export CPATH="$CONDA_PREFIX/include/rapids:$CONDA_PREFIX/include"

Build command:

- ${CUDF_HOME}/build.sh libcudf tests

Focused validation target:

- ./cpp/build/gtests/PARQUET_FASTLANES_TEST
- ctest --test-dir cpp/build -R PARQUET_FASTLANES_TEST --output-on-failure

## 8. Current Practical Status

- INT32 path: stable for the current logical allowlist.
- INT64/UINT64 path: split32 implementation is active and tested.
- Test suite for PARQUET_FASTLANES_TEST: currently green in direct run and ctest in this environment.
- Native64 layout: not yet implemented; extension points are now documented in code.

## 9. Recommended Next Steps

1. Keep split32 as default while native64 is prototyped behind explicit layout handling.
2. Add native64-specific tests without removing split32 coverage.
3. Keep compatibility read support for split32 pages even after native64 write path lands.
4. Run mixed-encoding regression and matrix tests after each metadata contract change.
