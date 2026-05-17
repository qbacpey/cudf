# FastLanes Integration Handover Report (2026-03-28)

## 1. Purpose

This document is a handover report for the FastLanes integration in cuDF Parquet.
It explains:

- what was integrated and why,
- where code was changed and what each change means,
- which trade-offs were explicitly chosen,
- how to reproduce current behavior and results,
- how to extend support (logical INT32 types, and future INT64 direction).

## 2. Decision History and Scope Rationale

This work started from exploring three families:

- FastLanesGPU (integer-oriented)
- G-ALP (floating-point-oriented)
- FSST-GPU (string-oriented)

Initial decision from discussion:

- prioritize FastLanes in cuDF first,
- keep path simple and practical,
- focus on integer path before broader datatype coverage.

Why cuDF-first instead of Arrow-RS/Vortex path:

- decode path is GPU-dependent and already coupled with cuDF parquet pipeline,
- integrating at cuDF C++ layer gives direct control of writer/reader kernels,
- avoids introducing Rust runtime/dependency coupling during initial implementation.

Pragmatic scope that was chosen:

- first make write/read path work for strict INT32-centric workloads,
- defer broader type support and advanced bitwidth tuning until baseline pipeline is stable.

## 3. Current Supported Behavior (Important for Handover)

Current FastLanes path is intentionally constrained:

- flat columns only (no list/nested decode path in FastLanes kernel)
- pages with nulls are rejected by FastLanes decode path
- strict INT32 physical path with logical-type gating for safety
- currently enabled logical classes (INT32 physical): INT8, UINT8, INT16, UINT16, INT32, UINT32, Date32, Decimal32, TimeMillis(duration_ms, duration_s)
- explicitly unsupported logical class (still INT32 physical): TimeMillis(duration_D)
- legacy UINT cast-mode pages (cast_mode=0) are rejected during decode
- explicit fallback (requested FastLanes ignored) for unsupported INT32 logical classes
- INT64 support is not production-ready in this path

Key internal notes are in:

- `cpp/include/cudf/fastlanes/IMPLEMENTATION_NOTES.md`

## 4. What Was Integrated (Architecture View)

### 4.1 API and encoding surface

- Added custom encodings in cuDF IO metadata API:
  - `FASTLANES_BITPACK`
  - `FASTLANES_DELTA_BINARY` *
- File: `cpp/include/cudf/io/types.hpp`

Meaning:

- caller can request FastLanes encoding through `column_in_metadata::set_encoding()`.

### 4.2 FastLanes core components in cuDF tree

Added FastLanes core headers and implementation glue:

- `cpp/include/cudf/fastlanes/common.cuh`
- `cpp/include/cudf/fastlanes/fastlanes_encode.cuh`
- `cpp/include/cudf/fastlanes/fastlanes_decode.cuh`
- `cpp/include/cudf/fastlanes/debug.hpp`
- `cpp/include/cudf/fastlanes/fls_gen/*`
- `cpp/src/fastlanes/fastlanes.cu`
- `cpp/src/fastlanes/pack.cpp`
- `cpp/src/fastlanes/transpose.cpp`
- `cpp/src/fastlanes/unrsum.cpp`

Meaning:

- FastLanes-generated pack/unpack and page-header logic are now available inside cuDF build.

### 4.3 Writer path integration

Main writer hooks:

- `cpp/src/io/parquet/writer_impl.cu`
  - validates requested encoding and currently enforces `FASTLANES_BITPACK` on INT32 columns only
- `cpp/src/io/parquet/page_enc.cu`
  - maps requested encoding to FastLanes kernel mask
  - adds reservation logic for FastLanes tail pages
  - includes pre-encoded page-copy path for FastLanes payloads

Key meaning of these changes:

- FastLanes payload is prepared per page and copied into parquet page buffer,
- writer reserves a conservative max page size for FastLanes tail pages to avoid under-allocation,
- behavior is fail-fast instead of per-page mixed fallback when constraints are violated.

### 4.4 Reader path integration

Reader hooks:

- `cpp/src/io/parquet/page_hdr.cu`
  - maps `Encoding::FASTLANES_BITPACK` to dedicated decode kernel mask
- `cpp/src/io/parquet/reader_impl.cpp`
  - launches `decode_fastlanes_binary` + debug path when FastLanes pages are present
- `cpp/src/io/parquet/page_fastlanes_decode.cu`
  - dedicated FastLanes decode kernel

Key meaning of decode changes:

- one warp per page decode model,
- strict checks for unsupported shapes (nested/list/null pages),
- payload staging into aligned shared-memory words before unpack (mitigates mixed-column misalignment cases where payload pointer may be unaligned).

## 5. Critical Trade-offs That Were Chosen

### 5.1 Strict shape constraints

Chosen for initial reliability and implementation speed:

- no list/nested/null-heavy generalized FastLanes decode in first pass,
- reject unsupported pages instead of silently producing partial behavior.

Trade-off:

- simpler and safer kernel path now,
- broader schema coverage postponed.

### 5.2 INT32-first strategy

Even though some code paths touched INT64 experiments, production path remains INT32-centric.

Why:

- upstream FastLanes GPU unpack is hardwired around 32-bit generated kernels,
- extending to 64-bit is not a simple `uint32_t -> uint64_t` replacement.

Trade-off:

- predictable progress for INT32 workloads,
- INT64 remains future work.

### 5.3 FOR normalization for negative INT32

Problem observed:

- direct int32->uint32 reinterpret on negative values can force 32-bit bitwidth,
- then encoded body is not smaller and header overhead can make page larger.

Chosen solution:

- page-local FOR (store deltas from page min),
- keep `min_value` in FastLanes page header,
- reconstruct during decode.

Trade-off:

- fixes common negative-value edge cases,
- still does not eliminate all mixed-schema corner cases.

### 5.4 No per-page fallback mixing in same chunk (for now)

Not implemented in this phase.

Trade-off:

- lower complexity and cleaner invariants,
- fewer rescue paths for pathological pages.

## 6. File-Level Handover Map (What to Read First)

### 6.1 Encoding API and metadata surface

- `cpp/include/cudf/io/types.hpp`
  - `column_encoding` includes FastLanes entries.

### 6.2 FastLanes implementation internals

- `cpp/include/cudf/fastlanes/common.cuh`
  - FastLanes page header, cast mode, min-value metadata.
- `cpp/include/cudf/fastlanes/fastlanes_encode.cuh`
  - host/device transfer + page normalization + bitwidth selection + serialization.
- `cpp/include/cudf/fastlanes/debug.hpp`
  - debug structures and dump helpers.
- `cpp/include/cudf/fastlanes/IMPLEMENTATION_NOTES.md`
  - assumptions and boundary decisions.

### 6.3 Parquet writer integration

- `cpp/src/io/parquet/writer_impl.cu`
  - request validation and current datatype gating.
- `cpp/src/io/parquet/page_enc.cu`
  - kernel mask routing, page sizing reservation, and page copy path.

### 6.4 Parquet reader integration

- `cpp/src/io/parquet/page_hdr.cu`
  - decode kernel mask mapping.
- `cpp/src/io/parquet/reader_impl.cpp`
  - decode launch orchestration.
- `cpp/src/io/parquet/page_fastlanes_decode.cu`
  - FastLanes decode kernel and aligned payload staging.

### 6.5 Test and reproducibility tooling

- `cpp/tests/io/parquet_fastlanes_test.cpp`
  - focused FastLanes regression and edge cases.
- `cpp/examples/parquet_io/fastlane_*`
  - sanity/verification examples.
- `cpp/examples/parquet_io/tools/roundtrip/parquet_io_roundtrip_check.py`
  - c++ chunked rewrite + validation driver.
- `cpp/examples/parquet_io/tools/search/search_best_parquet_encoding.py`
  - reproducible encoding-combination search.

## 7. Reproduction Guide

Run from repo root unless stated.

### 7.1 Build and tests

```bash
cd $CUDF_HOME
./build.sh
./build.sh libcudf tests
ctest --test-dir ${CUDF_HOME}/cpp/build -R PARQUET_TEST
```

FastLanes runtime-linking note for this repo layout:

- `PARQUET_FASTLANES_TEST` may resolve `libcudf.so` from conda env instead of `cpp/build`.
- For validating local source edits before install, run with preload:

```bash
export LD_PRELOAD=$CUDF_HOME/cpp/build/libcudf.so:$CUDF_HOME/cpp/build/libcudftest_default_stream.so
./cpp/build/gtests/PARQUET_FASTLANES_TEST
```

### 7.2 FastLanes benchmark smoke

```bash
cd $CUDF_HOME/cpp/examples/parquet_io
../build.sh
./build/fastlanes_bench_delta
./build/fastlanes_bench_bitpack
```

### 7.3 Roundtrip validation with chunked C++ path

```bash
cd $CUDF_HOME/cpp/examples/parquet_io
python3 ./tools/roundtrip/parquet_io_roundtrip_check.py \
  --input CUDF-0003.parquet \
  --conversion-engine cpp \
  --cpp-binary ./build/parquet_io_chunk \
  --validator auto \
  --keep-output \
  --cpp-enable-log \
  --encoding-spec "l_orderkey:DELTA_BINARY_PACKED,\
l_partkey:DELTA_BINARY_PACKED,\
l_suppkey:DELTA_BINARY_PACKED,\
l_linenumber:DICTIONARY,\
l_quantity:DICTIONARY,\
l_extendedprice:DELTA_BINARY_PACKED,\
l_discount:DICTIONARY,\
l_tax:DICTIONARY,\
l_returnflag:FASTLANES_BITPACK,\
l_linestatus:FASTLANES_BITPACK,\
l_shipdate:DICTIONARY,\
l_commitdate:DICTIONARY,\
l_receiptdate:DELTA_BINARY_PACKED,\
l_shipinstruct:FASTLANES_BITPACK,\
l_shipmode:FASTLANES_BITPACK,\
l_comment:DICTIONARY"
```

### 7.4 TPCH100 encoding search (reproducible)

```bash
cd $CUDF_HOME/cpp/examples/parquet_io
python3 ./tools/search/search_best_parquet_encoding.py \
  --input CUDF-0003.roundtrip.parquet \
  --binary ./build/parquet_io_chunk \
  --work-dir ./reports/fastlanes/encoding_search_tpch100 \
  --compressions NONE,SNAPPY,ZSTD \
  --batch-size 2 \
  --search-scope fastlane-eligible \
  --max-iterations 2 \
  --search-skip-validation \
  --validate-best
```

Expected output artifacts:

- `cpp/examples/parquet_io/reports/fastlanes/encoding_search_tpch100/search_summary.md`
- `cpp/examples/parquet_io/reports/fastlanes/encoding_search_tpch100/search_summary.json`

## 8. Result Snapshot (Current)

From current reproducible search artifacts:

- best compression: `ZSTD`
- final validated size: around `9.12 GB`
- best FastLanes subset map in this run:
  - `l_returnflag: DICTIONARY`
  - `l_linestatus: DICTIONARY`
  - `l_shipinstruct: FASTLANES_BITPACK`
  - `l_shipmode: FASTLANES_BITPACK`

Interpretation:

- FastLanes can help selected low-cardinality INT32 columns,
- not all candidate columns benefit from FastLanes in mixed-schema TPCH workloads.

## 9. Known Gaps and Open Issues

### 9.1 Mixed-schema and mixed-encoding corner cases

Misalignment-class issues were partially mitigated by aligned staging, but broader stress coverage is still needed.

### 9.2 Logical INT32 coverage (Date32 etc.)

Current matrix is intentionally scoped:

- enabled now (INT32 physical): `INT8`, `UINT8`, `INT16`, `UINT16`, `INT32`, `UINT32`, `Date32`, `Decimal32`, `TimeMillis(duration_ms, duration_s)`
- unsupported but INT32 physical: `TimeMillis(duration_D)`
- legacy UINT cast mode (`cast_mode=0`) is treated as unsupported during decode
- fallback now: `Duration(us/ns)` and `TimeMillis(other scaled forms / negative-value semantics outside current validation scope)`

Test-backed verification anchor:

- `ParquetCpuEncoderTest.FastLanesInt32PhysicalLogicalTypeSupportMatrix`
- `ParquetCpuEncoderTest.FastLanesInt32PhysicalLogicalTypeUnsupportedMatrix`

Reason: keep decode/write semantics safe while retaining deterministic fallback for unsupported
logical classes.

### 9.3 INT64 support

INT64 support (discussed in issue `[CUDA Kernel] fastlane只能uint32, 思考简单扩展到64bit`) is not a quick patch. The generated unpack kernels are hardwired around 32-bit patterns and magic constants, so a proper extension requires deeper kernel-generation work.

### 9.4 Performance benchmark maturity

The pipeline is functionally stronger than before, but end-to-end benchmark coverage on broader real datasets is still incomplete and should follow after edge-case hardening.

## 10. Extension Playbook for Next Engineer

### 10.1 Add more logical INT32 types (recommended next step)

1. Add explicit negative-value semantic coverage for TimeMillis(duration_s) before broadening that scope.
2. Add guarded support for additional TimeMillis scaled forms only after semantic tests pass.
3. Keep strict fallback for any logical class without proven roundtrip/statistics correctness.
4. Re-run TPCH search to measure net gain/loss after any eligibility expansion.

### 10.2 INT64 integration (separate topic)

Do not treat as a small patch.

Recommended plan:

1. Confirm required FastLanes kernel generation strategy for 64-bit unpack.
2. Avoid metadata spoofing tricks that can break value-count/statistics/physical-type contracts.
3. Introduce isolated prototype path first, then add parquet metadata correctness tests.
4. Only merge after deterministic roundtrip + stats correctness are proven.

## 11. Quick Reading Order for Handover

1. `cpp/include/cudf/fastlanes/IMPLEMENTATION_NOTES.md`
2. `cpp/src/io/parquet/writer_impl.cu`
3. `cpp/src/io/parquet/page_enc.cu`
4. `cpp/src/io/parquet/page_hdr.cu`
5. `cpp/src/io/parquet/reader_impl.cpp`
6. `cpp/src/io/parquet/page_fastlanes_decode.cu`
7. `cpp/tests/io/parquet_fastlanes_test.cpp`
8. `cpp/examples/parquet_io/tools/roundtrip/parquet_io_roundtrip_check.py`
9. `cpp/examples/parquet_io/tools/search/search_best_parquet_encoding.py`

## 12. Related Documents

- FastLanes method doc:
  - `cpp/examples/parquet_io/docs/fastlanes/ENCODING_SEARCH_METHOD.md`
- nvComp handover doc (same style):
  - `cpp/examples/parquet_io/docs/nvcomp/NVCOMP_BRANCH_BASE_ANALYSIS_2026-03-28.md`
