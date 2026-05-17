# nvComp Integration Handover Report (2026-03-28)

## 1. Purpose

This report is a practical handover for the nvComp integration work in cuDF Parquet.
It focuses on:

- what was integrated,
- where code was changed and why,
- trade-offs and known limitations,
- how to reproduce results,
- how to add a new compression codec using the same integration pattern.

## 2. Scope and Branch Context

Primary development branch for this work:

- `nvcomp-25.12-rebase251221`

Real base on `upstream/main` used during development:

- commit: `1a9071e46aff06997e829e99655de58515d527ba`
- date: `2025-12-21`
- subject: `Replace thrust count_if and copy_if with CUB + pinned memory based wrappers (#20861)`

This matters because latest `upstream/main` has moved significantly; comparisons must be base-aware.

## 3. What Was Integrated

### 3.1 End-to-end nvComp codec plumbing in cuDF IO

Added nvComp codec support across cuDF IO compression/decompression path and Parquet mapping for:

- `CASCADED`
- `DEFLATE`
- `BITCOMP`
- `GDEFLATE`
- `ANS`

Core integration behavior:

1. Extend codec enums and Parquet compression IDs.
2. Map cuDF compression type to nvComp compression type.
3. Enable these codecs in generic compression/decompression capability checks.
4. Add actual nvComp batched API calls (temp-size, compress, decompress).
5. Hook Parquet writer/reader to recognize and use these codecs.

### 3.2 Parquet chunked rewrite path for large-file workflows

Implemented and iterated `parquet_io_chunk` to process Row Groups in batches.
This was used to reduce OOM pressure and create a reproducible testing harness for nvComp + encoding combinations.

### 3.3 Validation tooling

Built a validation pipeline (later Python-first) to run forward/backward roundtrip checks and compare outputs with PyArrow/DuckDB.

## 4. File-Level Handover: What Was Changed and Why

Below are the key files a second engineer should read first.

### 4.1 Codec and mapping layer

- `cpp/include/cudf/io/types.hpp`
  - Extended public `compression_type` with nvComp codecs.
  - Meaning: exposes new codec options to IO API users.

- `cpp/include/cudf/io/parquet_schema.hpp`
  - Added Parquet `Compression` enum values for nvComp codecs.
  - Meaning: writer/reader can serialize/interpret these codec IDs in Parquet metadata.

- `cpp/include/cudf/io/detail/nvcomp_adapter.hpp`
  - Extended internal nvComp compression enum with new codec entries.
  - Meaning: internal adapter layer can dispatch all nvComp codecs.

- `cpp/src/io/comp/common_internal.hpp`
  - Extended `to_nvcomp_compression()` mapping.
  - Added `LIBCUDF_NVCOMP_SNAPPY_CASCADE` switch (SNAPPY path can be redirected to CASCADED).
  - Meaning: central runtime mapping policy point for nvComp codec routing.

- `cpp/src/io/comp/common.cpp`
  - Added string names for new codec types.
  - Meaning: logging/diagnostics can print codec names clearly.

- `cpp/src/io/comp/compression.cpp`
- `cpp/src/io/comp/decompression.cpp`
  - Added support checks for new nvComp codecs in generic IO path.
  - Meaning: cuDF treats these codecs as supported in compression/decompression pipeline.

- `cpp/src/io/comp/nvcomp_adapter.cpp`
  - Added nvComp includes and batched API calls for new codecs:
    - temp-size query,
    - compress,
    - decompress,
    - alignment requirement checks for some codecs.
  - Meaning: this is the implementation core for nvComp execution.

### 4.2 Parquet writer/reader integration points

- `cpp/src/io/parquet/writer_impl.cu`
  - Added `to_parquet_compression()` mapping for nvComp codecs.
  - Meaning: writer can emit Parquet pages with these compression identifiers.

- `cpp/src/io/parquet/page_enc.cu`
  - Changed write decision to keep compressed output whenever compression succeeds (even if compressed size is larger than raw).
  - Meaning: experimental/validation-friendly behavior; no silent fallback due ratio check.
  - Trade-off: may increase file size for incompressible chunks.

- `cpp/src/io/parquet/reader_impl_chunking_utils.cu`
  - Added Parquet->codec mapping for nvComp codec IDs.
  - Added aligned compressed-page copy path before decompression (`copy_pages_to_buffer` with alignment).
  - Kept compressed-page buffer lifetime alongside decompressed buffers.
  - Meaning: mitigates misalignment/lifetime issues in chunked decode path.

- `cpp/src/io/parquet/reader_impl_chunking.cu`
  - Updated subpass setup to carry both decompressed buffers and compressed-page copy buffers.
  - Meaning: prevents dangling page pointers during chunked processing.

### 4.3 Tests and reproducible tooling

- `cpp/tests/io/parquet_writer_test.cpp`
  - Extended nvComp compression parametrization and added encoding+compression roundtrip test coverage.

- `cpp/tests/io/parquet_reader_test.cpp`
- `cpp/tests/io/parquet_chunked_reader_test.cu`
  - Extended compression test matrix with nvComp codecs.

- `cpp/examples/parquet_io/parquet_io_chunk.cpp`
  - Added row-group-wise rewrite, per-batch processing, logging, and validation-oriented flow.

- `cpp/examples/parquet_io/verify_compression_roundtrip.py`
- `cpp/examples/parquet_io/py_utils/*`
- `cpp/examples/parquet_io/compare_parquet_duckdb.sh`
- `cpp/examples/parquet_io/compare_parquet_pyarrow.sh`
  - Added practical end-to-end benchmark/verification harness.

## 5. Design Trade-offs Taken

### 5.1 V1 first, V2 partially blocked

- V1 path reached stable behavior for tested cases.
- V2 remains sensitive to alignment due variable-size page header layout and compressed payload offsets.

Trade-off:

- prioritized practical V1 progress and reproducible validation,
- deferred full V2 robustness while alignment strategy remains costly/complex.

### 5.2 Compressed-write decision policy

In `page_enc.cu`, compressed output is kept if compression succeeds, even when size is larger.

Trade-off:

- pros: preserves codec behavior for validation and debugging,
- cons: can hurt ratio on bad pages.

### 5.3 Large-file strategy

Managed memory alone was not sufficient for large mixed workloads (OOM/corruption risks).

Trade-off:

- moved toward row-group-wise chunked processing (`parquet_io_chunk`) to control memory footprint and improve observability.

### 5.4 Validation baseline policy

Direct CPU parquet vs cuDF parquet comparisons can produce false negatives (precision/type representation differences, e.g. Decimal128).

Trade-off:

- use cuDF-to-cuDF rewritten baseline before compression comparison to isolate compression integrity from representation mismatch.

## 6. Current Support Matrix (as observed in development)

| Parquet | File size | Encodings | nvComp status | Action |
| --- | ---: | --- | --- | --- |
| V1 | small (< GPU memory) | all tested | stable | none |
| V1 | large (> GPU memory) | simple (PLAIN, DELTA_BINARY_PACKED) | may OOM depending on settings | use chunked rewrite |
| V1 | large (> GPU memory) | complex (e.g. DICTIONARY-heavy) | mismatch observed in some runs | use chunked rewrite + validation |
| V2 | any | any | alignment-sensitive / unstable for some codecs | alignment-copy strategy decision needed |

## 7. Reproduction Guide

Run from repo root unless noted.

### 7.1 Build

```bash
cd $CUDF_HOME
./build.sh
./build.sh libcudf tests
ctest --test-dir ${CUDF_HOME}/cpp/build -R PARQUET_TEST

cd $CUDF_HOME/cpp/examples/parquet_io
../build.sh
```

### 7.2 Basic write command (single encoding + single codec)

```bash
./build/parquet_io input.parquet output.parquet PLAIN BITCOMP
```

### 7.3 Per-column encoding command

```bash
./build/parquet_io \
  input.parquet \
  output.parquet \
  "l_orderkey:DELTA_BINARY_PACKED,\
l_partkey:DELTA_BINARY_PACKED,\
l_suppkey:DELTA_BINARY_PACKED,\
l_linenumber:DELTA_BINARY_PACKED,\
l_quantity:DELTA_BINARY_PACKED,\
l_extendedprice:DELTA_BINARY_PACKED,\
l_discount:DELTA_BINARY_PACKED,\
l_tax:DELTA_BINARY_PACKED,\
l_returnflag:DICTIONARY,\
l_linestatus:DICTIONARY,\
l_shipdate:DELTA_BINARY_PACKED,\
l_commitdate:DELTA_BINARY_PACKED,\
l_receiptdate:DELTA_BINARY_PACKED,\
l_shipinstruct:DICTIONARY,\
l_shipmode:DICTIONARY,\
l_comment:DICTIONARY" \
  CASCADED
```

### 7.4 Managed memory mode (when needed)

```bash
LIBCUDF_USE_MANAGED_MEMORY=ON ./build/parquet_io ...
```

### 7.5 Validation/benchmark workflow

Historical branch command form:

```bash
python verify_compression_roundtrip.py CUDF-0003-20251024-184914.parquet \
  --validator=pyarrow \
  --encoding="l_orderkey:DELTA_BINARY_PACKED,\
l_partkey:DELTA_BINARY_PACKED,\
l_suppkey:DELTA_BINARY_PACKED,\
l_linenumber:DICTIONARY,\
l_quantity:DICTIONARY,\
l_extendedprice:DELTA_BINARY_PACKED,\
l_discount:DICTIONARY,\
l_tax:DICTIONARY,\
l_returnflag:DICTIONARY,\
l_linestatus:DICTIONARY,\
l_shipdate:DICTIONARY,\
l_commitdate:DICTIONARY,\
l_receiptdate:DICTIONARY" \
  -b=64
```

Important methodology note:

- first rewrite baseline with cuDF,
- then compare cuDF-produced files,
- avoid direct CPU-generated parquet vs cuDF parquet as corruption signal.

## 8. Reproduced Result Snapshot (from development logs)

Input:

- `CUDF-0003-20251024-184914.parquet`
- input size around `10.13 GiB` in one measured run

Observed summary (example run):

- `SNAPPY`: `9.12 GB`, `0.9224x`, `7.76%` saving, `10.74s`
- `CASCADED`: `9.30 GB`, `0.9407x`, `5.93%` saving, `23.92s`
- `BITCOMP`: `9.31 GB`, `0.9413x`, `5.87%` saving, `23.00s`
- `GDEFLATE`: `8.84 GB`, `0.8939x`, `10.61%` saving, `25.88s`
- `ANS`: `8.90 GB`, `0.9003x`, `9.97%` saving, `22.97s`

Operational observations:

- per-RG timing logs were added and used for diagnosis,
- single-stream batched RG submission gave strong throughput on SF100-like runs,
- rewritten cuDF parquet inputs were much faster to rewrite than non-cuDF-source inputs in tests.

## 9. Known Issues and Pending Decisions

1. Large-file mixed encoding reliability
   - symptom: mismatch or instability in some heavy mixed cases.
   - direction: chunked writer/reader path + strict validation.

2. V2 alignment sensitivity
   - symptom: misaligned-address failures for some codec/header combinations.
   - direction: explicit parse+copy-to-aligned-buffer path for compressed pages, or scope down initial support to V1.

3. Compression ratio variability
   - some nvComp codecs do not beat strong SNAPPY baseline on every dataset.
   - evaluate with workload-specific baseline and per-column encoding policy.

## 10. How to Integrate a New Compression Codec (Playbook)

Use this checklist when adding another codec via nvComp path.

1. Add enum entries
   - `cpp/include/cudf/io/types.hpp`
   - `cpp/include/cudf/io/parquet_schema.hpp`
   - `cpp/include/cudf/io/detail/nvcomp_adapter.hpp`

2. Add string/mapping logic
   - `cpp/src/io/comp/common.cpp`
   - `cpp/src/io/comp/common_internal.hpp` (`to_nvcomp_compression`)

3. Enable capability checks
   - `cpp/src/io/comp/compression.cpp`
   - `cpp/src/io/comp/decompression.cpp`

4. Implement nvComp adapter calls
   - `cpp/src/io/comp/nvcomp_adapter.cpp`
   - wire: temp-size, max-output, compress, decompress, status handling, alignment requirement handling.

5. Wire parquet writer/reader
   - `cpp/src/io/parquet/writer_impl.cu` (`to_parquet_compression`)
   - `cpp/src/io/parquet/reader_impl_chunking_utils.cu` (Parquet codec mapping and decompression handling)
   - verify buffer alignment/lifetime across subpasses.

6. Add tests
   - extend compression parametrization in:
     - `cpp/tests/io/parquet_writer_test.cpp`
     - `cpp/tests/io/parquet_reader_test.cpp`
     - `cpp/tests/io/parquet_chunked_reader_test.cu`

7. Add reproducible example path
   - `cpp/examples/parquet_io/parquet_io_chunk.cpp`
   - validation tooling under `cpp/examples/parquet_io`.

8. Validate with cuDF baseline policy
   - use cuDF-rewritten input baseline,
   - run forward/backward roundtrip plus PyArrow/DuckDB checks,
   - inspect per-RG logs for OOM/alignment hotspots.

## 11. Quick Ownership Map for Next Engineer

Start reading in this order:

1. `cpp/src/io/comp/nvcomp_adapter.cpp`
2. `cpp/src/io/comp/common_internal.hpp`
3. `cpp/src/io/parquet/writer_impl.cu`
4. `cpp/src/io/parquet/reader_impl_chunking_utils.cu`
5. `cpp/examples/parquet_io/parquet_io_chunk.cpp`
6. `cpp/examples/parquet_io/verify_compression_roundtrip.py`
7. parquet IO tests listed above

If your immediate goal is reliability, focus first on:

- V2 alignment strategy,
- large-file chunked stability with DICTIONARY-heavy cases,
- test matrix expansion for codec+encoding combinations.
