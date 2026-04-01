# Parquet Encoding Search Method (Chunked, Reproducible)

This document describes the reproducible method used to find a strong parquet
encoding + compression combination for large datasets (for example TPC-H SF100)
without OOM.

## Goal

Find an encoding map plus compression codec that minimizes output size while
remaining valid under cudf chunked rewrite/read validation.

Compression search space is intentionally constrained to:

- NONE (uncompressed)
- SNAPPY
- ZSTD

## Why This Method

A full exhaustive search over all columns and all encoding choices is too large.
This workflow uses deterministic greedy coordinate descent:

1. Start from a metadata-driven baseline map.
2. For each compression, optimize one column at a time.
3. Keep changes only if output size improves.
4. Repeat for a small number of passes.
5. Re-run the winner with C++ validation enabled.

This keeps runtime practical and is easy to re-run after feature changes.

## Tooling

- C++ chunked binary: cpp/examples/parquet_io/build/parquet_io_chunk
- Search driver: cpp/examples/parquet_io/tools/search/search_best_parquet_encoding.py

The driver uses parquet_io_chunk for all rewrite trials, so memory behavior is
chunked and suitable for very large files.

## FastLanes Eligibility Rule (Current)

By default, the search script treats FASTLANES_BITPACK as a candidate only for:

- Physical INT32
- Logical INT32

Other INT32 logical types (for example Date32) are intentionally excluded for now.
After support is implemented, enable:

- --allow-future-fastlanes

This makes the same workflow automatically test those columns as well.

## Recommended Procedure for TPCH100

Run from cpp/examples/parquet_io on the GPU host.

1) Full search (fastlane-eligible columns only, all 3 compressions):

python3 ./tools/search/search_best_parquet_encoding.py \
  --input CUDF-0003.roundtrip.parquet \
  --binary ./build/parquet_io_chunk \
  --work-dir ./artifacts/fastlanes/encoding_search_tpch100/run_<timestamp>/03_raw \
  --compressions NONE,SNAPPY,ZSTD \
  --batch-size 2 \
  --search-scope fastlane-eligible \
  --max-iterations 2 \
  --search-skip-validation \
  --validate-best

1) Optional wider search after enabling more FastLanes logical types:

python3 ./tools/search/search_best_parquet_encoding.py \
  --input CUDF-0003.roundtrip.parquet \
  --binary ./build/parquet_io_chunk \
  --work-dir ./artifacts/fastlanes/encoding_search_tpch100_future/run_<timestamp>/03_raw \
  --compressions NONE,SNAPPY,ZSTD \
  --batch-size 2 \
  --search-scope all \
  --allow-future-fastlanes \
  --max-iterations 2 \
  --search-skip-validation \
  --validate-best

## Outputs

For each run, the script writes under --work-dir (recommended under `03_raw`):

- search_summary.json: machine-readable result
- search_summary.md: human summary + reproduction command
- trial_*.log: logs from parquet_io_chunk

If --keep-trial-files is not set, non-best trial parquet files are removed.

## Notes on Runtime and Stability

- Use --batch-size 1 or 2 when memory is tight.
- Use --search-skip-validation for speed during trials.
- Always keep --validate-best enabled so final winner is validated in C++.
- If a trial fails, inspect the corresponding trial_*.log file.

## Handover Checklist

When handing over to another engineer:

1. Keep this document with the script.
2. Archive the full work-dir for the final search run.
3. Share the generated search_summary.md reproduction command.
4. Re-run after any FastLanes eligibility/decoder logic change.

## Related Workflow: INT64 Page Analytics

For page-level SPLIT32 metadata extraction and reporting on SNAPPY INT64 sensitivity
runs, see:

- `cpp/examples/parquet_io/docs/fastlanes/INT64_PAGE_STATS_WORKFLOW.md`
