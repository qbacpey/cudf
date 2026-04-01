# INT64 SNAPPY DELTA vs FASTLANES Sensitivity Report (2026-04-02)

## Objective

Measure the effect of switching INT64 columns from `DELTA_BINARY_PACKED` to `FASTLANES_BITPACK` under a single compression codec (`SNAPPY`) on a cuDF-produced parquet source.

## Input and Constraints

- Input parquet: `CUDF-0003.roundtrip.parquet`
- Conversion engine: `parquet_io_chunk` through `parquet_io_roundtrip_check.py`
- Compression: `SNAPPY` only
- Validation: C++ internal validation enabled for every run
- Fixed non-INT64 encoding map:
  - `l_quantity:DICTIONARY`
  - `l_extendedprice:DELTA_BINARY_PACKED`
  - `l_discount:DICTIONARY`
  - `l_tax:DICTIONARY`
  - `l_returnflag:DICTIONARY`
  - `l_linestatus:DICTIONARY`
  - `l_shipdate:DICTIONARY`
  - `l_commitdate:DICTIONARY`
  - `l_receiptdate:DELTA_BINARY_PACKED`
  - `l_shipinstruct:DICTIONARY`
  - `l_shipmode:DICTIONARY`
  - `l_comment:DICTIONARY`

## Experiment Design

Baseline run:
- All 4 INT64 columns use `DELTA_BINARY_PACKED`.

Second-pass sensitivity runs:
- Toggle one INT64 column at a time to `FASTLANES_BITPACK`.
- Keep the other 3 INT64 columns as `DELTA_BINARY_PACKED`.
- Keep all non-INT64 columns fixed.

Additional aggregate control:
- Toggle all 4 INT64 columns to `FASTLANES_BITPACK`.

## Results

Baseline:
- output size: `10,165,701,676` bytes
- convert time: `62,223.83` ms

Per-case deltas vs baseline:

| Case | Output Size (bytes) | Size Delta (bytes) | Size Delta (%) | Convert Time (ms) | Time Delta (ms) |
|---|---:|---:|---:|---:|---:|
| baseline_delta | 10,165,701,676 | 0 | 0.0000% | 62,223.83 | 0.00 |
| l_orderkey_fastlane | 10,687,653,132 | +521,951,456 | +5.1344% | 75,000.00 | +12,776.17 |
| l_partkey_fastlane | 10,108,077,792 | -57,623,884 | -0.5668% | 74,474.87 | +12,251.04 |
| l_suppkey_fastlane | 10,073,799,638 | -91,902,038 | -0.9040% | 73,643.15 | +11,419.32 |
| l_linenumber_fastlane | 10,217,832,300 | +52,130,624 | +0.5128% | 72,689.02 | +10,465.19 |
| all_int64_fastlane | 10,590,257,791 | +424,556,115 | +4.1764% | 109,846.75 | +47,622.92 |

Human-readable size deltas vs baseline:
- `l_orderkey_fastlane`: +497.77 MiB
- `l_partkey_fastlane`: -54.95 MiB
- `l_suppkey_fastlane`: -87.64 MiB
- `l_linenumber_fastlane`: +49.72 MiB
- `all_int64_fastlane`: +404.89 MiB

## Interpretation

Observed behavior is mixed by column:
- `l_partkey` and `l_suppkey` improved with FASTLANES under SNAPPY.
- `l_orderkey` and `l_linenumber` regressed with FASTLANES under SNAPPY.
- Enabling FASTLANES for all four INT64 columns together is worse than all-DELTA baseline for both size and runtime.

Likely reason:
- Column value distributions differ. Some columns benefit from split32 bitpacking + SNAPPY interaction, while others are better handled by DELTA patterns.
- FASTLANES also increased conversion time in every tested case, often substantially.

## Practical Recommendation (SNAPPY-only)

For this dataset and fixed encoding map:
- Keep `l_orderkey` and `l_linenumber` on `DELTA_BINARY_PACKED`.
- Consider `FASTLANES_BITPACK` for `l_partkey` and `l_suppkey` if the small size win is worth the extra conversion time.
- Avoid all-INT64 FASTLANES in this SNAPPY setup.

## Repro Notes

All runs were executed on remote GPU server `qchen@fng01` with:
- `CUDF_HOME=/home/qchen/GPUFileFormat-cudf`
- `conda activate cudf_dev`
- roundtrip tool: `cpp/examples/parquet_io/tools/roundtrip/parquet_io_roundtrip_check.py`

Output artifacts are under:
- `cpp/examples/parquet_io/reports/fastlanes/snappy_int64_delta_vs_fastlanes/`
