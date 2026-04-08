# INT64 FastLanes Page-Stats Workflow (SNAPPY)

This workflow captures page-level FastLanes SPLIT64 metadata for INT64 sensitivity runs,
then exports a canonical CSV and plot report.

## Scope

- Compression codec: `SNAPPY` only
- Cases:
  - baseline: all INT64 columns set to `DELTA_BINARY_PACKED`
  - one-by-one: each INT64 column toggled to `FASTLANE_BITPACK_SPLIT64`
  - aggregate control: all INT64 columns toggled to `FASTLANE_BITPACK_SPLIT64`
- Validation: C++ row-group validation in `parquet_io_chunk` remains enabled

## Preconditions

1. Build `parquet_io_chunk` from `cpp/examples/parquet_io/build/parquet_io_chunk`.
2. Debug env variables are set automatically by `run_int64_snappy_page_stats.py`.
3. Run from `cpp/examples/parquet_io`.
4. Set `CUDF_HOME`, `PARQUET_IO_SHARED_ROOT`, and `WT=$(basename "$CUDF_HOME")` in your remote shell.

## Layered Output Layout

Each run is written under:

- `${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/01_human`
- `${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/02_machine`
- `${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/03_raw`

Meaning:

- `01_human`: report markdown and PNG plots
- `02_machine`: config/index JSON summaries
- `03_raw`: CSV, logs, stdout captures, and output parquet files

## Step 1: Run Cases And Emit Manifest

```bash
python3 ./tools/search/run_int64_snappy_page_stats.py \
  --input <input.parquet> \
  --cpp-binary ./build/parquet_io_chunk \
  --output-dir "${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats" \
  --batch-size 2
```

Outputs under `run_<timestamp>/`:

- `02_machine/run_config.json`
- `02_machine/run_index.json`
- `01_human/run_summary.md`
- `03_raw/run_manifest.csv`
- `03_raw/case_summary.csv`
- `03_raw/logs/*.log`
- `03_raw/cases/*.parquet`

## Step 2: Extract Page-Level CSV

```bash
python3 ./tools/search/extract_fastlanes_page_stats.py \
  --manifest-csv "${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/03_raw/run_manifest.csv" \
  --machine-summary-json "${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/02_machine/extract_summary.json"
```

Default output:

- `03_raw/page_stats_int64_fastlanes.csv`

Core schema columns include:

- identity: `run_id`, `case_name`, `compression`, `encoding_map_hash`
- location: `column_name`, `column_index`, `row_group_index`, `page_index`, `chunk_id`
- internals: `layout_mode`, `cast_mode`, `component_bitwidth_low`, `component_bitwidth_high`
- counts/sizes: `original_count`, `padded_count`, `vector_count`, `body_size_bytes`, `blob_size_bytes`
- derived: `encoded_bytes_per_value`, `padding_overhead_rows`, `padding_overhead_pct`

## Step 3: Generate Plots And Markdown Report

```bash
Rscript ./tools/search/plot_int64_fastlanes_page_stats.R \
  --case-summary-csv "${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/03_raw/case_summary.csv" \
  --page-stats-csv "${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/03_raw/page_stats_int64_fastlanes.csv" \
  --output-dir "${PARQUET_IO_SHARED_ROOT}/artifacts/${WT}/fastlanes/snappy_int64_page_stats/run_<timestamp>/01_human/r_report"
```

Generated assets:

- `bitwidth_distribution.png`
- `encoded_bytes_per_value.png`
- `padding_overhead.png`
- `min_value_drift.png`
- `int64_fastlanes_page_report.md`

## Notes

- `row_group_index` and `column_index` are derived from `chunk_id` and parquet column count.
- `page_index` is the in-chunk FastLanes page ordinal.
- Use the same input and run environment across cases to keep comparisons fair.
