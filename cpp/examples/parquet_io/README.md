# Parquet IO Examples: Structure and Script Guide

This directory contains C++ parquet examples plus helper tooling for validation,
encoding search, and FastLanes page-level analysis.

The goal of this README is to make each script easy to understand:
- what it expects as input
- what it does
- what it writes
- where outputs should live

## High-Level Structure

- `build/`
  - CMake output (not source)
  - expected binary: `build/parquet_io_chunk`
- `tools/`
  - automation scripts (Python/R/shell)
  - `tools/roundtrip/`: conversion + validation helpers
  - `tools/search/`: search and page-level analytics helpers
- `docs/`
  - human-facing documentation and workflow notes

Generated outputs are external-only and should be written under:

- `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/...`
- `${PARQUET_IO_SHARED_ROOT}/reports/<worktree>/...`

## Layered Artifact Convention

For all new generated runs, use a 3-layer layout under
`${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/`:

- `01_human/`
  - concise human-readable outputs
  - markdown summaries and plots (`.png`)
- `02_machine/`
  - machine-oriented metadata
  - run indices, config json, extraction summary json
- `03_raw/`
  - raw generated data and bulky artifacts
  - csv tables, logs, stdout captures, parquet outputs

Example run directory:

```text
${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/fastlanes/snappy_int64_page_stats/run_YYYYMMDD_HHMMSS/
  01_human/
    run_summary.md
    r_report/
      *.png
      int64_fastlanes_page_report.md
  02_machine/
    run_config.json
    run_index.json
    extract_summary.json
  03_raw/
    run_manifest.csv
    case_summary.csv
    page_stats_int64_fastlanes.csv
    logs/*.log
    stdout/*.txt
    cases/*.parquet
```

## Script Catalog

### tools/search/search_best_parquet_encoding.py
- Effect: greedy search for encoding + compression combinations
- Input: parquet file + `parquet_io_chunk` binary
- Output: `search_summary.json`, `search_summary.md`, trial logs, optional trial parquet files
- Typical use:

```bash
python3 ./tools/search/search_best_parquet_encoding.py --help
```

### tools/search/run_int64_snappy_page_stats.py
- Effect: runs baseline + one-by-one + all-fastlane INT64 sensitivity cases
- Input: parquet file, `parquet_io_chunk`, roundtrip helper script
- Output: layered run folder (`01_human`, `02_machine`, `03_raw`)
- Notes:
  - automatically sets `FLS_DEBUG_WORKLOAD=1` and `FLS_DEBUG_HEADER=1`
  - writes case parquet files in `03_raw/cases/`

```bash
python3 ./tools/search/run_int64_snappy_page_stats.py --help
```

### tools/search/extract_fastlanes_page_stats.py
- Effect: parses FastLanes debug lines and builds page-level INT64 CSV
- Input: run manifest CSV (typically `03_raw/run_manifest.csv`)
- Output: `page_stats_int64_fastlanes.csv` (raw), optional extraction summary json (machine)

```bash
python3 ./tools/search/extract_fastlanes_page_stats.py --help
```

### tools/search/plot_int64_fastlanes_page_stats.R
- Effect: produces human-facing plots + markdown report from case/page CSVs
- Input: `case_summary.csv`, `page_stats_int64_fastlanes.csv`
- Output: plot PNGs + report markdown (recommended in `01_human/r_report`)

```bash
Rscript ./tools/search/plot_int64_fastlanes_page_stats.R --help
```

### tools/roundtrip/parquet_io_roundtrip_check.py
- Effect: chunked C++ or cudf roundtrip conversion and validation
- Input: parquet file (+ compare target in compare mode)
- Output: converted parquet, validator result, optional C++ logs
- Notes:
  - if `--cpp-log-file` is set and logs are not streamed, captured stdout/stderr
    are appended to the C++ log so debug parsing is possible.

```bash
python3 ./tools/roundtrip/parquet_io_roundtrip_check.py --help
```

### tools/roundtrip/verify_compression_roundtrip.py
- Effect: tests codec roundtrip behavior and validates data integrity
- Input: parquet file + optional codec/config flags
- Output: timestamped comparison logs and summary tables

```bash
python3 ./tools/roundtrip/verify_compression_roundtrip.py --help
```

### tools/roundtrip/compare_parquet_duckdb.sh
- Effect: compares two parquet files with DuckDB `EXCEPT` logic
- Output: row-count and symmetric-difference verdict

### tools/roundtrip/compare_parquet_pyarrow.sh
- Effect: compares two parquet files with PyArrow/Pandas merge diff
- Output: schema + row-difference verdict

## Recommended End-to-End INT64 Workflow

Run from `cpp/examples/parquet_io`:

```bash
python3 ./tools/search/run_int64_snappy_page_stats.py \
  --input <input.parquet> \
  --cpp-binary ./build/parquet_io_chunk \
  --output-dir ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/fastlanes/snappy_int64_page_stats

python3 ./tools/search/extract_fastlanes_page_stats.py \
  --manifest-csv ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/fastlanes/snappy_int64_page_stats/run_<id>/03_raw/run_manifest.csv \
  --machine-summary-json ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/fastlanes/snappy_int64_page_stats/run_<id>/02_machine/extract_summary.json

Rscript ./tools/search/plot_int64_fastlanes_page_stats.R \
  --case-summary-csv ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/fastlanes/snappy_int64_page_stats/run_<id>/03_raw/case_summary.csv \
  --page-stats-csv ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/fastlanes/snappy_int64_page_stats/run_<id>/03_raw/page_stats_int64_fastlanes.csv \
  --output-dir ${PARQUET_IO_SHARED_ROOT}/artifacts/$(basename ${CUDF_HOME})/fastlanes/snappy_int64_page_stats/run_<id>/01_human/r_report
```

## Authoring Rules

If you add a new script or generated artifact flow, follow
`docs/PROJECT_STRUCTURE_RULES.md`.
