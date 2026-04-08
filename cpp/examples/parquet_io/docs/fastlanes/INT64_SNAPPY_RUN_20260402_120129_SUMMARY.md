# INT64 SNAPPY Sensitivity Summary (run_20260402_120129)

This note summarizes one executed run and is intentionally concise.

## Data Source

- Raw case table:
  - `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/fastlanes/snappy_int64_page_stats/run_20260402_120129/03_raw/case_summary.csv`
- Raw page table:
  - `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/fastlanes/snappy_int64_page_stats/run_20260402_120129/03_raw/page_stats_int64_fastlanes.csv`
- Raw logs:
  - `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/fastlanes/snappy_int64_page_stats/run_20260402_120129/03_raw/logs/*.log`

## Key Results

Baseline:
- size: `10890136016` bytes
- time: `113.129055` sec

One-by-one INT64 toggles:
- best size gain: `l_quantity_fastlane` (`-601231752` bytes, `-5.520884%`)
- worst size regression: `l_extendedprice_fastlane` (`+1264854017` bytes, `+11.614676%`)

Aggregate control:
- `all_int64_fastlane`: `+956567441` bytes (`+8.783797%`), `+52.708530` sec

## Caveats

- Several cases include warnings:
  - `FASTLANE_BITPACK_SPLIT64 encoding is unsupported for this logical type; the requested encoding will be ignored`
- Therefore, not every requested FastLanes mapping is guaranteed to be applied exactly as requested.

## Interpretation Boundaries

- Case-level size/time deltas are directly measured and reliable for this run.
- Mechanistic interpretation for unsupported logical types should remain cautious.
