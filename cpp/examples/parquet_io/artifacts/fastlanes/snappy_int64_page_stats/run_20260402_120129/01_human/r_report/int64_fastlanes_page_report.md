# INT64 FastLanes Page-Level Report

## Inputs
- case summary: ./reports/fastlanes/snappy_int64_page_stats/run_20260402_120129/case_summary.csv
- page stats: ./reports/fastlanes/snappy_int64_page_stats/run_20260402_120129/page_stats_int64_fastlanes.csv

## Key Plots
- bitwidth_distribution.png
- encoded_bytes_per_value.png
- padding_overhead.png
- min_value_drift.png

## Case-Level Delta Table

| case_name | status | output_size_bytes | size_delta_bytes | size_delta_pct | elapsed_seconds | time_delta_seconds | time_delta_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_delta | PASS | 10890136016 | 0 | 0 | 113.1291 | 0 | 0 |
| l_orderkey_fastlane | PASS | 11412066996 | 521930980 | 4.792695 | 125.8456 | 12.716523 | 11.240722 |
| l_partkey_fastlane | PASS | 10832501370 | -57634646 | -0.529237 | 125.2362 | 12.107128 | 10.70205 |
| l_suppkey_fastlane | PASS | 10798244856 | -91891160 | -0.843802 | 124.7064 | 11.577324 | 10.233731 |
| l_linenumber_fastlane | PASS | 10942236028 | 52100012 | 0.478415 | 123.5273 | 10.398263 | 9.191505 |
| l_quantity_fastlane | PASS | 10288904264 | -601231752 | -5.520884 | 112.1409 | -0.988137 | -0.87346 |
| l_extendedprice_fastlane | PASS | 12154990033 | 1264854017 | 11.614676 | 119.9468 | 6.817751 | 6.026525 |
| l_discount_fastlane | PASS | 10799364331 | -90771685 | -0.833522 | 113.264 | 0.134995 | 0.119329 |
| l_tax_fastlane | PASS | 10849397751 | -40738265 | -0.374084 | 113.7372 | 0.608183 | 0.537601 |
| all_int64_fastlane | PASS | 11846703457 | 956567441 | 8.783797 | 165.8376 | 52.70853 | 46.591506 |

## Notes
- Page index is the in-chunk FastLanes page ordinal.
- Row-group and column are inferred from chunk_id and parquet column count.
- Compare case deltas with page-level bitwidth and padding trends to explain SNAPPY outcomes.
