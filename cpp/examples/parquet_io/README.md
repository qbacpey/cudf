# Parquet IO Examples: Layout Guide

This folder now separates C++ examples from temporary analysis scripts and generated reports.

## Directory Layout

- `build/`
  - CMake build output (binary target `parquet_io_chunk`)
- `tools/roundtrip/`
  - Roundtrip and validation scripts:
    - `parquet_io_roundtrip_check.py`
    - `verify_compression_roundtrip.py`
    - `compare_parquet_duckdb.sh`
    - `compare_parquet_pyarrow.sh`
    - `py_utils/`
- `tools/search/`
  - Encoding/compression search driver:
    - `search_best_parquet_encoding.py`
- `docs/fastlanes/`
  - FastLanes-specific method and progress documents
- `docs/nvcomp/`
  - nvComp branch analysis and integration notes
- `reports/fastlanes/`
  - Generated run artifacts (`search_summary.md/json`, logs, etc.)

## Quick Commands

Run from `cpp/examples/parquet_io`.

```bash
python3 ./tools/search/search_best_parquet_encoding.py --help
python3 ./tools/roundtrip/parquet_io_roundtrip_check.py --help
python3 ./tools/roundtrip/verify_compression_roundtrip.py --help
```

## Notes

- Keep C++ source/example files at folder root.
- Put Python/shell helper scripts under `tools/`.
- Put generated reports under `reports/` and avoid committing large temporary files.
