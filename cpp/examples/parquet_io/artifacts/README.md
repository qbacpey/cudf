# Generated Artifacts

This directory stores generated outputs using a layered convention:

- `01_human`: concise markdown + plot images for people
- `02_machine`: machine-readable metadata/index/config JSON
- `03_raw`: raw logs, CSV, stdout/stderr captures, parquet intermediates

Example:

```text
artifacts/fastlanes/snappy_int64_page_stats/run_YYYYMMDD_HHMMSS/
  01_human/
  02_machine/
  03_raw/
```

Do not place source code here.
