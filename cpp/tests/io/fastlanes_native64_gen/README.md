# Native64 Generation Toolchain (FL64-R4)

This folder contains test-only generation and validation tooling for the native64 fastlanes test path.

## Scope

- Purpose: generate and validate native64 test artifacts for `PARQUET_FASTLANES_TEST`.
- Non-purpose: this folder is not a production runtime implementation path.

## File Map

### Inputs and generators

- `spec/generate_native64_bw_table.py`
  - Generates deterministic bw metadata for `bw=0..64`.
  - Output: `spec/native64_bw_table.json`.

- `tools/generate_native64_bw_kernels.py`
  - Reads spec + template and emits generated kernel include.
  - Input: `spec/native64_bw_table.json`, `templates/native64_bw_kernels.inl.tmpl`.
  - Output: `generated/native64_bw_kernels.inl`.
  - Optional output: `generated/native64_bw_anchor_snapshot.json`.

### Configuration and execution helpers

- `config/parity_matrix.json`
  - Defines filter names and matrix contract (anchors/full-sweep/stability).

- `tools/run_parity_matrix.py`
  - Orchestrates selected gtest filters and optional ctest checkpoint.
  - Emits a machine-readable summary JSON.

### Outputs consumed by tests

- `generated/native64_bw_kernels.inl`
  - Included by `cpp/tests/io/parquet_fastlanes_native64_generated_test.cu`.

## Dependency Flow

1. Generate spec JSON.
2. Generate kernel include from spec + template.
3. Build and run `PARQUET_FASTLANES_TEST`.
4. Optionally run matrix helper to produce summary JSON evidence.

## Typical Workflows

### A) Run existing tests only (no regeneration)

Use this when generated artifacts are already committed and you only want validation.

```bash
cpp/build/gtests/PARQUET_FASTLANES_TEST \
  --gtest_filter=ParquetFastLanesNative64GeneratedTest.*
```

### B) Regenerate deterministic artifacts

Use this when you changed generation logic, template, or spec rules.

```bash
python3 cpp/tests/io/fastlanes_native64_gen/spec/generate_native64_bw_table.py \
  --out cpp/tests/io/fastlanes_native64_gen/spec/native64_bw_table.json

python3 cpp/tests/io/fastlanes_native64_gen/tools/generate_native64_bw_kernels.py \
  --spec cpp/tests/io/fastlanes_native64_gen/spec/native64_bw_table.json \
  --template cpp/tests/io/fastlanes_native64_gen/templates/native64_bw_kernels.inl.tmpl \
  --out cpp/tests/io/fastlanes_native64_gen/generated/native64_bw_kernels.inl \
  --anchor-out cpp/tests/io/fastlanes_native64_gen/generated/native64_bw_anchor_snapshot.json
```

### C) Produce matrix summary evidence

```bash
python3 cpp/tests/io/fastlanes_native64_gen/tools/run_parity_matrix.py \
  --gtest-binary cpp/build/gtests/PARQUET_FASTLANES_TEST \
  --run-tag rt_YYYYMMDD_a \
  --report-json /path/to/r4_matrix_summary.json \
  --phase all \
  --config cpp/tests/io/fastlanes_native64_gen/config/parity_matrix.json \
  --with-stability \
  --with-bw37-guard \
  --with-ctest
```

## Handoff Notes

- Keep `generated/native64_bw_kernels.inl` committed if you expect others to run tests without local regeneration.
- If regeneration scripts are removed, future updates to generated include become manual and error-prone.
- If matrix tooling is removed, tests still run, but structured evidence/report generation is lost.
