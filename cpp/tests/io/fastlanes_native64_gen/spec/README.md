# Native64 bw Spec Table (FL64-R4)

This directory contains the canonical test-only metadata table for native64 bitwidths 0..64.

For a full handoff map of generation scripts, configs, and execution flow, see:

- `cpp/tests/io/fastlanes_native64_gen/README.md`

## Source of truth

- Generator script: `generate_native64_bw_table.py`
- Generated artifact: `native64_bw_table.json`

## Contract

Each row stores:

- `bw`
- `words_per_lane`
- `words_per_vector`
- `cross_boundary_map` with predicate and crossing indices

The file also stores a deterministic `rows_sha256` fingerprint over the canonical rows payload.

## Regenerate

```bash
/usr/bin/python3 cpp/tests/io/fastlanes_native64_gen/spec/generate_native64_bw_table.py \
  --out cpp/tests/io/fastlanes_native64_gen/spec/native64_bw_table.json
```

## Check up to date

```bash
/usr/bin/python3 cpp/tests/io/fastlanes_native64_gen/spec/generate_native64_bw_table.py \
  --out cpp/tests/io/fastlanes_native64_gen/spec/native64_bw_table.json \
  --check
```
