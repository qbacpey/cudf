# Parquet IO Project Structure Rules

This file is both a human checklist and an instruction contract for AI/LLM agents.

## Goal

Keep this example folder understandable by separating:

1. Human-facing docs and visuals
2. Machine-facing metadata
3. Raw generated data

## Required Layout

Shared output root top-level policy:

- `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/...`
- `${PARQUET_IO_SHARED_ROOT}/reports/<worktree>/...`
- Do not use `${PARQUET_IO_SHARED_ROOT}/docs` or `${PARQUET_IO_SHARED_ROOT}/tools` for source content.

For any new generated run output, use:

- `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/<topic>/<workflow>/run_<timestamp>/01_human/`
- `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/<topic>/<workflow>/run_<timestamp>/02_machine/`
- `${PARQUET_IO_SHARED_ROOT}/artifacts/<worktree>/<topic>/<workflow>/run_<timestamp>/03_raw/`

### Layer meaning

- `01_human`
  - short markdown summaries
  - plot PNGs
  - content intended for direct reading
- `02_machine`
  - JSON indices, configs, compact summaries
  - easy to parse and trace provenance
- `03_raw`
  - full logs, csv, stdout/stderr captures, parquet intermediates
  - bulky files and low-level details

## Script Placement Rules

- New conversion/validation helpers: `tools/roundtrip/`
- New search/analysis helpers: `tools/search/`
- Reusable helper modules for roundtrip: `tools/roundtrip/py_utils/`
- Static explanation docs: `docs/`

## Script Documentation Contract

Every new script must include at the top:

1. What the script does
2. Required inputs
3. Produced outputs
4. One concrete usage example

For Python and R scripts, make `--help` informative.

## Output Contract

When a script writes files, it should:

1. Print exact output paths
2. Prefer writing machine JSON indices in `02_machine`
3. Avoid mixing raw CSV/logs into `01_human`

## Backward Compatibility

Legacy in-repo outputs under `cpp/examples/parquet_io/artifacts` and
`cpp/examples/parquet_io/reports` are deprecated and should not be regenerated.
New workflows should always write to `${PARQUET_IO_SHARED_ROOT}`.

## LLM Instruction Block

If you are an LLM editing this folder:

1. Keep source content in `docs/` and `tools/`; write generated outputs only to `${PARQUET_IO_SHARED_ROOT}`.
2. Never mirror source files into `${PARQUET_IO_SHARED_ROOT}/docs` or `${PARQUET_IO_SHARED_ROOT}/tools`.
3. Keep human-readable summaries short and in `01_human`.
4. Put parsable metadata in `02_machine` as JSON.
5. Put raw logs/CSV/parquet in `03_raw`.
6. Update `README.md` whenever you add or rename scripts.
