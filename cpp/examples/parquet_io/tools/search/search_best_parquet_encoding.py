#!/usr/bin/env python3
"""Search for a strong Parquet encoding/compression combination using parquet_io_chunk.

This script is designed for large datasets (for example TPC-H SF100) and uses
chunked rewrite via the existing C++ binary to avoid OOM behavior.

Search strategy:
1) Build candidate encodings from Parquet physical/logical metadata.
2) For each compression codec, start from a deterministic baseline map.
3) Run greedy coordinate descent over selected columns.
4) Keep the best result by file size.
5) Optionally run one final C++ validation pass on the winner.

Default scope optimizes INT32 physical columns that are currently FastLanes-supported.
You can expand to all columns or a custom list.

Input:
- --input parquet file
- --binary parquet_io_chunk executable

Effect:
- Runs multiple trial rewrites, compares output parquet size, and keeps the best map.

Outputs:
- search_summary.json and search_summary.md in --work-dir
- trial_*.log (and optionally trial parquet files)

Recommended layered storage:
- Write --work-dir under 03_raw for generated machine data.

Usage example:
    python3 ./tools/search/search_best_parquet_encoding.py \
        --input ./CUDF-0003.roundtrip.parquet \
        --binary ./build/parquet_io_chunk \
        --work-dir ${PARQUET_IO_SHARED_ROOT}/artifacts/${WORKTREE}/fastlanes/encoding_search_runs/run_<id>/03_raw
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyArrow is required for schema inspection. Install with: pip install pyarrow"
    ) from exc


UNCOMPRESSED_ALIASES = {"UNCOMPRESSED", "UNCOMPRESS", "NONE"}
VALID_COMPRESSIONS = {"NONE", "SNAPPY", "ZSTD"}

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[5]


def _resolve_worktree_name() -> str:
    cudf_home = os.environ.get("CUDF_HOME", "").strip()
    if cudf_home:
        return Path(cudf_home).name
    return REPO_ROOT.name


def _default_work_dir() -> str:
    shared_root = os.environ.get("PARQUET_IO_SHARED_ROOT", "").strip()
    if not shared_root:
        return "./artifacts/fastlanes/encoding_search_runs"

    return str(
        Path(shared_root).expanduser()
        / "artifacts"
        / _resolve_worktree_name()
        / "fastlanes"
        / "encoding_search_runs"
    )


@dataclass
class ColumnInfo:
    name: str
    physical_type: str
    logical_type: str
    converted_type: str


@dataclass
class TrialResult:
    compression: str
    encoding_spec: str
    output_size_bytes: int
    elapsed_seconds: float
    return_code: int
    output_path: str
    log_path: str
    validated: bool


@dataclass
class SearchSummary:
    input_file: str
    binary: str
    batch_size: int
    search_scope: str
    compressions: List[str]
    best_result: TrialResult
    best_map: Dict[str, str]
    per_compression_best: Dict[str, TrialResult]


def _normalize_compressions(raw: str) -> List[str]:
    items = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not items:
        raise ValueError("No compressions provided")

    normalized: List[str] = []
    for item in items:
        if item in UNCOMPRESSED_ALIASES:
            item = "NONE"
        if item not in VALID_COMPRESSIONS:
            raise ValueError(
                f"Unsupported compression '{item}'. Allowed: {sorted(VALID_COMPRESSIONS)}"
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def _load_columns(input_file: Path) -> List[ColumnInfo]:
    pf = pq.ParquetFile(str(input_file))
    schema = pf.schema
    cols: List[ColumnInfo] = []
    for i in range(len(schema)):
        c = schema.column(i)
        cols.append(
            ColumnInfo(
                name=c.name,
                physical_type=str(c.physical_type),
                logical_type=str(c.logical_type),
                converted_type=str(c.converted_type),
            )
        )
    return cols


def _is_fastlanes_eligible_int32(col: ColumnInfo) -> bool:
    if col.physical_type != "INT32":
        return False

    # Supported INT32 physical logical classes in current FastLanes path.
    if (
        _is_int_subtype_8_16_like(col)
        or _is_date32_like(col)
        or _is_decimal_like(col)
        or _is_time_millis_like(col)
    ):
        return True

    # Many parquet writers store plain int32 columns without logical/converted annotations.
    if col.converted_type == "NONE" and col.logical_type == "None":
        return True
    if col.converted_type == "INT_32":
        return True
    return "Int(bitWidth=32" in col.logical_type


def _is_date32_like(col: ColumnInfo) -> bool:
    return col.converted_type == "DATE" or "Date" in col.logical_type


def _is_decimal_like(col: ColumnInfo) -> bool:
    return "Decimal" in col.logical_type or col.converted_type == "DECIMAL"


def _is_int_subtype_8_16_like(col: ColumnInfo) -> bool:
    if col.converted_type in {"INT_8", "INT_16", "UINT_8", "UINT_16"}:
        return True

    logical = col.logical_type.replace(" ", "")
    return "Int(bitWidth=8" in logical or "Int(bitWidth=16" in logical


def _is_time_millis_like(col: ColumnInfo) -> bool:
    logical_lower = col.logical_type.lower()
    return col.converted_type == "TIME_MILLIS" or (
        "time" in logical_lower and ("millis" in logical_lower or "milliseconds" in logical_lower)
    )


def _candidate_encodings(col: ColumnInfo, allow_future_fastlanes: bool = False) -> List[str]:
    if _is_fastlanes_eligible_int32(col):
        return ["FASTLANE_BITPACK_RAW", "DICTIONARY", "DELTA_BINARY_PACKED"]

    # Keep this hook for future extension once more INT32 logical classes are validated.
    if allow_future_fastlanes and col.physical_type == "INT32":
        return ["FASTLANE_BITPACK_RAW", "DICTIONARY", "DELTA_BINARY_PACKED"]

    if col.physical_type == "INT32":
        return ["DICTIONARY", "DELTA_BINARY_PACKED"]
    if col.physical_type == "INT64":
        if _is_decimal_like(col):
            return ["DICTIONARY", "DELTA_BINARY_PACKED"]
        return ["DELTA_BINARY_PACKED", "DICTIONARY"]
    if col.physical_type in {"BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY"}:
        return ["DICTIONARY", "DELTA_BYTE_ARRAY"]
    return ["DEFAULT"]


def _default_map(columns: List[ColumnInfo], allow_future_fastlanes: bool = False) -> Dict[str, str]:
    encoding_map: Dict[str, str] = {}
    for col in columns:
        candidates = _candidate_encodings(col, allow_future_fastlanes=allow_future_fastlanes)
        encoding_map[col.name] = candidates[0]
    return encoding_map


def _map_to_spec(encoding_map: Dict[str, str], ordered_columns: List[str]) -> str:
    return ",".join(f"{c}:{encoding_map[c]}" for c in ordered_columns)


def _run_chunk_rewrite(
    *,
    binary: Path,
    input_file: Path,
    output_file: Path,
    log_file: Path,
    encoding_spec: str,
    compression: str,
    batch_size: int,
    skip_validation: bool,
    show_chunk_logs: bool,
) -> TrialResult:
    cmd = [
        str(binary),
        str(input_file),
        str(output_file),
        encoding_spec,
        compression,
        f"--batch-size={batch_size}",
        "--enable-log",
        f"--log-file={log_file}",
    ]
    if skip_validation:
        cmd.append("--skip-validation")

    t0 = time.perf_counter()
    run_kwargs = {"text": True, "check": False}
    if show_chunk_logs:
        proc = subprocess.run(cmd, **run_kwargs)
    else:
        proc = subprocess.run(cmd, capture_output=True, **run_kwargs)
        # Keep process output alongside the C++ log file for quick diagnosis.
        if proc.stdout:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n=== STDOUT CAPTURE ===\n")
                f.write(proc.stdout)
        if proc.stderr:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n=== STDERR CAPTURE ===\n")
                f.write(proc.stderr)

    elapsed = time.perf_counter() - t0
    size_bytes = output_file.stat().st_size if output_file.exists() else -1

    return TrialResult(
        compression=compression,
        encoding_spec=encoding_spec,
        output_size_bytes=size_bytes,
        elapsed_seconds=elapsed,
        return_code=proc.returncode,
        output_path=str(output_file),
        log_path=str(log_file),
        validated=(not skip_validation),
    )


def _pick_search_columns(
    columns: List[ColumnInfo],
    scope: str,
    custom_columns_raw: str,
) -> List[str]:
    all_names = [c.name for c in columns]

    if scope == "all":
        return list(all_names)
    if scope == "fastlane-eligible":
        return [c.name for c in columns if _is_fastlanes_eligible_int32(c)]

    requested = [c.strip() for c in custom_columns_raw.split(",") if c.strip()]
    if not requested:
        raise ValueError("search-scope=custom requires --search-columns")

    unknown = [c for c in requested if c not in all_names]
    if unknown:
        raise ValueError(f"Unknown columns in --search-columns: {unknown}")
    return requested


def _format_size(num_bytes: int) -> str:
    if num_bytes < 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(num_bytes)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{num_bytes} B"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Search for a strong encoding+compression combination using chunked parquet_io_chunk "
            "to avoid OOM on large datasets."
        )
    )
    p.add_argument("--input", required=True, help="Input parquet path (for example TPCH SF100)")
    p.add_argument(
        "--binary",
        default="./build/parquet_io_chunk",
        help="Path to parquet_io_chunk binary",
    )
    p.add_argument(
        "--work-dir",
        default=_default_work_dir(),
        help="Directory for trial outputs/logs/results (uses PARQUET_IO_SHARED_ROOT when set)",
    )
    p.add_argument(
        "--compressions",
        default="NONE,SNAPPY,ZSTD",
        help=(
            "Comma-separated compression search space "
            "(NONE/UNCOMPRESS/UNCOMPRESSED,SNAPPY,ZSTD)"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for parquet_io_chunk (controls memory/speed)",
    )
    p.add_argument(
        "--search-scope",
        choices=["fastlane-eligible", "all", "custom"],
        default="fastlane-eligible",
        help=(
            "Columns to optimize: fastlane-eligible (supported INT32 physical logical classes), "
            "all, or custom list"
        ),
    )
    p.add_argument(
        "--search-columns",
        default="",
        help="Comma-separated column names used when --search-scope=custom",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Greedy coordinate-descent passes per compression",
    )
    p.add_argument(
        "--allow-future-fastlanes",
        action="store_true",
        help=(
            "Also include FASTLANE_BITPACK_RAW candidate for INT32 columns that are not currently "
            "in the supported logical-class set."
        ),
    )
    p.add_argument(
        "--search-skip-validation",
        action="store_true",
        help="Skip C++ validation during search trials for speed",
    )
    p.add_argument(
        "--validate-best",
        action="store_true",
        help="Run one final C++ validated rewrite on the best combination",
    )
    p.add_argument(
        "--show-chunk-logs",
        action="store_true",
        help="Stream parquet_io_chunk logs during each trial",
    )
    p.add_argument(
        "--keep-trial-files",
        action="store_true",
        help="Keep all trial output parquet files (default deletes non-best outputs)",
    )
    p.add_argument(
        "--final-output",
        default="",
        help="Optional explicit output parquet path for the final best validated run",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    input_file = Path(args.input).expanduser().resolve()
    binary = Path(args.binary).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")
    if not binary.exists():
        raise SystemExit(f"Chunk binary not found: {binary}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    compressions = _normalize_compressions(args.compressions)
    columns = _load_columns(input_file)
    ordered_names = [c.name for c in columns]
    search_columns = _pick_search_columns(columns, args.search_scope, args.search_columns)

    print("=== Encoding Search Setup ===")
    print(f"Input file     : {input_file}")
    print(f"Input size     : {_format_size(input_file.stat().st_size)}")
    print(f"Chunk binary   : {binary}")
    print(f"Work dir       : {work_dir}")
    print(f"Compressions   : {compressions}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Search scope   : {args.search_scope}")
    print(f"Search columns : {search_columns}")
    print(f"Max iterations : {args.max_iterations}")
    print(f"Skip validation: {args.search_skip_validation}")
    print("")

    base_map = _default_map(columns, allow_future_fastlanes=args.allow_future_fastlanes)

    cache: Dict[Tuple[str, str], TrialResult] = {}
    trial_counter = 0

    def evaluate(encoding_map: Dict[str, str], compression: str, skip_validation: bool) -> TrialResult:
        nonlocal trial_counter
        spec = _map_to_spec(encoding_map, ordered_names)
        key = (compression, spec + f"|skip={skip_validation}")
        if key in cache:
            return cache[key]

        trial_counter += 1
        out = work_dir / f"trial_{trial_counter:04d}_{compression}.parquet"
        log = work_dir / f"trial_{trial_counter:04d}_{compression}.log"

        result = _run_chunk_rewrite(
            binary=binary,
            input_file=input_file,
            output_file=out,
            log_file=log,
            encoding_spec=spec,
            compression=compression,
            batch_size=args.batch_size,
            skip_validation=skip_validation,
            show_chunk_logs=args.show_chunk_logs,
        )
        cache[key] = result

        print(
            f"trial={trial_counter:04d} compression={compression:<6} "
            f"rc={result.return_code:<3} size={_format_size(result.output_size_bytes):>10} "
            f"time={result.elapsed_seconds:8.2f}s"
        )

        if result.return_code != 0:
            print(f"  failed log: {result.log_path}")

        return result

    per_comp_best: Dict[str, TrialResult] = {}
    per_comp_map: Dict[str, Dict[str, str]] = {}

    for comp in compressions:
        print(f"\n=== Searching compression {comp} ===")
        current_map = dict(base_map)
        current = evaluate(current_map, comp, skip_validation=args.search_skip_validation)

        if current.return_code != 0:
            print(f"Compression {comp} baseline failed, skipping this codec.")
            continue

        for it in range(args.max_iterations):
            improved = False
            print(f"  -- iteration {it + 1} --")
            for col in search_columns:
                col_info = next(c for c in columns if c.name == col)
                candidates = _candidate_encodings(
                    col_info,
                    allow_future_fastlanes=args.allow_future_fastlanes,
                )
                best_local = current
                best_encoding = current_map[col]

                for enc in candidates:
                    if enc == current_map[col]:
                        continue
                    trial_map = dict(current_map)
                    trial_map[col] = enc
                    trial = evaluate(trial_map, comp, skip_validation=args.search_skip_validation)
                    if trial.return_code != 0:
                        continue
                    if trial.output_size_bytes < best_local.output_size_bytes:
                        best_local = trial
                        best_encoding = enc

                if best_encoding != current_map[col]:
                    old = current_map[col]
                    current_map[col] = best_encoding
                    current = best_local
                    improved = True
                    print(
                        f"    improved column={col} {old} -> {best_encoding} "
                        f"new_size={_format_size(current.output_size_bytes)}"
                    )

            if not improved:
                print("    no improvement in this iteration")
                break

        per_comp_best[comp] = current
        per_comp_map[comp] = dict(current_map)

    if not per_comp_best:
        raise SystemExit("No successful trials. Check logs under work-dir.")

    best_comp = min(per_comp_best.keys(), key=lambda c: per_comp_best[c].output_size_bytes)
    best_result = per_comp_best[best_comp]
    best_map = per_comp_map[best_comp]

    print("\n=== Best Result ===")
    print(f"Compression : {best_comp}")
    print(f"Size        : {_format_size(best_result.output_size_bytes)}")
    print(f"Trial file  : {best_result.output_path}")
    print(f"Trial log   : {best_result.log_path}")

    # Final validated run (optional)
    final_result = best_result
    if args.validate_best:
        final_output = (
            Path(args.final_output).expanduser().resolve()
            if args.final_output
            else work_dir / f"best_validated_{best_comp}.parquet"
        )
        final_log = work_dir / f"best_validated_{best_comp}.log"
        print("\nRunning final validated rewrite on best combination...")
        final_result = _run_chunk_rewrite(
            binary=binary,
            input_file=input_file,
            output_file=final_output,
            log_file=final_log,
            encoding_spec=_map_to_spec(best_map, ordered_names),
            compression=best_comp,
            batch_size=args.batch_size,
            skip_validation=False,
            show_chunk_logs=args.show_chunk_logs,
        )
        print(
            f"final rc={final_result.return_code} "
            f"size={_format_size(final_result.output_size_bytes)} "
            f"time={final_result.elapsed_seconds:.2f}s"
        )
        print(f"final output: {final_result.output_path}")
        print(f"final log   : {final_result.log_path}")

    # Cleanup non-best trial outputs unless requested.
    if not args.keep_trial_files:
        keep_paths = {best_result.output_path, final_result.output_path}
        for res in cache.values():
            if res.output_path not in keep_paths:
                try:
                    Path(res.output_path).unlink(missing_ok=True)
                except OSError:
                    pass

    # Persist structured summary.
    summary = SearchSummary(
        input_file=str(input_file),
        binary=str(binary),
        batch_size=args.batch_size,
        search_scope=args.search_scope,
        compressions=compressions,
        best_result=final_result,
        best_map=best_map,
        per_compression_best=per_comp_best,
    )

    summary_json_path = work_dir / "search_summary.json"
    summary_md_path = work_dir / "search_summary.md"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_file": summary.input_file,
                "binary": summary.binary,
                "batch_size": summary.batch_size,
                "search_scope": summary.search_scope,
                "compressions": summary.compressions,
                "best_result": asdict(summary.best_result),
                "best_map": summary.best_map,
                "per_compression_best": {
                    c: asdict(r) for c, r in summary.per_compression_best.items()
                },
            },
            f,
            indent=2,
        )

    best_spec = _map_to_spec(best_map, ordered_names)
    reproduce_cmd = " ".join(
        [
            shlex.quote(str(binary)),
            shlex.quote(str(input_file)),
            shlex.quote(str(final_result.output_path)),
            shlex.quote(best_spec),
            shlex.quote(best_comp),
            f"--batch-size={args.batch_size}",
        ]
    )

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Parquet Encoding Search Summary\n\n")
        f.write(f"- Input: {summary.input_file}\n")
        f.write(f"- Binary: {summary.binary}\n")
        f.write(f"- Batch size: {summary.batch_size}\n")
        f.write(f"- Search scope: {summary.search_scope}\n")
        f.write(f"- Compressions: {', '.join(summary.compressions)}\n\n")

        f.write("## Best Result\n\n")
        f.write(f"- Compression: {best_comp}\n")
        f.write(f"- Output size: {_format_size(final_result.output_size_bytes)}\n")
        f.write(f"- Output file: {final_result.output_path}\n")
        f.write(f"- Log file: {final_result.log_path}\n")
        f.write(f"- Validation in run: {final_result.validated}\n\n")

        f.write("## Reproduction Command\n\n")
        f.write(reproduce_cmd + "\n\n")

        f.write("## Best Encoding Map\n\n")
        for name in ordered_names:
            f.write(f"- {name}: {best_map[name]}\n")

    print(f"\nWrote summary JSON: {summary_json_path}")
    print(f"Wrote summary MD  : {summary_md_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
