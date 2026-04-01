#!/usr/bin/env python3
"""Run SNAPPY INT64 FastLanes sensitivity cases and emit layered artifacts.

What this script does:
1. Builds a baseline encoding map where all INT64 columns use DELTA_BINARY_PACKED.
2. Runs one-by-one FASTLANES_BITPACK toggles for selected INT64 columns.
3. Optionally runs an aggregate all-int64-fastlane control case.
4. Persists outputs in a 3-layer artifact layout for human + machine + raw data.

Input:
- --input: Source parquet file.
- --cpp-binary: parquet_io_chunk executable.
- --roundtrip-script: parquet_io_roundtrip_check.py path.

Output layout (run_<timestamp>):
- 01_human/: concise markdown summary for people.
- 02_machine/: run_config.json and run_index.json for tooling.
- 03_raw/: logs, stdout captures, csv tables, output parquet files.

Primary outputs:
- 03_raw/run_manifest.csv
- 03_raw/case_summary.csv
- 03_raw/logs/*.log
- 03_raw/stdout/*.txt
- 03_raw/cases/*.parquet

Usage example:
    python3 ./tools/search/run_int64_snappy_page_stats.py \
        --input ./CUDF-0003.roundtrip.parquet \
        --cpp-binary ./build/parquet_io_chunk \
        --output-dir ./artifacts/fastlanes/snappy_int64_page_stats
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from search_best_parquet_encoding import _default_map, _load_columns  # noqa: E402


@dataclass
class CaseResult:
    run_id: str
    case_name: str
    input_file: str
    output_file: str
    log_file: str
    stdout_file: str
    compression: str
    encoding_spec: str
    encoding_map_hash: str
    return_code: int
    elapsed_seconds: float
    output_size_bytes: int
    status: str


@dataclass
class RunLayout:
    run_root: Path
    human_dir: Path
    machine_dir: Path
    raw_dir: Path
    cases_dir: Path
    logs_dir: Path
    stdout_dir: Path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SNAPPY-only INT64 DELTA vs FASTLANES sensitivity cases and write manifest CSV."
        )
    )
    p.add_argument("--input", required=True, help="Input parquet path")
    p.add_argument(
        "--roundtrip-script",
        default=str((THIS_DIR.parent / "roundtrip" / "parquet_io_roundtrip_check.py").resolve()),
        help="Path to parquet_io_roundtrip_check.py",
    )
    p.add_argument(
        "--cpp-binary",
        default="./build/parquet_io_chunk",
        help="Path to parquet_io_chunk executable",
    )
    p.add_argument(
        "--output-dir",
        default="./artifacts/fastlanes/snappy_int64_page_stats",
        help=(
            "Base directory for run_* folders using layered artifacts "
            "(01_human, 02_machine, 03_raw)"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size passed to parquet_io_chunk",
    )
    p.add_argument(
        "--int64-columns",
        default="",
        help="Optional comma-separated subset of INT64 columns to analyze (default: all INT64)",
    )
    p.add_argument(
        "--skip-all-fastlane-case",
        action="store_true",
        help="Skip the all_int64_fastlane aggregate control case",
    )
    p.add_argument(
        "--show-chunk-logs",
        action="store_true",
        help="Stream parquet_io_chunk output to terminal while runs execute",
    )
    p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately on the first failed case",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return p


def _pick_int64_columns(input_file: Path, requested: str) -> List[str]:
    cols = _load_columns(input_file)
    all_int64 = [c.name for c in cols if c.physical_type == "INT64"]
    if not all_int64:
        raise SystemExit("No INT64 physical columns were found in the input parquet")

    if not requested.strip():
        return all_int64

    requested_list = [x.strip() for x in requested.split(",") if x.strip()]
    unknown = [x for x in requested_list if x not in all_int64]
    if unknown:
        raise SystemExit(f"Unknown/non-INT64 columns in --int64-columns: {unknown}")
    return requested_list


def _spec_from_map(encoding_map: Dict[str, str], ordered_columns: List[str]) -> str:
    return ",".join(f"{c}:{encoding_map[c]}" for c in ordered_columns)


def _hash_spec(spec: str) -> str:
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def _make_case_maps(input_file: Path, int64_targets: List[str], include_all_fastlane: bool) -> List[tuple[str, Dict[str, str]]]:
    cols = _load_columns(input_file)
    ordered_names = [c.name for c in cols]
    base_map = _default_map(cols)

    # Force all INT64 columns to DELTA as the baseline for this sensitivity workflow.
    for c in cols:
        if c.physical_type == "INT64":
            base_map[c.name] = "DELTA_BINARY_PACKED"

    cases: List[tuple[str, Dict[str, str]]] = []
    cases.append(("baseline_delta", dict(base_map)))

    for col in int64_targets:
        case_map = dict(base_map)
        case_map[col] = "FASTLANES_BITPACK"
        cases.append((f"{col}_fastlane", case_map))

    if include_all_fastlane:
        all_map = dict(base_map)
        for col in int64_targets:
            all_map[col] = "FASTLANES_BITPACK"
        cases.append(("all_int64_fastlane", all_map))

    # Normalize map ordering now so all downstream hashes/specs are stable.
    normalized_cases = []
    for name, mp in cases:
        normalized = {k: mp[k] for k in ordered_names}
        normalized_cases.append((name, normalized))
    return normalized_cases


def _run_case(
    *,
    run_id: str,
    case_name: str,
    input_file: Path,
    output_file: Path,
    log_file: Path,
    stdout_file: Path,
    roundtrip_script: Path,
    cpp_binary: Path,
    batch_size: int,
    encoding_spec: str,
    show_chunk_logs: bool,
    dry_run: bool,
) -> CaseResult:
    cmd = [
        sys.executable,
        str(roundtrip_script),
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--keep-output",
        "--conversion-engine",
        "cpp",
        "--cpp-binary",
        str(cpp_binary),
        "--validator",
        "auto",
        "--compression",
        "SNAPPY",
        "--batch-size",
        str(batch_size),
        "--encoding-spec",
        encoding_spec,
        "--cpp-enable-log",
        "--cpp-log-file",
        str(log_file),
    ]
    if show_chunk_logs:
        cmd.append("--cpp-show-log")

    env = os.environ.copy()
    env["FLS_DEBUG_WORKLOAD"] = "1"
    env["FLS_DEBUG_HEADER"] = "1"

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return CaseResult(
            run_id=run_id,
            case_name=case_name,
            input_file=str(input_file),
            output_file=str(output_file),
            log_file=str(log_file),
            stdout_file=str(stdout_file),
            compression="SNAPPY",
            encoding_spec=encoding_spec,
            encoding_map_hash=_hash_spec(encoding_spec),
            return_code=0,
            elapsed_seconds=0.0,
            output_size_bytes=-1,
            status="DRY_RUN",
        )

    t0 = time.perf_counter()
    run_kwargs = {"text": True, "check": False, "env": env}
    if show_chunk_logs:
        proc = subprocess.run(cmd, **run_kwargs)
        captured_stdout = ""
        captured_stderr = ""
    else:
        proc = subprocess.run(cmd, capture_output=True, **run_kwargs)
        captured_stdout = proc.stdout or ""
        captured_stderr = proc.stderr or ""
    elapsed = time.perf_counter() - t0

    with open(stdout_file, "w", encoding="utf-8") as f:
        if captured_stdout:
            f.write("=== STDOUT ===\n")
            f.write(captured_stdout)
            if not captured_stdout.endswith("\n"):
                f.write("\n")
        if captured_stderr:
            f.write("=== STDERR ===\n")
            f.write(captured_stderr)
            if not captured_stderr.endswith("\n"):
                f.write("\n")

    out_size = output_file.stat().st_size if output_file.exists() else -1
    status = "PASS" if proc.returncode == 0 and out_size >= 0 else "FAIL"

    return CaseResult(
        run_id=run_id,
        case_name=case_name,
        input_file=str(input_file),
        output_file=str(output_file),
        log_file=str(log_file),
        stdout_file=str(stdout_file),
        compression="SNAPPY",
        encoding_spec=encoding_spec,
        encoding_map_hash=_hash_spec(encoding_spec),
        return_code=proc.returncode,
        elapsed_seconds=elapsed,
        output_size_bytes=out_size,
        status=status,
    )


def _write_manifest(manifest_path: Path, results: List[CaseResult]) -> None:
    fields = [
        "run_id",
        "case_name",
        "input_file",
        "output_file",
        "log_file",
        "stdout_file",
        "compression",
        "encoding_spec",
        "encoding_map_hash",
        "return_code",
        "elapsed_seconds",
        "output_size_bytes",
        "status",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "run_id": r.run_id,
                    "case_name": r.case_name,
                    "input_file": r.input_file,
                    "output_file": r.output_file,
                    "log_file": r.log_file,
                    "stdout_file": r.stdout_file,
                    "compression": r.compression,
                    "encoding_spec": r.encoding_spec,
                    "encoding_map_hash": r.encoding_map_hash,
                    "return_code": r.return_code,
                    "elapsed_seconds": f"{r.elapsed_seconds:.6f}",
                    "output_size_bytes": r.output_size_bytes,
                    "status": r.status,
                }
            )


def _write_case_summary(summary_path: Path, results: List[CaseResult]) -> None:
    baseline = next((r for r in results if r.case_name == "baseline_delta"), None)
    base_size = baseline.output_size_bytes if baseline and baseline.output_size_bytes > 0 else None
    base_time = baseline.elapsed_seconds if baseline and baseline.elapsed_seconds > 0 else None

    fields = [
        "case_name",
        "status",
        "output_size_bytes",
        "size_delta_bytes",
        "size_delta_pct",
        "elapsed_seconds",
        "time_delta_seconds",
        "time_delta_pct",
        "log_file",
        "output_file",
    ]

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            size_delta = ""
            size_delta_pct = ""
            if base_size is not None and r.output_size_bytes >= 0:
                d = r.output_size_bytes - base_size
                size_delta = str(d)
                size_delta_pct = f"{(d / base_size) * 100.0:.6f}"

            time_delta = ""
            time_delta_pct = ""
            if base_time is not None:
                dt = r.elapsed_seconds - base_time
                time_delta = f"{dt:.6f}"
                if base_time > 0:
                    time_delta_pct = f"{(dt / base_time) * 100.0:.6f}"

            writer.writerow(
                {
                    "case_name": r.case_name,
                    "status": r.status,
                    "output_size_bytes": r.output_size_bytes,
                    "size_delta_bytes": size_delta,
                    "size_delta_pct": size_delta_pct,
                    "elapsed_seconds": f"{r.elapsed_seconds:.6f}",
                    "time_delta_seconds": time_delta,
                    "time_delta_pct": time_delta_pct,
                    "log_file": r.log_file,
                    "output_file": r.output_file,
                }
            )


def _init_run_layout(base_output_dir: Path, run_id: str) -> RunLayout:
    run_root = base_output_dir / f"run_{run_id}"
    human_dir = run_root / "01_human"
    machine_dir = run_root / "02_machine"
    raw_dir = run_root / "03_raw"
    cases_dir = raw_dir / "cases"
    logs_dir = raw_dir / "logs"
    stdout_dir = raw_dir / "stdout"

    for d in [human_dir, machine_dir, cases_dir, logs_dir, stdout_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return RunLayout(
        run_root=run_root,
        human_dir=human_dir,
        machine_dir=machine_dir,
        raw_dir=raw_dir,
        cases_dir=cases_dir,
        logs_dir=logs_dir,
        stdout_dir=stdout_dir,
    )


def _write_human_summary_md(path: Path, run_id: str, input_file: Path, results: List[CaseResult]) -> None:
    baseline = next((r for r in results if r.case_name == "baseline_delta"), None)
    base_size = baseline.output_size_bytes if baseline and baseline.output_size_bytes > 0 else None
    base_time = baseline.elapsed_seconds if baseline and baseline.elapsed_seconds > 0 else None

    with open(path, "w", encoding="utf-8") as f:
        f.write("# INT64 SNAPPY Sensitivity Run Summary\n\n")
        f.write(f"- run_id: {run_id}\n")
        f.write(f"- input: {input_file}\n")
        f.write("- purpose: compare one-by-one INT64 FASTLANES toggles against DELTA baseline\n\n")
        f.write("## Cases\n\n")
        f.write("| case | status | size_bytes | size_delta_pct | time_seconds | time_delta_pct |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in results:
            size_delta_pct = ""
            time_delta_pct = ""
            if base_size is not None and r.output_size_bytes >= 0:
                size_delta_pct = f"{((r.output_size_bytes - base_size) / base_size) * 100.0:.6f}"
            if base_time is not None and base_time > 0:
                time_delta_pct = f"{((r.elapsed_seconds - base_time) / base_time) * 100.0:.6f}"

            f.write(
                f"| {r.case_name} | {r.status} | {r.output_size_bytes} | {size_delta_pct} "
                f"| {r.elapsed_seconds:.6f} | {time_delta_pct} |\n"
            )


def _write_machine_index_json(path: Path, layout: RunLayout, run_id: str, results: List[CaseResult]) -> None:
    payload = {
        "run_id": run_id,
        "layout": {
            "human_dir": str(layout.human_dir),
            "machine_dir": str(layout.machine_dir),
            "raw_dir": str(layout.raw_dir),
            "cases_dir": str(layout.cases_dir),
            "logs_dir": str(layout.logs_dir),
            "stdout_dir": str(layout.stdout_dir),
        },
        "results": [
            {
                "case_name": r.case_name,
                "status": r.status,
                "return_code": r.return_code,
                "output_size_bytes": r.output_size_bytes,
                "elapsed_seconds": round(r.elapsed_seconds, 6),
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = _build_arg_parser().parse_args()

    input_file = Path(args.input).expanduser().resolve()
    roundtrip_script = Path(args.roundtrip_script).expanduser().resolve()
    cpp_binary = Path(args.cpp_binary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_file.exists():
        raise SystemExit(f"Input parquet not found: {input_file}")
    if not roundtrip_script.exists():
        raise SystemExit(f"Roundtrip script not found: {roundtrip_script}")
    if not args.dry_run and not cpp_binary.exists():
        raise SystemExit(f"C++ binary not found: {cpp_binary}")

    int64_targets = _pick_int64_columns(input_file, args.int64_columns)
    include_all_fastlane = not args.skip_all_fastlane_case

    cases = _make_case_maps(input_file, int64_targets, include_all_fastlane)
    ordered_columns = [c.name for c in _load_columns(input_file)]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    layout = _init_run_layout(output_dir, run_id)

    run_config_path = layout.machine_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "input_file": str(input_file),
                "roundtrip_script": str(roundtrip_script),
                "cpp_binary": str(cpp_binary),
                "compression": "SNAPPY",
                "batch_size": args.batch_size,
                "int64_targets": int64_targets,
                "cases": [name for name, _ in cases],
                "enable_debug_env": {"FLS_DEBUG_WORKLOAD": "1", "FLS_DEBUG_HEADER": "1"},
            },
            f,
            indent=2,
        )

    print("=== INT64 SNAPPY Page-Stats Run Setup ===")
    print(f"Input file      : {input_file}")
    print(f"Roundtrip script: {roundtrip_script}")
    print(f"C++ binary      : {cpp_binary}")
    print(f"Run dir         : {layout.run_root}")
    print(f"Human layer     : {layout.human_dir}")
    print(f"Machine layer   : {layout.machine_dir}")
    print(f"Raw layer       : {layout.raw_dir}")
    print(f"INT64 targets   : {int64_targets}")
    print(f"Cases           : {[name for name, _ in cases]}")

    results: List[CaseResult] = []
    for case_name, case_map in cases:
        spec = _spec_from_map(case_map, ordered_columns)
        output_file = layout.cases_dir / f"{case_name}.parquet"
        log_file = layout.logs_dir / f"{case_name}.log"
        stdout_file = layout.stdout_dir / f"{case_name}.txt"

        print(f"\nRunning case: {case_name}")
        result = _run_case(
            run_id=run_id,
            case_name=case_name,
            input_file=input_file,
            output_file=output_file,
            log_file=log_file,
            stdout_file=stdout_file,
            roundtrip_script=roundtrip_script,
            cpp_binary=cpp_binary,
            batch_size=args.batch_size,
            encoding_spec=spec,
            show_chunk_logs=args.show_chunk_logs,
            dry_run=args.dry_run,
        )
        print(
            f"  status={result.status} rc={result.return_code} "
            f"size={result.output_size_bytes} elapsed={result.elapsed_seconds:.2f}s"
        )

        results.append(result)

        if args.stop_on_failure and result.status == "FAIL":
            print("Stopping due to --stop-on-failure")
            break

    manifest_path = layout.raw_dir / "run_manifest.csv"
    summary_path = layout.raw_dir / "case_summary.csv"
    human_summary_path = layout.human_dir / "run_summary.md"
    machine_index_path = layout.machine_dir / "run_index.json"

    _write_manifest(manifest_path, results)
    _write_case_summary(summary_path, results)
    _write_human_summary_md(human_summary_path, run_id, input_file, results)
    _write_machine_index_json(machine_index_path, layout, run_id, results)

    print("\n=== Run Outputs ===")
    print(f"Run config   : {run_config_path}")
    print(f"Run index    : {machine_index_path}")
    print(f"Human summary: {human_summary_path}")
    print(f"Run manifest : {manifest_path}")
    print(f"Case summary : {summary_path}")

    print("\nNext step:")
    print(
        "  python3 ./tools/search/extract_fastlanes_page_stats.py "
        f"--manifest-csv {manifest_path}"
    )

    return 0 if all(r.status in {"PASS", "DRY_RUN"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
