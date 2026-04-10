#!/usr/bin/env python3
"""Run FL64-R4 parity checks and emit a machine-readable summary JSON.

This is a test orchestration helper for the native64 fastlanes test suite.
It does not participate in production encode/decode paths.

Inputs:
- gtest binary path
- matrix config JSON (filters and matrix contract)
- run tag + output summary location

Outputs:
- summary JSON containing per-command pass/fail and metadata

Typical use:
- anchor/full/stability checks via gtest filters
- optional bw37 guard and full ctest checkpoint
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

RUN_TAG_RE = re.compile(r"^rt_[0-9]{8}_[a-z]$")

DEFAULT_MATRIX_CONFIG = {
    "schema_version": "1.0.0",
    "run_tag_pattern": r"^rt_[0-9]{8}_[a-z]$",
    "report_path_pattern": "*/parquet_io_shared/reports/cudf-fastlane/<run_tag>/r4_matrix_summary.json",
    "anchors": {
        "bitwidths": [0, 1, 31, 32, 33, 37, 63, 64],
        "checks": [
            "cpu_pack_to_gpu_decode",
            "gpu_encode_to_cpu_unpack",
            "gpu_encode_to_gpu_decode_roundtrip",
            "signed_int64_bitcast_parity",
        ],
    },
    "full_sweep": {
        "bitwidth_range": {"min": 0, "max": 64},
        "patterns": ["randomized", "adversarial", "pathological"],
        "bases": ["zero", "near_int64_min", "near_int64_max", "high_unsigned"],
        "counts": [1023, 1024, 1025, 2047, 2048, 2049],
    },
    "stability": {"repeats": 3, "selected_seeds": [17, 29, 47]},
    "gtest_filters": {
        "anchor": "ParquetFastLanesNative64GeneratedTest.AnchorMatrixParity",
        "full_sweep": "ParquetFastLanesNative64GeneratedTest.FullSweepParityMatrix",
        "stability": "ParquetFastLanesNative64GeneratedTest.StabilityRepeatsSelectedSeeds",
        "bw37_guard": "ParquetFastLanesNative64Bw37Test.*",
    },
}


@dataclass
class CommandResult:
    """Normalized result payload for each executed command."""

    name: str
    filter: str
    command: str
    return_code: int
    duration_seconds: float
    passed: bool
    passed_tests_reported: Optional[int]
    failed_tests_reported: Optional[int]


def _read_json(path: Path) -> dict:
    """Read UTF-8 JSON file into a Python dict."""

    return json.loads(path.read_text(encoding="utf-8"))


def _parse_count(pattern: str, text: str) -> Optional[int]:
    """Extract first integer capture for a regex pattern from command output."""

    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1))


def _run_shell(command: str, cwd: Optional[Path] = None) -> tuple[int, str, str, float]:
    """Run a shell command and return rc/stdout/stderr/elapsed_seconds."""

    start = time.monotonic()
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.monotonic() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def _ensure_report_path_policy(report_json: Path, run_tag: str) -> bool:
    """Enforce report path contract used by remote validation workflows."""

    norm = report_json.as_posix()
    expected_suffix = f"/parquet_io_shared/reports/cudf-fastlane/{run_tag}/r4_matrix_summary.json"
    return norm.endswith(expected_suffix)


def _build_commands(args: argparse.Namespace, cfg: dict) -> list[tuple[str, str, str, Optional[Path]]]:
    """Build ordered command list from CLI flags + matrix config.

    Return tuples of (logical_name, gtest_filter, shell_command, optional_cwd).
    """

    filters = cfg["gtest_filters"]
    gtest_bin = Path(args.gtest_binary).resolve()
    cmds: list[tuple[str, str, str, Optional[Path]]] = []

    def gtest_cmd(name: str, gfilter: str) -> tuple[str, str, str, Optional[Path]]:
        cmd = f"{gtest_bin} --gtest_filter={gfilter}"
        return (name, gfilter, cmd, None)

    if args.phase in ("anchor", "all"):
        cmds.append(gtest_cmd("anchor", filters["anchor"]))

    if args.phase in ("full", "all"):
        cmds.append(gtest_cmd("full_sweep", filters["full_sweep"]))

    if args.with_stability and args.phase in ("full", "all"):
        cmds.append(gtest_cmd("stability", filters["stability"]))

    if args.with_bw37_guard:
        cmds.append(gtest_cmd("bw37_guard", filters["bw37_guard"]))

    if args.with_ctest:
        # gtest binary is expected in cpp/build/gtests, so the ctest root is parent dir.
        build_dir = gtest_bin.parent.parent
        cmds.append(("ctest_checkpoint", "", "ctest --output-on-failure --no-tests=error", build_dir))

    return cmds


def main() -> int:
    """CLI entrypoint.

    Returns:
    - 0 if all selected checks pass
    - 2 if any selected check fails
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtest-binary", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--phase", choices=["anchor", "full", "all"], default="all")
    parser.add_argument("--config", default=None)
    parser.add_argument("--with-stability", action="store_true")
    parser.add_argument("--with-bw37-guard", action="store_true")
    parser.add_argument("--with-ctest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not RUN_TAG_RE.match(args.run_tag):
        raise SystemExit(f"run_tag must match {RUN_TAG_RE.pattern}: {args.run_tag}")

    cfg = copy.deepcopy(DEFAULT_MATRIX_CONFIG)
    if args.config:
        cfg = _read_json(Path(args.config))

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    commands = _build_commands(args, cfg)

    results: list[CommandResult] = []
    command_log: list[str] = []

    for name, gfilter, cmd, cwd in commands:
        command_log.append(cmd if cwd is None else f"(cd {cwd} && {cmd})")

        if args.dry_run:
            # Dry-run reserves command ordering and payload shape without execution.
            results.append(
                CommandResult(
                    name=name,
                    filter=gfilter,
                    command=cmd,
                    return_code=0,
                    duration_seconds=0.0,
                    passed=True,
                    passed_tests_reported=None,
                    failed_tests_reported=None,
                )
            )
            continue

        rc, out, err, elapsed = _run_shell(cmd, cwd=cwd)
        merged = (out or "") + "\n" + (err or "")

        passed_count = _parse_count(r"\[\s*PASSED\s*\]\s*(\d+)\s+tests?", merged)
        failed_count = _parse_count(r"\[\s*FAILED\s*\]\s*(\d+)\s+tests?", merged)

        result = CommandResult(
            name=name,
            filter=gfilter,
            command=cmd,
            return_code=rc,
            duration_seconds=round(elapsed, 3),
            passed=(rc == 0),
            passed_tests_reported=passed_count,
            failed_tests_reported=failed_count,
        )
        results.append(result)

    passed_commands = sum(1 for r in results if r.passed)
    failed_commands = len(results) - passed_commands

    payload = {
        "schema_version": "1.0.0",
        "run_tag": args.run_tag,
        "report_path": report_json.as_posix(),
        "path_policy_ok": _ensure_report_path_policy(report_json, args.run_tag),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "gtest_binary": str(Path(args.gtest_binary).resolve()),
        "phase": args.phase,
        "commands": command_log,
        "results": [asdict(r) for r in results],
        "summary": {
            "total_commands": len(results),
            "passed_commands": passed_commands,
            "failed_commands": failed_commands,
            "all_passed": failed_commands == 0,
        },
        "matrix_contract": {
            "anchors": cfg["anchors"]["bitwidths"],
            "full_sweep": cfg["full_sweep"]["bitwidth_range"],
            "stability_repeats": cfg["stability"]["repeats"],
        },
    }

    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failed_commands != 0:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
