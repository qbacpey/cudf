#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""TPC-H FastLanes benchmark driver.

Steps (select with --steps, default: all of them in this order):
  sweep      fastlanes_encoding_bench sweep over every FastLanes-eligible column of every table:
             PLAIN, DICTIONARY, DELTA_BINARY_PACKED, BYTE_STREAM_SPLIT and FastLanes, each with
             NONE, SNAPPY and ZSTD.
  tables     full-table rewrites with parquet_io_chunk for four encoding plans per codec, followed
             by warm-cache read timing with `fastlanes_encoding_bench read`.
  ablation   FastLanes vs. the best standard encoding under three page layouts: a 614-row cap on
             cuDF's default 5000-row fragments (what parquet_io_chunk produced by default on these
             files), cuDF's default layout (20000-row pages), and exact 20480-row pages.
  summarize  aggregate every CSV into 02_machine/summary.json and 01_human/summary.md.

Parquet encodes column chunks independently, so the per-column sweep yields the exact best encoding
for each column. The full-table plans then check that the per-column results compose:
  cudf-default         every column uses cudf's default encoding
  best-standard        eligible columns use their smallest standard encoding (per codec)
  fastlanes-all        eligible columns use FastLanes
  best-with-fastlanes  eligible columns use their smallest encoding, FastLanes included
Columns FastLanes cannot encode keep DEFAULT in every plan, so plan differences come only from
eligible columns.

Every step appends to CSVs under <run-dir>/03_raw and skips work that is already recorded, so the
driver can simply be re-run after an interruption.

Usage (inside the cudf devcontainer, from cpp/examples/parquet_io):
    python3 tools/bench/run_tpch_sf100_fastlanes_bench.py \
        --data-dir artifacts/tpch100/sf100 \
        --run-dir artifacts/fastlanes_bench_sf100_<YYYYMMDD>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyArrow is required. Activate the rapids conda env first.") from exc

TABLES = ["lineitem", "orders", "partsupp", "part", "customer", "supplier", "nation", "region"]
STANDARD_ENCODINGS = ["PLAIN", "DICTIONARY", "DELTA_BINARY_PACKED", "BYTE_STREAM_SPLIT"]
SWEEP_ENCODINGS = STANDARD_ENCODINGS + ["FASTLANES"]
SWEEP_CODECS = ["NONE", "SNAPPY", "ZSTD"]
TABLE_CODECS = ["SNAPPY", "ZSTD"]
PLANS = ["cudf-default", "best-standard", "fastlanes-all", "best-with-fastlanes"]
ROW_GROUP_ROWS = 122_880
# Pages are built from whole page fragments, so the fragment size equals the page size: every page
# holds exactly 20 FastLanes vectors and every row group exactly 6 pages.
PAGE_ROWS = 20_480
ABLATION_COLUMNS = [("lineitem", "l_partkey"), ("lineitem", "l_shipdate")]
# (max page rows, fragment rows; 0 = cuDF default fragments of 5000 rows)
ABLATION_LAYOUTS = [(614, 0), (20_000, 0), (PAGE_ROWS, PAGE_ROWS)]


@dataclass
class ColumnInfo:
    name: str
    physical: str
    logical: str
    converted: str

    @property
    def fastlanes_kind(self) -> Optional[str]:
        """Return "INT64"/"INT32" when FastLanes can encode the column, otherwise None.

        Mirrors the writer's eligibility rules: INT64 without a decimal or temporal annotation maps
        to FASTLANES_DELTA_BINARY; INT32 integers, dates, TIME_MILLIS and 32-bit decimals map to
        FASTLANE_BITPACK_RAW.
        """
        logical = self.logical.replace(" ", "")
        if self.physical == "INT64":
            if "Decimal" in logical or self.converted == "DECIMAL":
                return None
            if logical in {"None", ""} and self.converted in {"NONE", "INT_64", "UINT_64"}:
                return "INT64"
            return "INT64" if "Int(bitWidth=64" in logical else None
        if self.physical == "INT32":
            if self.converted in {
                "NONE",
                "INT_8",
                "INT_16",
                "INT_32",
                "UINT_8",
                "UINT_16",
                "UINT_32",
                "DATE",
                "TIME_MILLIS",
                "DECIMAL",
            }:
                return "INT32"
            keys = ("Int(bitWidth=", "Date", "Decimal", "Time(")
            return "INT32" if any(k in logical for k in keys) else None
        return None

    @property
    def fastlanes_encoding(self) -> str:
        return "FASTLANES_DELTA_BINARY" if self.fastlanes_kind == "INT64" else "FASTLANE_BITPACK_RAW"


def load_schema(path: Path) -> List[ColumnInfo]:
    schema = pq.ParquetFile(str(path)).schema
    cols = []
    for i in range(len(schema)):
        c = schema.column(i)
        cols.append(ColumnInfo(c.name, str(c.physical_type), str(c.logical_type), str(c.converted_type)))
    return cols


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, row: Dict[str, object]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def dedupe(rows: Iterable[Dict[str, str]], keys: Tuple[str, ...]) -> List[Dict[str, str]]:
    """Keep the last row for each key, preserving first-seen order."""
    latest: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for r in rows:
        latest[tuple(r[k] for k in keys)] = r
    return list(latest.values())


def run(cmd: List[str], log_path: Path, timeout_s: Optional[float] = None) -> int:
    """Run `cmd` appending its output to `log_path`; returns -9 if it exceeds `timeout_s`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            log.write(f"\n[driver] killed after exceeding {timeout_s:.0f} s\n")
            return -9
    return proc.returncode


def is_ok(row: Dict[str, str]) -> bool:
    return row["error"] == "" and row["validated"] == "1" and row["unexpected_pages"] == "0"


def sweep_winners(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, Optional[Dict[str, str]]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault((r["table"], r["column"], r["compression"]), []).append(r)

    winners = {}
    for key, cands in grouped.items():
        ok = [r for r in cands if is_ok(r)]
        standard = [r for r in ok if r["requested"] in STANDARD_ENCODINGS]
        fastlanes = [r for r in ok if r["requested"] == "FASTLANES"]
        best_std = min(standard, key=lambda r: int(r["bytes"])) if standard else None
        fl = fastlanes[0] if fastlanes else None
        pool = [r for r in (best_std, fl) if r is not None]
        best_all = min(pool, key=lambda r: int(r["bytes"])) if pool else None
        winners[key] = {"best_standard": best_std, "fastlanes": fl, "best_overall": best_all}
    return winners


def plan_map(
    plan: str,
    table: str,
    codec: str,
    columns: List[ColumnInfo],
    winners: Dict[Tuple[str, str, str], Dict[str, Optional[Dict[str, str]]]],
) -> Dict[str, str]:
    mapping = {}
    for c in columns:
        enc = "DEFAULT"
        if c.fastlanes_kind and plan != "cudf-default":
            w = winners.get((table, c.name, codec), {})
            if plan == "fastlanes-all":
                enc = c.fastlanes_encoding
            elif plan == "best-standard" and w.get("best_standard"):
                enc = w["best_standard"]["resolved"]
            elif plan == "best-with-fastlanes" and w.get("best_overall"):
                enc = w["best_overall"]["resolved"]
        mapping[c.name] = enc
    return mapping


def capture_environment(args: argparse.Namespace, machine_dir: Path) -> None:
    def sh(cmd: List[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            return ""

    repo = Path(__file__).resolve().parents[5]
    env = {
        "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
        "nvcc": sh(["nvcc", "--version"]).splitlines()[-1:] or [""],
        "cudf_version": (repo / "VERSION").read_text().strip() if (repo / "VERSION").exists() else "",
        "git_commit": sh(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_branch": sh(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"]),
        "cpu": sh(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip(),
        "cpus": sh(["nproc"]),
        "row_group_rows": ROW_GROUP_ROWS,
        "page_rows": PAGE_ROWS,
        "v2_headers": True,
        "warmup": args.warmup,
        "repeats": args.repeats,
    }
    env["nvcc"] = env["nvcc"][0]
    machine_dir.mkdir(parents=True, exist_ok=True)
    (machine_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


def step_sweep(args: argparse.Namespace, raw: Path) -> None:
    sweep_csv = raw / "sweep.csv"
    all_combos = [(e, c) for e in SWEEP_ENCODINGS for c in SWEEP_CODECS]

    for table in args.tables:
        path = args.data_dir / f"{table}.parquet"
        for col in [c.name for c in load_schema(path) if c.fastlanes_kind]:
            # A watchdog exit (rc 3) records the hung case, so each retry makes progress.
            for _ in range(len(all_combos)):
                recorded = {
                    (r["requested"], r["compression"])
                    for r in read_csv(sweep_csv)
                    if r["table"] == table and r["column"] == col
                }
                todo = [x for x in all_combos if x not in recorded]
                if not todo:
                    break
                t0 = time.perf_counter()
                rc = run(
                    [
                        str(args.bench_bin),
                        "sweep",
                        f"--input={path}",
                        f"--table={table}",
                        f"--columns={col}",
                        f"--output={sweep_csv}",
                        f"--combos={','.join(f'{e}/{c}' for e, c in todo)}",
                        f"--warmup={args.warmup}",
                        f"--repeats={args.repeats}",
                        f"--row-group-rows={ROW_GROUP_ROWS}",
                        f"--page-rows={PAGE_ROWS}",
                        f"--hang-timeout-s={args.hang_timeout}",
                    ],
                    raw / "logs" / f"sweep_{table}_{col}.log",
                )
                print(
                    f"[sweep] {table}.{col}: {len(todo)} cases, rc={rc} {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
                if rc != 3:
                    break


def parse_rewrite_log(log_path: Path) -> Tuple[float, bool]:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    m = re.search(r"Total time:\s+([\d.]+) ms", text)
    return (float(m.group(1)) if m else -1.0), ("=== SUCCESS ===" in text)


def step_tables(args: argparse.Namespace, raw: Path) -> None:
    tables_csv = raw / "tables.csv"
    reads_csv = raw / "reads.csv"
    done = {(r["table"], r["plan"], r["compression"]) for r in read_csv(tables_csv)}
    winners = sweep_winners(dedupe(read_csv(raw / "sweep.csv"), ("table", "column", "requested", "compression")))
    scratch = args.scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)

    for table in args.tables:
        path = args.data_dir / f"{table}.parquet"
        columns = load_schema(path)
        for codec in TABLE_CODECS:
            seen_specs: Dict[str, str] = {}
            for plan in PLANS:
                mapping = plan_map(plan, table, codec, columns, winners)
                spec = ",".join(f"{c.name}:{mapping[c.name]}" for c in columns)
                if (table, plan, codec) in done:
                    seen_specs.setdefault(spec, plan)
                    continue
                uses_fastlanes = any(v.startswith("FASTLANE") for v in mapping.values())
                eligible_map = ";".join(f"{c.name}={mapping[c.name]}" for c in columns if c.fastlanes_kind)

                if spec in seen_specs:
                    prior = next(
                        r
                        for r in read_csv(tables_csv)
                        if (r["table"], r["plan"], r["compression"]) == (table, seen_specs[spec], codec)
                    )
                    row = dict(prior)
                    row.update({"plan": plan, "same_as": seen_specs[spec], "eligible_map": eligible_map})
                    append_csv(tables_csv, row)
                    print(f"[tables] {table}/{plan}/{codec}: identical to {seen_specs[spec]}", flush=True)
                    continue

                out = scratch / f"{table}_{plan}_{codec}.parquet"
                log = raw / "logs" / f"table_{table}_{plan}_{codec}.log"
                log.unlink(missing_ok=True)
                cmd = [
                    str(args.chunk_bin),
                    str(path),
                    str(out),
                    spec,
                    codec,
                    "--enable-v2-headers",
                    f"--max-page-rows={PAGE_ROWS}",
                    f"--page-fragment-rows={PAGE_ROWS}",
                    f"--batch-size={args.batch_size}",
                    f"--rgs-per-read={args.rgs_per_read}",
                    f"--log-file={log}",
                ]
                if not uses_fastlanes:
                    cmd.append("--skip-validation")
                # parquet_io_chunk has no watchdog, so bound it by input size.
                timeout_s = 300 + 120 * path.stat().st_size / 1e9
                rc = run(cmd, raw / "logs" / f"table_{table}_{plan}_{codec}.stdout", timeout_s)
                rewrite_ms, success = parse_rewrite_log(log)
                out_bytes = out.stat().st_size if out.exists() else -1

                label = f"{table}/{plan}/{codec}"
                if rc == 0 and out.exists():
                    run(
                        [
                            str(args.bench_bin),
                            "read",
                            f"--input={out}",
                            f"--label={label}",
                            f"--output={reads_csv}",
                            "--warmup=1",
                            f"--repeats={args.repeats}",
                            "--read-batch-row-groups=128",
                            f"--hang-timeout-s={args.hang_timeout}",
                        ],
                        raw / "logs" / f"read_{table}_{plan}_{codec}.log",
                    )
                append_csv(
                    tables_csv,
                    {
                        "table": table,
                        "plan": plan,
                        "compression": codec,
                        "output_bytes": out_bytes,
                        "rewrite_ms": rewrite_ms,
                        "validated": (1 if success else 0) if uses_fastlanes else -1,
                        "return_code": rc,
                        "uses_fastlanes": int(uses_fastlanes),
                        "same_as": "",
                        "eligible_map": eligible_map,
                    },
                )
                seen_specs[spec] = plan
                print(
                    f"[tables] {label}: rc={rc} bytes={out_bytes} rewrite_ms={rewrite_ms:.0f} "
                    f"validated={success if uses_fastlanes else 'skipped'}",
                    flush=True,
                )
                if not args.keep_outputs:
                    out.unlink(missing_ok=True)


def step_ablation(args: argparse.Namespace, raw: Path) -> None:
    abl_csv = raw / "ablation.csv"
    winners = sweep_winners(dedupe(read_csv(raw / "sweep.csv"), ("table", "column", "requested", "compression")))
    done = {(r["table"], r["column"], r["page_rows"], r["fragment_rows"]) for r in read_csv(abl_csv)}
    for table, col in ABLATION_COLUMNS:
        best = (winners.get((table, col, "SNAPPY")) or {}).get("best_standard")
        encodings = ["FASTLANES"] + ([best["requested"]] if best else [])
        for page_rows, fragment_rows in ABLATION_LAYOUTS:
            if (table, col, str(page_rows), str(fragment_rows)) in done:
                continue
            rc = run(
                [
                    str(args.bench_bin),
                    "sweep",
                    f"--input={args.data_dir / f'{table}.parquet'}",
                    f"--table={table}",
                    f"--columns={col}",
                    f"--output={abl_csv}",
                    f"--encodings={','.join(encodings)}",
                    "--compressions=SNAPPY",
                    f"--warmup={args.warmup}",
                    f"--repeats={args.repeats}",
                    f"--row-group-rows={ROW_GROUP_ROWS}",
                    f"--page-rows={page_rows}",
                    f"--fragment-rows={fragment_rows}",
                    f"--hang-timeout-s={args.hang_timeout}",
                ],
                raw / "logs" / f"ablation_{table}_{col}_{page_rows}_{fragment_rows}.log",
            )
            print(
                f"[ablation] {table}.{col} page_rows={page_rows} fragment_rows={fragment_rows}: rc={rc}",
                flush=True,
            )


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def step_summarize(args: argparse.Namespace, run_dir: Path) -> None:
    raw, machine, human = run_dir / "03_raw", run_dir / "02_machine", run_dir / "01_human"
    machine.mkdir(parents=True, exist_ok=True)
    human.mkdir(parents=True, exist_ok=True)

    sweep = dedupe(read_csv(raw / "sweep.csv"), ("table", "column", "requested", "compression"))
    winners = sweep_winners(sweep)
    tables = dedupe(read_csv(raw / "tables.csv"), ("table", "plan", "compression"))
    reads = {r["label"]: r for r in dedupe(read_csv(raw / "reads.csv"), ("label",))}
    ablation = dedupe(
        read_csv(raw / "ablation.csv"), ("table", "column", "requested", "page_rows", "fragment_rows")
    )
    env_path = machine / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}

    winner_rows = []
    for (table, column, codec), w in sorted(winners.items()):
        std, fl, best = w["best_standard"], w["fastlanes"], w["best_overall"]
        winner_rows.append(
            {
                "table": table,
                "column": column,
                "type": (std or fl or {}).get("type", ""),
                "compression": codec,
                "best_standard": std["resolved"] if std else "",
                "best_standard_bytes": int(std["bytes"]) if std else None,
                "fastlanes": fl["resolved"] if fl else "",
                "fastlanes_bytes": int(fl["bytes"]) if fl else None,
                "fastlanes_vs_best_standard_pct": (
                    100.0 * (int(fl["bytes"]) - int(std["bytes"])) / int(std["bytes"]) if std and fl else None
                ),
                "winner": best["resolved"] if best else "",
            }
        )

    for r in tables:
        read = reads.get(f"{r['table']}/{r['same_as'] or r['plan']}/{r['compression']}")
        r["read_ms"] = float(read["read_ms_median"]) if read else None

    totals: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in tables:
        t = totals.setdefault(r["compression"], {}).setdefault(
            r["plan"], {"bytes": 0, "rewrite_ms": 0.0, "read_ms": 0.0, "tables": 0}
        )
        t["bytes"] += int(r["output_bytes"])
        t["rewrite_ms"] += float(r["rewrite_ms"])
        t["read_ms"] += r["read_ms"] or 0.0
        t["tables"] += 1

    summary = {
        "environment": env,
        "sweep": sweep,
        "winners": winner_rows,
        "tables": tables,
        "totals": totals,
        "ablation": ablation,
    }
    (machine / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# TPC-H SF100 FastLanes benchmark summary", ""]
    if env:
        lines += [f"- GPU: {env.get('gpu', '')}", f"- CUDA: {env.get('nvcc', '')}", f"- cuDF commit: {env.get('git_commit', '')}", ""]
    lines += ["## Per-column winners", "", "| table | column | codec | best standard | bytes | FastLanes bytes | FastLanes vs best | winner |", "|---|---|---|---|---:|---:|---:|---|"]
    for w in winner_rows:
        pct = w["fastlanes_vs_best_standard_pct"]
        lines.append(
            f"| {w['table']} | {w['column']} | {w['compression']} | {w['best_standard']} | "
            f"{fmt_bytes(w['best_standard_bytes'] or 0)} | {fmt_bytes(w['fastlanes_bytes'] or 0)} | "
            f"{'' if pct is None else f'{pct:+.2f}%'} | {w['winner']} |"
        )
    lines += ["", "## Full-table plan totals", "", "| codec | plan | total size | rewrite time (s) | read time (s) |", "|---|---|---:|---:|---:|"]
    for codec, plans in totals.items():
        for plan in PLANS:
            if plan in plans:
                t = plans[plan]
                lines.append(
                    f"| {codec} | {plan} | {fmt_bytes(t['bytes'])} | {t['rewrite_ms'] / 1e3:.1f} | {t['read_ms'] / 1e3:.1f} |"
                )
    (human / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[summarize] wrote {machine / 'summary.json'} and {human / 'summary.md'}", flush=True)


def main() -> int:
    today = dt.date.today().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("artifacts/tpch100/sf100"))
    p.add_argument("--run-dir", type=Path, default=Path(f"artifacts/fastlanes_bench_sf100_{today}"))
    p.add_argument("--scratch-dir", type=Path, default=None, help="Full-table outputs (default: <run-dir>/03_raw/cases)")
    p.add_argument("--bench-bin", type=Path, default=Path("build/fastlanes_encoding_bench"))
    p.add_argument("--chunk-bin", type=Path, default=Path("build/parquet_io_chunk"))
    p.add_argument("--tables", default=",".join(TABLES))
    p.add_argument("--steps", default="sweep,tables,ablation,summarize")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64, help="parquet_io_chunk --batch-size")
    p.add_argument("--rgs-per-read", type=int, default=64, help="parquet_io_chunk --rgs-per-read")
    p.add_argument(
        "--hang-timeout",
        type=int,
        default=600,
        help="fastlanes_encoding_bench --hang-timeout-s: longest single write/read before it is recorded as hung",
    )
    p.add_argument("--keep-outputs", action="store_true", help="Keep full-table output files")
    args = p.parse_args()

    args.tables = [t for t in args.tables.split(",") if t]
    args.data_dir = args.data_dir.resolve()
    args.bench_bin = args.bench_bin.resolve()
    args.chunk_bin = args.chunk_bin.resolve()
    run_dir = args.run_dir.resolve()
    raw = run_dir / "03_raw"
    raw.mkdir(parents=True, exist_ok=True)
    args.scratch_dir = (args.scratch_dir or raw / "cases").resolve()
    capture_environment(args, run_dir / "02_machine")

    steps = [s for s in args.steps.split(",") if s]
    t0 = time.perf_counter()
    if "sweep" in steps:
        step_sweep(args, raw)
    if "tables" in steps:
        step_tables(args, raw)
    if "ablation" in steps:
        step_ablation(args, raw)
    if "summarize" in steps:
        step_summarize(args, run_dir)
    print(f"[done] {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
