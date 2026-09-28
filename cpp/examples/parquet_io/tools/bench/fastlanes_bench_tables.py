#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the tables of the FastLanes TPC-H SF100 benchmark report from a run directory.

Reads the CSVs that run_tpch_sf100_fastlanes_bench.py writes under RUN_DIR/03_raw and prints the
markdown tables of docs/fastlanes/FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md. With --canvas-ts
it prints the data constants of docs/fastlanes/canvas/fastlanes-tpch-sf100-benchmark.canvas.tsx
instead. Labels are in Chinese to match the report. The report edits one table by hand: the columns
where a standard encoding always wins are grouped by kind, get a "特点" column, and the tiny-table
rows are rounded.

Usage: fastlanes_bench_tables.py RUN_DIR [--canvas-ts]
"""

import argparse
import csv
import sys
from pathlib import Path

STD = ["PLAIN", "DICTIONARY", "DELTA_BINARY_PACKED", "BYTE_STREAM_SPLIT"]
ENC = [*STD, "FASTLANES"]
CODECS = ["NONE", "SNAPPY", "ZSTD"]
TABLE_CODECS = ["SNAPPY", "ZSTD"]
SHORT = {
    "PLAIN": "PLAIN",
    "DICTIONARY": "DICT",
    "DELTA_BINARY_PACKED": "DELTA",
    "BYTE_STREAM_SPLIT": "BSS",
    "FASTLANES": "FastLanes",
}
TYPE = {"int64_t": "INT64", "int32_t": "INT32", "cudf::timestamp_D": "日期"}
TS_TYPE = {"int64_t": "INT64", "int32_t": "INT32", "cudf::timestamp_D": "date"}
TABLES = [
    "lineitem",
    "orders",
    "partsupp",
    "part",
    "customer",
    "supplier",
    "nation",
    "region",
]
LARGE_TABLES = ["lineitem", "orders", "partsupp"]
PLANS = [
    "cudf-default",
    "best-standard",
    "best-with-fastlanes",
    "fastlanes-all",
]
SIZE_PLANS = [
    "cudf-default",
    "best-standard",
    "fastlanes-all",
    "best-with-fastlanes",
]
MIN_ROWS = (
    1_000_000  # columns below this (nation, region) are left out of the totals
)


def load(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ok(r):
    return (
        r is not None
        and r["error"] == ""
        and r["validated"] == "1"
        and r["unexpected_pages"] == "0"
    )


def pct(a, b):
    return 100 * (a - b) / b


def fmt_pct(p):
    s = f"{p:+,.1f}%" if abs(p) < 1000 else f"{p:+,.0f}%"
    return s.replace("-", "−")


def fmt_rows(n):
    return f"{n / 1e6:,.0f}M" if n >= 1_000_000 else f"{n:,}"


def fmt_size(b):
    return f"{b / 1e6:,.1f} MB" if b >= 1e6 else f"{b:,} B"


def encoder_name(r):
    return "NATIVE64" if r["resolved"] == "FASTLANES_DELTA_BINARY" else "RAW32"


class Run:
    def __init__(self, run_dir):
        raw = Path(run_dir) / "03_raw"
        # A resumed sweep appends rows, so the last row for a key wins.
        self.sweep = {self._key(r): r for r in load(raw / "sweep.csv")}
        self.default_sweep = {
            self._key(r): r for r in load(raw / "sweep_default_fragments.csv")
        }
        self.tables = {
            (r["table"], r["compression"], r["plan"]): r
            for r in load(raw / "tables.csv")
        }
        self.reads = {r["label"]: r for r in load(raw / "reads.csv")}
        self.ablation = load(raw / "ablation.csv")
        if not self.sweep:
            sys.exit(f"no sweep results in {raw}")
        self.meta = {}
        for (t, c, _, _), r in self.sweep.items():
            self.meta.setdefault((t, c), r)
        self.columns = list(self.meta)
        # Per-column tables need FastLanes and one standard encoding per codec;
        # the totals need every encoding except DICTIONARY in every codec.
        self.complete = [
            (t, c)
            for t, c in self.columns
            if all(
                ok(self.row(t, c, codec, "FASTLANES"))
                and any(ok(self.row(t, c, codec, e)) for e in STD)
                for codec in CODECS
            )
        ]
        self.large = [
            (t, c)
            for t, c in self.complete
            if self.rows(t, c) >= MIN_ROWS
            and all(
                ok(self.row(t, c, codec, e))
                for codec in CODECS
                for e in ENC
                if e != "DICTIONARY"
            )
        ]
        self.table_names = [
            t
            for t in TABLES
            if all(
                (t, codec, p) in self.tables
                for codec in TABLE_CODECS
                for p in PLANS
            )
        ]

    @staticmethod
    def _key(r):
        return (r["table"], r["column"], r["compression"], r["requested"])

    def row(self, t, c, codec, e, default_layout=False):
        return (self.default_sweep if default_layout else self.sweep).get(
            (t, c, codec, e)
        )

    def bits(self, t, c, codec, e, default_layout=False):
        r = self.row(t, c, codec, e, default_layout)
        return float(r["bits_per_value"]) if ok(r) else None

    def best(self, t, c, codec, pool=STD, default_layout=False):
        """Smallest valid encoding for a column: (encoding, sweep row)."""
        cand = [(e, self.row(t, c, codec, e, default_layout)) for e in pool]
        cand = [(e, r) for e, r in cand if ok(r)]
        return min(cand, key=lambda er: int(er[1]["bytes"]))

    def rows(self, t, c):
        return int(self.meta[(t, c)]["rows"])

    def type(self, t, c):
        return TYPE[self.meta[(t, c)]["type"]]

    def aggregate(self, codec):
        """Totals over the large columns: fixed encodings, then the per-column best choices."""
        out = []
        choices = [
            (SHORT[e], [self.row(t, c, codec, e) for t, c in self.large])
            for e in ENC
            if e != "DICTIONARY"
        ]
        choices.append(
            ("best-std", [self.best(t, c, codec)[1] for t, c in self.large])
        )
        choices.append(
            (
                "best-all",
                [self.best(t, c, codec, ENC)[1] for t, c in self.large],
            )
        )
        for label, rows in choices:
            raw_bytes = sum(int(r["raw_bytes"]) for r in rows)
            out_bytes = sum(int(r["bytes"]) for r in rows)
            out.append(
                {
                    "label": label,
                    "bytes": out_bytes,
                    "bits": 8 * out_bytes / sum(int(r["rows"]) for r in rows),
                    "write_gbps": raw_bytes
                    / (sum(float(r["write_ms_median"]) for r in rows) * 1e-3)
                    / 1e9,
                    "read_gbps": raw_bytes
                    / (sum(float(r["read_ms_median"]) for r in rows) * 1e-3)
                    / 1e9,
                    "fastlanes_columns": sum(
                        1 for r in rows if r["requested"] == "FASTLANES"
                    ),
                }
            )
        return out

    def plan_row(self, t, codec, plan):
        return self.tables[(t, codec, plan)]

    def read_s(self, t, codec, plan):
        r = self.plan_row(t, codec, plan)
        return (
            float(
                self.reads[f"{t}/{r['same_as'] or plan}/{codec}"][
                    "read_ms_median"
                ]
            )
            / 1e3
        )


def table(headers, rows, align):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---:" if a == "r" else "---" for a in align) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def print_markdown(run):
    sweep = list(run.sweep.values())
    fallbacks = sum(1 for r in sweep if r["unexpected_pages"] != "0")
    print(
        f"逐列测试：{len(sweep)} 个结果，{sum(1 for r in sweep if r['error'])} 个错误，"
        f"{sum(1 for r in sweep if r['validated'] == '1')} 个校验通过，{fallbacks} 个 page 编码和请求不一致。\n"
    )
    skipped = [tc for tc in run.columns if tc not in run.complete]
    if skipped:
        print(
            "缺少有效的 FastLanes 或标准编码结果、没有列入逐列表格的列："
            + "、".join(f"`{t}.{c}`" for t, c in skipped)
            + "\n"
        )
    left_out = [
        tc
        for tc in run.complete
        if run.rows(*tc) >= MIN_ROWS and tc not in run.large
    ]
    if left_out:
        print(
            "有编码结果缺失或无效、没有计入汇总的列："
            + "、".join(f"`{t}.{c}`" for t, c in left_out)
            + "\n"
        )
    partial = sorted({t for t, _, _ in run.tables} - set(run.table_names))
    if partial:
        print(
            "整表结果不完整、没有列入整表表格的表："
            + "、".join(partial)
            + "\n"
        )

    winners, losers = [], []
    for t, c in run.complete:
        change = {
            codec: pct(
                run.bits(t, c, codec, "FASTLANES"),
                run.bits(t, c, codec, run.best(t, c, codec)[0]),
            )
            for codec in CODECS
        }
        (winners if any(v < 0 for v in change.values()) else losers).append(
            (t, c, change)
        )

    print("### 3.1 FastLanes 最小的列（至少在一种 codec 下）\n")
    rows = []
    for t, c, change in winners:
        e0 = run.best(t, c, "NONE")[0]
        cells = []
        for codec in CODECS:
            cell = fmt_pct(change[codec])
            if change[codec] < 0:
                cell = f"**{cell}**"
            e = run.best(t, c, codec)[0]
            cells.append(cell + (f"（{SHORT[e]}）" if e != e0 else ""))
        rows.append(
            [
                f"`{t}.{c}`",
                run.type(t, c),
                fmt_rows(run.rows(t, c)),
                f"{run.bits(t, c, 'NONE', 'FASTLANES'):.2f}",
                f"{SHORT[e0]} {run.bits(t, c, 'NONE', e0):.2f}",
                *cells,
            ]
        )
    table(
        [
            "列",
            "类型",
            "行数",
            "FastLanes，NONE",
            "最优标准，NONE",
            "FastLanes 相对最优：NONE",
            "SNAPPY",
            "ZSTD",
        ],
        rows,
        "llrrlrrr",
    )

    print(
        "### 3.1 标准编码始终更小的列（每值 bit 数，NONE / SNAPPY / ZSTD）\n"
    )
    rows = []
    for t, c, change in losers:
        best = [run.best(t, c, codec)[0] for codec in CODECS]
        values = [
            f"{run.bits(t, c, codec, e):.2f}"
            for codec, e in zip(CODECS, best, strict=True)
        ]
        if len(set(best)) == 1:
            best_cell = f"{SHORT[best[0]]} " + " / ".join(values)
        else:
            best_cell = " / ".join(
                f"{SHORT[e]} {v}" for e, v in zip(best, values, strict=True)
            )
        rows.append(
            [
                f"`{t}.{c}`",
                fmt_rows(run.rows(t, c)),
                " / ".join(
                    f"{run.bits(t, c, codec, 'FASTLANES'):.2f}"
                    for codec in CODECS
                ),
                best_cell,
                fmt_pct(change["SNAPPY"]),
            ]
        )
    table(
        ["列", "行数", "FastLanes", "最优标准", "FastLanes 相对最优，SNAPPY"],
        rows,
        "lrllr",
    )

    agg = {codec: run.aggregate(codec) for codec in CODECS}
    names = {
        "DELTA": "DELTA_BINARY_PACKED",
        "BSS": "BYTE_STREAM_SPLIT",
        "best-std": "每列选最优标准编码",
        "best-all": "每列选最优编码（含 FastLanes）",
    }
    print(
        f"### 3.3 适用列汇总（{len(run.large)} 个至少 {MIN_ROWS:,} 行的列）\n"
    )
    rows = []
    for i, a in enumerate(agg["NONE"]):
        cells = []
        for codec in CODECS:
            x = agg[codec][i]
            if x["label"] == "best-all":
                base = next(y for y in agg[codec] if y["label"] == "best-std")
                cells.append(
                    f"**{x['bytes'] / 1e9:.2f}**（{fmt_pct(pct(x['bytes'], base['bytes']))}，"
                    f"{x['fastlanes_columns']} 列用 FastLanes）"
                )
            else:
                cells.append(f"{x['bytes'] / 1e9:.2f} ({x['bits']:.2f})")
        rows.append([names.get(a["label"], a["label"]), *cells])
    table(["编码", "NONE：GB（每值 bit）", "SNAPPY", "ZSTD"], rows, "lrrr")

    print("### 3.4 速度（每秒处理的内存中列数据 GB 数）\n")
    rows = []
    for i, a in enumerate(agg["NONE"]):
        rows.append(
            [names.get(a["label"], a["label"])]
            + [f"{agg[codec][i]['write_gbps']:.2f}" for codec in CODECS]
            + [f"{agg[codec][i]['read_gbps']:.2f}" for codec in CODECS]
        )
    table(
        [
            "编码",
            "写入 NONE",
            "写入 SNAPPY",
            "写入 ZSTD",
            "读取 NONE",
            "读取 SNAPPY",
            "读取 ZSTD",
        ],
        rows,
        "lrrrrrr",
    )

    names_t = run.table_names
    large_t = [t for t in LARGE_TABLES if t in names_t]
    if names_t:
        for codec in TABLE_CODECS:
            print(f"### 3.5 文件大小，{codec}\n")
            rows, total = [], dict.fromkeys(SIZE_PLANS, 0)
            for t in names_t:
                size = {
                    p: int(run.plan_row(t, codec, p)["output_bytes"])
                    for p in SIZE_PLANS
                }
                for p in SIZE_PLANS:
                    total[p] += size[p]
                same = run.plan_row(t, codec, "best-with-fastlanes")["same_as"]
                last = (
                    "与 best-standard 相同"
                    if same
                    else fmt_pct(
                        pct(size["best-with-fastlanes"], size["best-standard"])
                    )
                )
                rows.append(
                    [t] + [fmt_size(size[p]) for p in SIZE_PLANS] + [last]
                )
            rows.append(
                [f"**{len(names_t)} 张表合计**"]
                + [f"**{total[p] / 1e9:.3f} GB**" for p in SIZE_PLANS]
                + [
                    f"**{fmt_pct(pct(total['best-with-fastlanes'], total['best-standard']))}**"
                ]
            )
            table(
                ["表", *SIZE_PLANS, "best-with-fastlanes 相对 best-standard"],
                rows,
                "lrrrrr",
            )

        print(f"### 3.5 时间（单位 s；方案顺序为 {' / '.join(PLANS)}）\n")
        rows, totals = [], {}
        for codec in TABLE_CODECS:
            rewrite = {
                t: [
                    float(run.plan_row(t, codec, p)["rewrite_ms"]) / 1e3
                    for p in PLANS
                ]
                for t in names_t
            }
            read = {
                t: [run.read_s(t, codec, p) for p in PLANS] for t in names_t
            }
            totals[codec] = (
                [
                    sum(rewrite[t][i] for t in names_t)
                    for i in range(len(PLANS))
                ],
                [sum(read[t][i] for t in names_t) for i in range(len(PLANS))],
            )
            for t in large_t:
                rows.append(
                    [
                        t,
                        codec,
                        " / ".join(f"{x:.1f}" for x in rewrite[t]),
                        " / ".join(f"{x:.2f}" for x in read[t]),
                    ]
                )
        for codec, (rewrite, read) in totals.items():
            rows.append(
                [
                    f"{len(names_t)} 张表合计",
                    codec,
                    " / ".join(f"{x:.1f}" for x in rewrite),
                    " / ".join(f"{x:.2f}" for x in read),
                ]
            )
        table(["表", "Codec", "重写", "热缓存读取"], rows, "llll")

        shares = []
        for codec in TABLE_CODECS:
            eligible = sum(
                int(run.best(t, c, codec)[1]["bytes"])
                for t, c in run.complete
                if t in names_t
            )
            total = sum(
                int(run.plan_row(t, codec, "best-standard")["output_bytes"])
                for t in names_t
            )
            shares.append(f"{codec} 文件的 {100 * eligible / total:.1f}%")
        print(f"适用列占 {'、'.join(shares)}（按最优标准编码计）。\n")

    if run.default_sweep:
        print(
            "### 3.6 Page 布局：cuDF 默认 fragment 对比正好 20,480 行的 page\n"
        )
        rows = []
        for codec in CODECS:
            fl, std_bits = [], {e: 0.0 for e in STD}
            for t, c in run.large:
                a, d = (
                    run.row(t, c, codec, "FASTLANES"),
                    run.row(t, c, codec, "FASTLANES", True),
                )
                if not ok(d):
                    continue
                fl.append(pct(float(a["bytes"]), float(d["bytes"])))
                for e in STD:
                    ba, bd = (
                        run.bits(t, c, codec, e),
                        run.bits(t, c, codec, e, True),
                    )
                    if (
                        ba is not None
                        and bd is not None
                        and abs(ba - bd) > abs(std_bits[e])
                    ):
                        std_bits[e] = ba - bd
            if not fl:
                continue
            fl.sort()
            rows.append(
                [
                    codec,
                    f"{fmt_pct(fl[0])} 到 {fmt_pct(fl[-1])}（中位数 {fmt_pct(fl[len(fl) // 2])}）",
                ]
                + [f"{std_bits[e]:+.3f}".replace("-", "−") for e in STD]
            )
        table(
            ["Codec", "FastLanes 大小变化"]
            + [f"{SHORT[e]} 最大变化（每值 bit）" for e in STD],
            rows,
            "llrrrr",
        )
        flips = []
        for t, c in run.complete:
            for codec in CODECS:
                if not ok(run.row(t, c, codec, "FASTLANES", True)) or not any(
                    ok(run.row(t, c, codec, e, True)) for e in STD
                ):
                    continue
                aligned = pct(
                    run.bits(t, c, codec, "FASTLANES"),
                    run.bits(t, c, codec, run.best(t, c, codec)[0]),
                )
                e = run.best(t, c, codec, default_layout=True)[0]
                default = pct(
                    run.bits(t, c, codec, "FASTLANES", True),
                    run.bits(t, c, codec, e, True),
                )
                if (aligned < 0) != (default < 0):
                    flips.append(
                        f"`{t}.{c}` {codec}：对齐 {fmt_pct(aligned)}，默认布局 {fmt_pct(default)}"
                    )
        print(
            "两种布局下胜负翻转的列："
            + ("；".join(flips) if flips else "无")
            + "\n"
        )

    if run.ablation:
        print("### 3.6 三种布局的对比，SNAPPY\n")
        rows = []
        for r in run.ablation:
            if not ok(r):
                continue
            page, frag = int(r["page_rows"]), int(r["fragment_rows"] or 0)
            setting = (
                f"{page:,} 行，fragment {frag:,} 行"
                if frag
                else f"{page:,} 行上限，默认 fragment"
            )
            enc = (
                f"FastLanes {encoder_name(r)}"
                if r["resolved"].startswith("FASTLANE")
                else SHORT.get(r["resolved"], r["resolved"])
            )
            rows.append(
                [
                    f"`{r['table']}.{r['column']}`",
                    enc,
                    setting,
                    f"{int(r['rows']) / int(r['data_pages']):,.0f}",
                    f"{float(r['bits_per_value']):.2f}",
                    f"{float(r['write_gbps']):.2f}",
                    f"{float(r['read_gbps']):.2f}",
                ]
            )
        table(
            [
                "列",
                "编码",
                "Page 设置",
                "每页行数",
                "每值 bit",
                "写入 GB/s",
                "读取 GB/s",
            ],
            rows,
            "lllrrrr",
        )

    print(
        "### 附录 A. 各编码的每值 bit 数（每行最小的值加粗；n/a：请求 DICTIONARY 但 cuDF 写的是别的编码）\n"
    )
    for codec in CODECS:
        print(f"**{codec}**\n")
        rows = []
        for t, c in run.columns:
            v = {e: run.bits(t, c, codec, e) for e in ENC}
            if all(x is None for x in v.values()):
                continue
            smallest = min(x for x in v.values() if x is not None)
            cells = [
                "n/a"
                if x is None
                else (f"**{x:,.2f}**" if x == smallest else f"{x:,.2f}")
                for x in v.values()
            ]
            rows.append(
                [
                    f"`{t}.{c}`",
                    run.type(t, c),
                    fmt_rows(run.rows(t, c)),
                    *cells,
                ]
            )
        table(
            ["列", "类型", "行数"] + [SHORT[e] for e in ENC], rows, "llrrrrrr"
        )

    print("### 附录 B. 各列速度，SNAPPY（GB/s）\n")
    rows = []
    for t, c in run.large:
        fl = run.row(t, c, "SNAPPY", "FASTLANES")
        e, best = run.best(t, c, "SNAPPY")
        where = "GPU" if encoder_name(fl) == "NATIVE64" else "CPU"
        rows.append(
            [
                f"`{t}.{c}`",
                f"{encoder_name(fl)}（{where}）",
                f"{float(fl['write_gbps']):.2f}",
                f"{float(best['write_gbps']):.2f} ({SHORT[e]})",
                f"{float(fl['read_gbps']):.2f}",
                f"{float(best['read_gbps']):.2f}",
            ]
        )
    table(
        [
            "列",
            "FastLanes 编码器",
            "FastLanes 写入",
            "最优标准写入",
            "FastLanes 读取",
            "最优标准读取",
        ],
        rows,
        "llrrrr",
    )


def print_canvas_ts(run):
    out = ["const COLUMNS: ColumnInfo[] = ["]
    for t, c in run.large:
        r = run.row(t, c, "NONE", "FASTLANES")
        out.append(
            f'  {{ name: "{t}.{c}", type: "{TS_TYPE[r["type"]]}", rows: {int(r["rows"])}, encoder: "{encoder_name(r)}" }},'
        )
    out.append("];\n")

    out.append("const BITS: Record<Codec, Record<Enc, (number | null)[]>> = {")
    for codec in CODECS:
        out.append(f"  {codec}: {{")
        for e in ENC:
            vals = [run.bits(t, c, codec, e) for t, c in run.large]
            out.append(
                f"    {SHORT[e]}: [{', '.join('null' if v is None else f'{v:.3f}' for v in vals)}],"
            )
        out.append("  },")
    out.append("};\n")

    out.append("const SPEED: Record<Codec, Record<Enc, Speed[]>> = {")
    for codec in CODECS:
        out.append(f"  {codec}: {{")
        for e in ENC:
            cells = []
            for t, c in run.large:
                r = run.row(t, c, codec, e)
                w, rd = (
                    (float(r["write_gbps"]), float(r["read_gbps"]))
                    if ok(r)
                    else (0, 0)
                )
                cells.append(f"{{ write: {w:.2f}, read: {rd:.2f} }}")
            out.append(f"    {SHORT[e]}: [{', '.join(cells)}],")
        out.append("  },")
    out.append("};\n")

    labels = {"best-std": "最优标准", "best-all": "最优（含 FastLanes）"}
    out.append("const AGGREGATE: Record<Codec, AggRow[]> = {")
    for codec in CODECS:
        out.append(f"  {codec}: [")
        for a in run.aggregate(codec):
            extra = (
                f", fastlanesColumns: {a['fastlanes_columns']}"
                if a["label"] == "best-all"
                else ""
            )
            out.append(
                f'    {{ label: "{labels.get(a["label"], a["label"])}", gb: {a["bytes"] / 1e9:.3f}, bits: {a["bits"]:.2f}, '
                f"write: {a['write_gbps']:.2f}, read: {a['read_gbps']:.2f}{extra} }},"
            )
        out.append("  ],")
    out.append("};\n")

    names_t = run.table_names
    if names_t:
        out.append(
            'const PLAN_TOTALS: Record<"SNAPPY" | "ZSTD", PlanTotal[]> = {'
        )
        for codec in TABLE_CODECS:
            out.append(f"  {codec}: [")
            for p in PLANS:
                gb = (
                    sum(
                        int(run.plan_row(t, codec, p)["output_bytes"])
                        for t in names_t
                    )
                    / 1e9
                )
                rewrite = (
                    sum(
                        float(run.plan_row(t, codec, p)["rewrite_ms"])
                        for t in names_t
                    )
                    / 1e3
                )
                read = sum(run.read_s(t, codec, p) for t in names_t)
                out.append(
                    f'    {{ plan: "{p}", gb: {gb:.3f}, rewriteS: {rewrite:.1f}, readS: {read:.2f} }},'
                )
            out.append("  ],")
        out.append("};\n")

        out.append("const TABLE_ROWS: TableRow[] = [")
        for codec in TABLE_CODECS:
            for t in [t for t in LARGE_TABLES if t in names_t]:
                cells = [
                    f"{{ mb: {int(run.plan_row(t, codec, p)['output_bytes']) / 1e6:.1f}, "
                    f"rewriteS: {float(run.plan_row(t, codec, p)['rewrite_ms']) / 1e3:.1f}, readS: {run.read_s(t, codec, p):.2f} }}"
                    for p in PLANS
                ]
                out.append(
                    f'  {{ table: "{t}", codec: "{codec}", plans: [{", ".join(cells)}] }},'
                )
        out.append("];\n")

    if run.ablation:
        out.append("const ABLATION: AblationRow[] = [")
        for r in run.ablation:
            if not ok(r):
                continue
            enc = (
                "FastLanes"
                if r["resolved"].startswith("FASTLANE")
                else SHORT.get(r["resolved"], r["resolved"])
            )
            out.append(
                f'  {{ column: "{r["column"]}", encoding: "{enc}", rowsPerPage: {round(int(r["rows"]) / int(r["data_pages"]))}, '
                f"bits: {float(r['bits_per_value']):.2f}, write: {float(r['write_gbps']):.2f}, read: {float(r['read_gbps']):.2f} }},"
            )
        out.append("];")
    print("\n".join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "run_dir", type=Path, help="benchmark run directory (contains 03_raw/)"
    )
    p.add_argument(
        "--canvas-ts",
        action="store_true",
        help="print the canvas data constants instead of markdown",
    )
    args = p.parse_args()
    run = Run(args.run_dir)
    if args.canvas_ts:
        print_canvas_ts(run)
    else:
        print_markdown(run)


if __name__ == "__main__":
    main()
