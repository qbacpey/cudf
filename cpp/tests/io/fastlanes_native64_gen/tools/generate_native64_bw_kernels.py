#!/usr/bin/env python3
"""Generate deterministic native64 kernel include from spec + template.

This script converts metadata rows from native64_bw_table.json into a concrete
test-only CUDA include used by parquet_fastlanes_native64_generated_test.cu.

Inputs:
- spec JSON (bw rows)
- template INL file

Outputs:
- generated/native64_bw_kernels.inl
- optional generated/native64_bw_anchor_snapshot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_spec(path: Path) -> list[dict]:
    """Load and validate contiguous bw rows [0..64] from spec."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("spec file missing rows list")
    if len(rows) != 65:
        raise SystemExit(f"expected 65 rows, got {len(rows)}")
    for expected_bw, row in enumerate(rows):
        bw = row.get("bw")
        if bw != expected_bw:
            raise SystemExit(f"rows must be sorted by bw and contiguous; expected {expected_bw}, got {bw}")
    return rows


def _render_rows(rows: list[dict]) -> str:
    """Render bw metadata table initializer lines for template substitution."""

    lines: list[str] = []
    for row in rows:
        bw = int(row["bw"])
        words_per_lane = int(row["words_per_lane"])
        words_per_vector = int(row["words_per_vector"])
        crossing_count = int(row.get("cross_boundary_map", {}).get("crossing_count", 0))
        lines.append(
            f"  bw_row{{{bw}u, {words_per_lane}u, {words_per_vector}u, {crossing_count}u}},"
        )
    return "\n".join(lines)


def _render_dispatch(symbol: str) -> str:
    """Render dispatch table entries for bw0..64 wrappers."""

    lines = [f"  &{symbol}<{bw}>," for bw in range(65)]
    return "\n".join(lines)


def _render(template_text: str, rows: list[dict]) -> str:
    """Apply all template substitutions and return final generated content."""

    rendered = template_text
    rendered = rendered.replace("{{ROW_INITIALIZERS}}", _render_rows(rows))
    rendered = rendered.replace("{{ENCODE_DISPATCH}}", _render_dispatch("encode_dispatch_wrapper"))
    rendered = rendered.replace("{{DECODE_DISPATCH}}", _render_dispatch("decode_dispatch_wrapper"))
    return rendered if rendered.endswith("\n") else rendered + "\n"


def _write_anchor_snapshot(rows: list[dict], out_path: Path) -> None:
    """Write compact anchor subset used for quick review/diff sanity checks."""

    anchors = [0, 1, 31, 32, 33, 37, 63, 64]
    payload = {
        "anchors": [row for row in rows if int(row["bw"]) in anchors],
        "anchor_bitwidths": anchors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entrypoint supporting generate mode and --check mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor-out", type=Path, required=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = _load_spec(args.spec)
    template_text = args.template.read_text(encoding="utf-8")
    rendered = _render(template_text, rows)

    if args.check:
        # Check mode is for CI/validation to ensure generated file is committed and up to date.
        if not args.out.exists():
            raise SystemExit(f"missing generated file: {args.out}")
        current = args.out.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit("generated include is out of date; regenerate it")
        if args.anchor_out is not None and not args.anchor_out.exists():
            raise SystemExit(f"missing anchor snapshot: {args.anchor_out}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")

    if args.anchor_out is not None:
        _write_anchor_snapshot(rows, args.anchor_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
