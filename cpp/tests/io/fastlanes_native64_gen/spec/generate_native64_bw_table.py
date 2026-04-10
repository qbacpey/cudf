#!/usr/bin/env python3
"""Generate deterministic native64 bitwidth metadata for test artifacts.

This script is part of the native64 test generation toolchain.
It does not affect production runtime encode/decode logic directly.

Primary output:
- native64_bw_table.json containing bw0..64 metadata rows and a content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VECTOR_SIZE = 1024
LANES_PER_VECTOR = 16
VALUES_PER_LANE = 64
MIN_BW = 0
MAX_BW = 64
PREDICATE = "bit_off > (64 - bw)"


def encoded_size_bytes(vector_size: int, bw: int) -> int:
    """Return packed size in bytes for vector_size values at bitwidth bw."""

    return (vector_size * bw + 7) // 8


def crossing_indices(bw: int) -> list[int]:
    """Return lane-local value indices where bitfields cross a 64-bit word boundary."""

    if bw <= 0:
        return []
    threshold = 64 - bw
    out: list[int] = []
    for i in range(VALUES_PER_LANE):
        bit_off = (i * bw) & 63
        if bit_off > threshold:
            out.append(i)
    return out


def build_entry(bw: int) -> dict:
    """Build one metadata row for a single bitwidth."""

    words_per_vector = encoded_size_bytes(VECTOR_SIZE, bw) // 8
    words_per_lane = words_per_vector // LANES_PER_VECTOR
    crossings = crossing_indices(bw)
    return {
        "bw": bw,
        "words_per_lane": words_per_lane,
        "words_per_vector": words_per_vector,
        "cross_boundary_map": {
            "predicate": PREDICATE,
            "crossing_value_indices": crossings,
            "crossing_count": len(crossings),
        },
    }


def build_payload() -> dict:
    """Build full deterministic payload for all bitwidths.

    rows_sha256 fingerprints canonical row content to detect accidental drift.
    """

    entries = [build_entry(bw) for bw in range(MIN_BW, MAX_BW + 1)]
    rows_json = json.dumps(entries, separators=(",", ":"), sort_keys=True)
    rows_sha256 = hashlib.sha256(rows_json.encode("utf-8")).hexdigest()

    return {
        "schema_version": "1.0.0",
        "owner": "qbacpey",
        "vector_layout": {
            "vector_size": VECTOR_SIZE,
            "lanes_per_vector": LANES_PER_VECTOR,
            "values_per_lane": VALUES_PER_LANE,
        },
        "bitwidth_range": {"min": MIN_BW, "max": MAX_BW},
        "rows_sha256": rows_sha256,
        "rows": entries,
    }


def main() -> int:
    """CLI entrypoint supporting generate mode and --check mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate output matches the regenerated content",
    )
    args = parser.parse_args()

    payload = build_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.check:
        # Check mode is intended for CI/validation to enforce deterministic output.
        if not args.out.exists():
            raise SystemExit(f"missing output file: {args.out}")
        current = args.out.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit("native64_bw_table.json is out of date; regenerate it")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
