#!/usr/bin/env python3
"""Simple parquet roundtrip/compare validation script.

This mirrors the high-level flow of cpp/examples/parquet_io/parquet_io.cpp:
1) read parquet
2) write parquet
3) read written parquet
4) validate equivalence

Validation backends:
- auto (default): fast path using DuckDB, fallback to PyArrow
- duckdb/pyarrow: force one fast backend
- cudf: strict in-memory cuDF dataframe comparison

FastLanes note:
- FASTLANES_BITPACK is currently signed-only for INT32 physical columns
- UINT logical types fall back to non-FastLanes encodings

Example:
  python3 parquet_io_roundtrip_check.py \
    --input /data/tpch/lineitem.parquet \
    --output /tmp/lineitem.roundtrip.parquet

    python3 parquet_io_roundtrip_check.py \
        --input /data/a.parquet \
        --compare-other /data/b.parquet
        
    ./cpp/examples/parquet_io/tools/roundtrip/parquet_io_roundtrip_check.py \
        --input 100lineitem.parquet \
        --conversion-engine cpp \
        --cpp-binary ./build/parquet_io_chunk \
        --validator auto
    
    ./parquet_io_roundtrip_check.py --input lineitem.parquet --conversion-engine cpp --cpp-binary ./build/parquet_io_chunk --validator auto --keep-output --encoding-spec "l_orderkey:DELTA_BINARY_PACKED,\
        l_partkey:DELTA_BINARY_PACKED,\
        l_suppkey:DELTA_BINARY_PACKED,\
        l_linenumber:DICTIONARY,\
        l_quantity:DICTIONARY,\
        l_extendedprice:DELTA_BINARY_PACKED,\
        l_discount:DICTIONARY,\
        l_tax:DICTIONARY,\
        l_returnflag:DICTIONARY,\
        l_linestatus:DICTIONARY,\
        l_shipdate:DICTIONARY,\
        l_commitdate:DICTIONARY,\
        l_receiptdate:DELTA_BINARY_PACKED,\
        l_shipinstruct:DICTIONARY,\
        l_shipmode:DICTIONARY,\
        l_comment:DICTIONARY"     
        
    ./parquet_io_roundtrip_check.py \
        --input CUDF-0003.parquet \
        --conversion-engine cpp \
        --cpp-binary ./build/parquet_io_chunk \
        --validator auto \
        --keep-output \
        --cpp-show-log \
        --cpp-enable-log \
        --encoding-spec \
            "l_orderkey:DELTA_BINARY_PACKED,\
            l_partkey:DELTA_BINARY_PACKED,\
            l_suppkey:DELTA_BINARY_PACKED,\
            l_linenumber:DICTIONARY,\
            l_quantity:DICTIONARY,\
            l_extendedprice:DELTA_BINARY_PACKED,\
            l_discount:DICTIONARY,\
            l_tax:DICTIONARY,\
            l_returnflag:DICTIONARY,\
            l_linestatus:DICTIONARY,\
            l_shipdate:DICTIONARY,\
            l_commitdate:DICTIONARY,\
            l_receiptdate:DELTA_BINARY_PACKED,\
            l_shipinstruct:DICTIONARY,\
            l_shipmode:DICTIONARY,\
            l_comment:DICTIONARY" 
        
    ./parquet_io_roundtrip_check.py \
        --input CUDF-0003.parquet \
        --conversion-engine cpp \
        --cpp-binary ./build/parquet_io_chunk \
        --validator auto \
        --keep-output \
        --cpp-show-log \
        --cpp-enable-log \
        --allow-cudf-fallback \
        --encoding-spec \
            "l_orderkey:DELTA_BINARY_PACKED,\
            l_partkey:DELTA_BINARY_PACKED,\
            l_suppkey:DELTA_BINARY_PACKED,\
            l_linenumber:DICTIONARY,\
            l_quantity:DICTIONARY,\
            l_extendedprice:DELTA_BINARY_PACKED,\
            l_discount:DICTIONARY,\
            l_tax:DICTIONARY,\
            l_returnflag:FASTLANES_BITPACK,\
            l_linestatus:FASTLANES_BITPACK,\
            l_shipdate:DICTIONARY,\
            l_commitdate:DICTIONARY,\
            l_receiptdate:DELTA_BINARY_PACKED,\
            l_shipinstruct:FASTLANES_BITPACK,\
            l_shipmode:FASTLANES_BITPACK,\
            l_comment:DICTIONARY"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cudf

try:
    from cudf.testing import assert_eq as cudf_assert_eq
except Exception:  # pragma: no cover - fallback for older cuDF packaging
    cudf_assert_eq = None

try:
    from py_utils.config import ValidatorType
    from py_utils.validation import ValidationStatus, validate_parquet_files
except Exception:  # pragma: no cover - keep script usable without optional deps
    ValidatorType = None
    ValidationStatus = None
    validate_parquet_files = None


def _default_output_path(input_path: str) -> str:
    p = Path(input_path)
    suffix = "".join(p.suffixes) if p.suffixes else ".parquet"
    stem = p.name[: -len(suffix)] if suffix else p.name
    return str(p.with_name(f"{stem}.roundtrip{suffix}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read/write/read parquet or compare two parquet files. "
            "Use --validator auto/duckdb/pyarrow for faster file-level checks."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input parquet file (e.g. a TPC-H parquet file). "
            "In compare mode, this is the left-hand parquet path."
        ),
    )
    parser.add_argument(
        "--compare-other",
        default=None,
        help="Path to another parquet file to compare content against --input.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to output parquet file for round-trip mode. "
            "Default: <input>.roundtrip.parquet"
        ),
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep output file after validation (default: remove it).",
    )
    parser.add_argument(
        "--validator",
        default="auto",
        choices=["auto", "duckdb", "pyarrow", "cudf"],
        help=(
            "Validation backend: auto/duckdb/pyarrow use faster file-level validation; "
            "cudf uses strict DataFrame equality. Default: auto"
        ),
    )
    parser.add_argument(
        "--conversion-engine",
        default="auto",
        choices=["auto", "cudf", "cpp"],
        help=(
            "Roundtrip conversion backend: auto (choose based on file size), "
            "cudf (read/write full DataFrame), cpp (chunked parquet_io_chunk). "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--auto-cpp-threshold-gb",
        type=float,
        default=4.0,
        help=(
            "In auto mode, use chunked C++ conversion when input file size is >= this many GB. "
            "Default: 4.0"
        ),
    )
    parser.add_argument(
        "--cpp-binary",
        default="./build/parquet_io_chunk",
        help="Path to parquet_io_chunk executable for chunked conversion.",
    )
    parser.add_argument(
        "--encoding-spec",
        default="DELTA_BINARY_PACKED",
        help=(
            "Encoding argument passed to parquet_io_chunk (default: DELTA_BINARY_PACKED). "
            "FASTLANES_BITPACK currently applies to signed INT32 logical classes only."
        ),
    )
    parser.add_argument(
        "--compression",
        default="SNAPPY",
        help="Compression argument passed to parquet_io_chunk (default: SNAPPY).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size passed to parquet_io_chunk in chunked mode (default: 2).",
    )
    parser.add_argument(
        "--enable-v2-headers",
        action="store_true",
        help="Pass --enable-v2-headers to parquet_io_chunk in chunked mode.",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        help="Pass --enable-stats to parquet_io_chunk in chunked mode.",
    )
    parser.add_argument(
        "--cpp-skip-validation",
        action="store_true",
        help=(
            "Pass --skip-validation to parquet_io_chunk. By default, cpp mode keeps "
            "the C++ row-group validation enabled."
        ),
    )
    parser.add_argument(
        "--cpp-use-python-validation",
        action="store_true",
        help=(
            "After cpp conversion, also run Python validation path (--validator). "
            "Default: disabled to avoid custom-encoding/OOM issues."
        ),
    )
    parser.add_argument(
        "--cpp-show-log",
        action="store_true",
        help="Stream parquet_io_chunk stdout/stderr directly to console.",
    )
    parser.add_argument(
        "--cpp-enable-log",
        action="store_true",
        help="Pass --enable-log to parquet_io_chunk so it writes a log file.",
    )
    parser.add_argument(
        "--cpp-log-file",
        default=None,
        help="Pass --log-file=PATH to parquet_io_chunk.",
    )
    parser.add_argument(
        "--allow-cudf-fallback",
        action="store_true",
        help=(
            "Allow cuDF DataFrame fallback compare when fast validators are unavailable. "
            "Disabled by default to avoid OOM on very large files."
        ),
    )
    return parser.parse_args()


def _should_use_cpp_conversion(args: argparse.Namespace, input_path: str) -> bool:
    if args.conversion_engine == "cpp":
        return True
    if args.conversion_engine == "cudf":
        return False

    # auto mode: switch to chunked C++ conversion for large files.
    try:
        size_bytes = os.path.getsize(input_path)
    except OSError:
        return False

    threshold_bytes = int(max(args.auto_cpp_threshold_gb, 0.0) * (1024**3))
    return size_bytes >= threshold_bytes


def _run_cpp_chunk_conversion(args: argparse.Namespace, input_path: str, output_path: str) -> tuple[bool, str, float]:
    binary_path = args.cpp_binary
    if not os.path.isfile(binary_path):
        return False, f"C++ chunked binary not found: {binary_path}", 0.0

    cmd = [
        binary_path,
        input_path,
        output_path,
        args.encoding_spec,
        args.compression,
        f"--batch-size={args.batch_size}",
    ]
    if args.cpp_skip_validation:
        cmd.append("--skip-validation")
    if args.enable_v2_headers:
        cmd.append("--enable-v2-headers")
    if args.enable_stats:
        cmd.append("--enable-stats")
    if args.cpp_enable_log:
        cmd.append("--enable-log")
    if args.cpp_log_file:
        cmd.append(f"--log-file={args.cpp_log_file}")

    t0 = time.perf_counter()
    try:
        run_kwargs = {"text": True, "check": False}
        if args.cpp_show_log:
            proc = subprocess.run(cmd, **run_kwargs)
            stdout_text = ""
            stderr_text = ""
        else:
            proc = subprocess.run(cmd, capture_output=True, **run_kwargs)
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
    except OSError as exc:
        return False, f"Failed to execute chunked binary: {exc}", 0.0

    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        if args.cpp_show_log:
            detail = "see parquet_io_chunk console output above"
        else:
            detail = stderr_text.strip() or stdout_text.strip() or "unknown error"
        return False, f"parquet_io_chunk failed (code={proc.returncode}): {detail}", elapsed

    if not os.path.isfile(output_path):
        return False, "parquet_io_chunk completed but output file not found", elapsed

    return True, "ok", elapsed


def _run_fast_validation(left_path: str, right_path: str, validator_name: str) -> tuple[str, str]:
    """Run file-level validation with py_utils (DuckDB/PyArrow).

    Returns:
        (status, message) where status is one of:
        - "pass": validation succeeded
        - "fail": validation found content mismatch
        - "skip": fast validator unavailable or not requested
        - "error": validator errored; caller may fallback to cuDF
    """
    if validator_name == "cudf":
        return "skip", "cuDF validator requested"

    if ValidatorType is None or ValidationStatus is None or validate_parquet_files is None:
        return "skip", "fast validator modules unavailable; falling back to cuDF"

    validator_map = {
        "auto": ValidatorType.AUTO,
        "duckdb": ValidatorType.DUCKDB,
        "pyarrow": ValidatorType.PYARROW,
    }

    result = validate_parquet_files(
        Path(left_path),
        Path(right_path),
        validator=validator_map[validator_name],
    )
    message = f"{result.method}: {result.message}"

    if result.status == ValidationStatus.PASS:
        return "pass", message
    if result.status == ValidationStatus.FAIL:
        return "fail", message
    if result.status == ValidationStatus.SKIPPED:
        return "skip", message
    return "error", message


def compare_dataframes(left_df: cudf.DataFrame, right_df: cudf.DataFrame) -> tuple[bool, str]:
    """Compare content of two cuDF dataframes (not just metadata)."""
    if len(left_df) != len(right_df):
        return False, f"row count mismatch left={len(left_df)} right={len(right_df)}"
    if list(left_df.columns) != list(right_df.columns):
        return False, "column names/order mismatch"
    if [str(t) for t in left_df.dtypes] != [str(t) for t in right_df.dtypes]:
        return False, "dtype mismatch"

    if cudf_assert_eq is not None:
        try:
            cudf_assert_eq(left_df, right_df, check_dtype=True)
            return True, "content matches"
        except AssertionError as exc:
            detail = str(exc).splitlines()[0] if str(exc) else "content mismatch"
            return False, detail

    if left_df.equals(right_df):
        return True, "content matches"
    return False, "content mismatch"


def compare_two_parquet(
    left_path: str,
    right_path: str,
    validator: str,
    allow_cudf_fallback: bool,
) -> int:
    if not os.path.isfile(left_path):
        print(f"ERROR: left parquet file not found: {left_path}", file=sys.stderr)
        return 2
    if not os.path.isfile(right_path):
        print(f"ERROR: right parquet file not found: {right_path}", file=sys.stderr)
        return 2

    status, message = _run_fast_validation(left_path, right_path, validator)
    if status == "pass":
        print(f"PASS: parquet content is equal ({message}).")
        return 0
    if status == "fail":
        print(f"FAIL: parquet content differs ({message}).", file=sys.stderr)
        return 1
    if status in ("skip", "error"):
        print(f"WARN: fast validation not used ({message})")
        if not allow_cudf_fallback:
            print(
                "ERROR: cuDF fallback disabled (--allow-cudf-fallback not set). "
                "Use duckdb/pyarrow validation or enable fallback explicitly.",
                file=sys.stderr,
            )
            return 2
        print("      Falling back to cuDF dataframe comparison.")

    print(f"[1/2] Reading left parquet : {left_path}")
    t0 = time.perf_counter()
    left_df = cudf.read_parquet(left_path)
    t1 = time.perf_counter()
    print(
        f"      rows={len(left_df)} cols={len(left_df.columns)} "
        f"read_time={(t1 - t0) * 1000:.2f} ms"
    )

    print(f"[2/2] Reading right parquet: {right_path}")
    t2 = time.perf_counter()
    right_df = cudf.read_parquet(right_path)
    t3 = time.perf_counter()
    print(
        f"      rows={len(right_df)} cols={len(right_df.columns)} "
        f"read_time={(t3 - t2) * 1000:.2f} ms"
    )

    ok, msg = compare_dataframes(left_df, right_df)
    if not ok:
        print(f"FAIL: parquet content differs: {msg}", file=sys.stderr)
        return 1

    print("PASS: parquet content is equal.")
    return 0


def main() -> int:
    args = parse_args()
    input_path = args.input

    if args.compare_other:
        return compare_two_parquet(
            input_path,
            args.compare_other,
            args.validator,
            args.allow_cudf_fallback,
        )

    output_path = args.output or _default_output_path(input_path)

    if not os.path.isfile(input_path):
        print(f"ERROR: input parquet file not found: {input_path}", file=sys.stderr)
        return 2

    out_parent = os.path.dirname(output_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    use_cpp = _should_use_cpp_conversion(args, input_path)
    used_engine = "cpp" if use_cpp else "cudf"

    if use_cpp:
        print(f"[1/2] Converting via chunked C++ backend: {args.cpp_binary}")
        print(
            f"      encoding={args.encoding_spec} compression={args.compression} "
            f"batch_size={args.batch_size}"
        )
        print(
            "      cpp_validation="
            + ("disabled" if args.cpp_skip_validation else "enabled")
            + " cpp_log_stream="
            + ("on" if args.cpp_show_log else "off")
        )
        ok, msg, elapsed = _run_cpp_chunk_conversion(args, input_path, output_path)
        if not ok:
            print(f"FAIL: chunked conversion failed: {msg}", file=sys.stderr)
            return 1
        print(f"      convert_time={elapsed * 1000:.2f} ms")

        # Default behavior in cpp mode: trust parquet_io_chunk's internal
        # row-group validation and skip Python validation to support custom encodings.
        if not args.cpp_skip_validation and not args.cpp_use_python_validation:
            print("PASS: cpp chunked conversion + cpp internal validation succeeded.")
            if not args.keep_output:
                try:
                    os.remove(output_path)
                    print(f"Removed output file: {output_path}")
                except OSError as exc:
                    print(f"WARN: failed to remove output file: {exc}", file=sys.stderr)
            return 0
    else:
        print(f"[1/3] Reading input parquet: {input_path}")
        t0 = time.perf_counter()
        in_df = cudf.read_parquet(input_path)
        t1 = time.perf_counter()
        print(
            f"      rows={len(in_df)} cols={len(in_df.columns)} "
            f"read_time={(t1 - t0) * 1000:.2f} ms"
        )

        print(f"[2/3] Writing output parquet: {output_path}")
        t2 = time.perf_counter()
        in_df.to_parquet(output_path)
        t3 = time.perf_counter()
        print(f"      write_time={(t3 - t2) * 1000:.2f} ms")

    status, message = _run_fast_validation(input_path, output_path, args.validator)
    if status == "pass":
        print(f"PASS: parquet round-trip content checks passed ({used_engine}, {message}).")
    elif status == "fail":
        print(f"FAIL: parquet round-trip content differs ({message}).", file=sys.stderr)
        return 1
    else:
        print(f"WARN: fast validation not used ({message})")
        if not args.allow_cudf_fallback:
            print(
                "ERROR: cuDF fallback disabled (--allow-cudf-fallback not set). "
                "Use duckdb/pyarrow validation or enable fallback explicitly.",
                file=sys.stderr,
            )
            return 2

        if use_cpp:
            print("[2/2] Falling back to cuDF DataFrame comparison (may use high memory)")
            print(f"      reading input parquet: {input_path}")
            t_in0 = time.perf_counter()
            in_df = cudf.read_parquet(input_path)
            t_in1 = time.perf_counter()
            print(
                f"      rows={len(in_df)} cols={len(in_df.columns)} "
                f"read_time={(t_in1 - t_in0) * 1000:.2f} ms"
            )

            print(f"      reading output parquet: {output_path}")
            t_out0 = time.perf_counter()
            out_df = cudf.read_parquet(output_path)
            t_out1 = time.perf_counter()
            print(
                f"      rows={len(out_df)} cols={len(out_df.columns)} "
                f"read_time={(t_out1 - t_out0) * 1000:.2f} ms"
            )
        else:
            print(f"[3/3] Reading output parquet: {output_path}")
            t4 = time.perf_counter()
            out_df = cudf.read_parquet(output_path)
            t5 = time.perf_counter()
            print(
                f"      rows={len(out_df)} cols={len(out_df.columns)} "
                f"read_time={(t5 - t4) * 1000:.2f} ms"
            )

        ok, msg = compare_dataframes(in_df, out_df)
        if not ok:
            print(f"FAIL: parquet round-trip content differs: {msg}", file=sys.stderr)
            return 1

        print(f"PASS: parquet round-trip content checks passed ({used_engine}, cuDF fallback).")

    if not args.keep_output:
        try:
            os.remove(output_path)
            print(f"Removed output file: {output_path}")
        except OSError as exc:
            print(f"WARN: failed to remove output file: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
