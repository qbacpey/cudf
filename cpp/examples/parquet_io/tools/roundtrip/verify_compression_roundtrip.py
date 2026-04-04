#!/usr/bin/env python3
# filepath: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/GPUFileFormat-cudf/cpp/examples/parquet_io/tools/roundtrip/verify_compression_roundtrip.py
"""
Parquet Compression Roundtrip Verification Script

This script tests GPU-specific compression codecs by:
1. BASELINE: Create SNAPPY compressed version as baseline for comparison
2. FORWARD: Compress original parquet with GPU codec (e.g., CASCADED)
3. BACKWARD: Decompress and recompress with standard codec (e.g., SNAPPY)
4. VALIDATE: Compare original vs backward file to verify data integrity
5. COMPARE: Compare compression ratios and times against SNAPPY baseline

Inputs:
- input parquet path
- optional encoding/batch-size/codec list/validator options

Outputs:
- timestamped result directory with logs and summary tables
- pass/fail status for each tested codec

Recommended layout:
- place generated directories under 03_raw when integrating with the layered
    parquet_io artifacts convention.

Usage:
    python verify_compression_roundtrip.py <input.parquet> [options]

Example:
    python verify_compression_roundtrip.py data.parquet --batch-size=4
    python verify_compression_roundtrip.py data.parquet --encoding=PLAIN --validator=pyarrow
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from py_utils.config import (
    BASELINE_CODEC,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BINARY_PATH,
    DEFAULT_ENCODING,
    DEFAULT_GPU_CODECS,
    CompressionCodec,
    Config,
    ValidatorType,
)
from py_utils.file_utils import (
    calculate_compression_ratio,
    calculate_space_savings,
    create_timestamped_dir,
    format_ratio,
    format_savings,
    format_time,
    generate_log_filename,
    get_file_size,
    human_readable_size,
)
from py_utils.logging_utils import ColoredLogger, Colors
from py_utils.runner import ConversionResult, ParquetIORunner
from py_utils.validation import ValidationResult, ValidationStatus, validate_parquet_files

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[5]


def _resolve_worktree_name() -> str:
    cudf_home = os.environ.get("CUDF_HOME", "").strip()
    if cudf_home:
        return Path(cudf_home).name
    return REPO_ROOT.name


def _default_output_parent() -> Optional[Path]:
    shared_root = os.environ.get("PARQUET_IO_SHARED_ROOT", "").strip()
    if not shared_root:
        return None

    return (
        Path(shared_root).expanduser()
        / "artifacts"
        / _resolve_worktree_name()
        / "roundtrip"
    )


@dataclass
class CodecTestResult:
    """Results for a single codec test."""
    
    codec: CompressionCodec
    forward_result: Optional[ConversionResult] = None
    backward_result: Optional[ConversionResult] = None
    validation_result: Optional[ValidationResult] = None
    
    @property
    def status(self) -> str:
        """Get overall status string."""
        if self.validation_result and self.validation_result.passed:
            return "PASS"
        if self.forward_result and not self.forward_result.success:
            return "FAIL"
        if self.backward_result and not self.backward_result.success:
            return "FAIL"
        if self.validation_result:
            return str(self.validation_result.status.value)
        return "UNKNOWN"
    
    @property
    def passed(self) -> bool:
        """Check if the test passed."""
        return self.status == "PASS"
    
    @property
    def total_time(self) -> float:
        """Get total time for forward + backward passes."""
        time = 0.0
        if self.forward_result:
            time += self.forward_result.elapsed_seconds
        if self.backward_result:
            time += self.backward_result.elapsed_seconds
        return time
    
    @property
    def compressed_size(self) -> int:
        """Get the forward-compressed file size."""
        if self.forward_result:
            return self.forward_result.output_size
        return 0


def parse_arguments() -> Config:
    """Parse command-line arguments and return a Config object."""
    parser = argparse.ArgumentParser(
        description="Parquet Compression Roundtrip Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.parquet
  %(prog)s data.parquet --encoding=DELTA_BINARY_PACKED --batch-size=4
  %(prog)s data.parquet --output-dir=./my_test --validator=pyarrow
  %(prog)s data.parquet --encoding="col1:DELTA_BINARY_PACKED,col2:DICTIONARY"
        """,
    )
    
    # Positional arguments
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input Parquet file to test",
    )
    
    # Optional arguments
    parser.add_argument(
        "--encoding",
        "-e",
        dest="encoding_spec",
        default=str(DEFAULT_ENCODING),
        help=f"Encoding specification (default: {DEFAULT_ENCODING})",
    )
    
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parallelization level (default: {DEFAULT_BATCH_SIZE})",
    )
    
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output directory (default: auto-generated with timestamp; "
            "uses PARQUET_IO_SHARED_ROOT when set)"
        ),
    )
    
    parser.add_argument(
        "--log-file",
        "-l",
        type=Path,
        default=None,
        help="Log file path (default: auto-generated in output dir)",
    )
    
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(DEFAULT_BINARY_PATH),
        help=f"Path to parquet_io_chunk binary (default: {DEFAULT_BINARY_PATH})",
    )
    
    parser.add_argument(
        "--enable-v2-headers",
        action="store_true",
        help="Enable Parquet V2 data page headers",
    )
    
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        help="Enable page-level statistics",
    )
    
    parser.add_argument(
        "--validator",
        type=str,
        choices=["auto", "duckdb", "pyarrow"],
        default="auto",
        help="Validation method to use (default: auto)",
    )
    
    parser.add_argument(
        "--codecs",
        type=str,
        default=None,
        help="Comma-separated list of GPU codecs to test (default: all)",
    )
    
    args = parser.parse_args()
    
    # Parse codecs if specified
    gpu_codecs = list(DEFAULT_GPU_CODECS)
    if args.codecs:
        codec_names = [c.strip().upper() for c in args.codecs.split(",")]
        gpu_codecs = []
        for name in codec_names:
            try:
                gpu_codecs.append(CompressionCodec(name))
            except ValueError:
                print(f"Warning: Unknown codec '{name}', skipping")
    
    # Parse validator
    validator_map = {
        "auto": ValidatorType.AUTO,
        "duckdb": ValidatorType.DUCKDB,
        "pyarrow": ValidatorType.PYARROW,
    }
    validator = validator_map.get(args.validator, ValidatorType.AUTO)
    
    return Config(
        input_file=args.input_file,
        output_dir=args.output_dir,
        log_file=args.log_file,
        encoding_spec=args.encoding_spec,
        batch_size=args.batch_size,
        enable_v2_headers=args.enable_v2_headers,
        enable_stats=args.enable_stats,
        binary_path=args.binary,
        gpu_codecs=gpu_codecs,
        validator=validator,
    )


def print_header(config: Config, logger: ColoredLogger, input_size: int) -> None:
    """Print the script header with configuration information."""
    logger.header("=" * 50)
    logger.header("Parquet Compression Roundtrip Verification")
    logger.header("=" * 50)
    
    logger.metric("Input file", str(config.input_file))
    logger.metric("Input size", f"{human_readable_size(input_size)} ({input_size} bytes)")
    logger.metric("Encoding", config.encoding_spec)
    logger.metric("Baseline codec", str(config.baseline_codec))
    logger.metric("GPU codecs", ", ".join(str(c) for c in config.gpu_codecs))
    logger.metric("Batch size", str(config.batch_size))
    logger.metric("V2 Headers", "enabled" if config.enable_v2_headers else "disabled")
    logger.metric("Page Stats", "enabled" if config.enable_stats else "disabled")
    logger.metric("Validator", str(config.validator))
    logger.metric("Output dir", str(config.output_dir))
    logger.metric("Log file", str(config.log_file))
    logger.raw("")


def print_summary(
    config: Config,
    logger: ColoredLogger,
    input_size: int,
    baseline_result: ConversionResult,
    codec_results: Dict[CompressionCodec, CodecTestResult],
) -> None:
    """Print the final summary table."""
    logger.header("=" * 50)
    logger.header("VERIFICATION SUMMARY")
    logger.header("=" * 50)
    logger.raw("")
    
    # Count results
    total_tests = len(codec_results)
    passed_tests = sum(1 for r in codec_results.values() if r.passed)
    failed_tests = total_tests - passed_tests
    
    logger.info("Test Results:")
    logger.metric("Total tests", str(total_tests))
    logger.metric("Passed", str(passed_tests))
    logger.metric("Failed", str(failed_tests))
    logger.raw("")
    
    # Summary table
    logger.info(f"Compression Comparison (vs Original: {human_readable_size(input_size)}):")
    
    # Table header
    widths = [15, 12, 10, 12, 10, 8]
    headers = ["Codec", "Size", "Ratio", "Savings", "Time", "Status"]
    
    logger.table_row(headers, widths)
    logger.separator("-", sum(widths) + len(widths) * 3)
    
    # Baseline row
    baseline_ratio = calculate_compression_ratio(input_size, baseline_result.output_size)
    baseline_savings = calculate_space_savings(input_size, baseline_result.output_size)
    
    baseline_row = [
        f"{config.baseline_codec} (base)",
        human_readable_size(baseline_result.output_size),
        format_ratio(baseline_ratio),
        format_savings(baseline_savings),
        format_time(baseline_result.elapsed_seconds),
        "PASS" if baseline_result.success else "FAIL",
    ]
    logger.table_row(baseline_row, widths, Colors.CYAN)
    
    # Codec rows
    for codec in config.gpu_codecs:
        result = codec_results.get(codec)
        if not result:
            continue
        
        ratio = calculate_compression_ratio(input_size, result.compressed_size)
        savings = calculate_space_savings(input_size, result.compressed_size)
        
        color = Colors.GREEN if result.passed else Colors.RED
        
        row = [
            str(codec),
            human_readable_size(result.compressed_size) if result.compressed_size > 0 else "N/A",
            format_ratio(ratio) if ratio > 0 else "N/A",
            format_savings(savings) if savings > 0 else "N/A",
            format_time(result.total_time),
            result.status,
        ]
        logger.table_row(row, widths, color)
    
    logger.raw("")


def main() -> int:
    """Main entry point."""
    # Parse arguments
    config = parse_arguments()
    
    # Setup output directory
    if config.output_dir is None:
        config.output_dir = create_timestamped_dir(
            "roundtrip_test",
            parent=_default_output_parent(),
        )
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    if config.log_file is None:
        config.log_file = generate_log_filename(config.output_dir)
    
    # Validate configuration
    try:
        config.validate()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Initialize logger
    with ColoredLogger(log_file=config.log_file) as logger:
        # Get input file size
        input_size = get_file_size(config.input_file)
        
        # Print header
        print_header(config, logger, input_size)
        
        # Initialize runner
        runner = ParquetIORunner(config.binary_path, logger)
        
        # =====================================================================
        # Step 1: Create baseline
        # =====================================================================
        logger.header("Creating SNAPPY Baseline")
        
        baseline_result = runner.run_baseline_conversion(config)
        
        if baseline_result.success:
            ratio = calculate_compression_ratio(input_size, baseline_result.output_size)
            savings = calculate_space_savings(input_size, baseline_result.output_size)
            logger.metric("Compression ratio", format_ratio(ratio))
            logger.metric("Space savings", format_savings(savings))
            logger.metric("Time", format_time(baseline_result.elapsed_seconds))
            
            # Validate baseline
            baseline_validation = validate_parquet_files(
                config.input_file,
                baseline_result.output_file,
                "Baseline validation",
                config.validator,
                logger,
            )
            if baseline_validation.passed:
                logger.success("Baseline validation passed")
            else:
                logger.warning(f"Baseline validation: {baseline_validation.message}")
        else:
            logger.error("Baseline creation failed")
            return 1
        
        # =====================================================================
        # Step 2: Test GPU codecs
        # =====================================================================
        codec_results: Dict[CompressionCodec, CodecTestResult] = {}
        
        for codec in config.gpu_codecs:
            logger.header(f"Testing Codec: {codec}")
            
            test_result = CodecTestResult(codec=codec)
            
            # Forward pass
            logger.info(f"[FORWARD] Original -> {codec}")
            forward_result = runner.run_forward_conversion(config, codec)
            test_result.forward_result = forward_result
            
            if not forward_result.success:
                logger.error(f"Forward compression failed: {codec}")
                codec_results[codec] = test_result
                continue
            
            # Log forward metrics
            forward_ratio = calculate_compression_ratio(input_size, forward_result.output_size)
            forward_savings = calculate_space_savings(input_size, forward_result.output_size)
            vs_baseline = calculate_compression_ratio(
                baseline_result.output_size, forward_result.output_size
            )
            
            logger.metric("Forward compression ratio", format_ratio(forward_ratio))
            logger.metric("Forward space savings", format_savings(forward_savings))
            logger.metric("vs SNAPPY baseline", format_ratio(vs_baseline))
            
            # Backward pass
            logger.info(f"[BACKWARD] {codec} -> {config.baseline_codec}")
            backward_result = runner.run_backward_conversion(
                config, codec, forward_result.output_file
            )
            test_result.backward_result = backward_result
            
            if not backward_result.success:
                logger.error(f"Backward compression failed: {codec}")
                codec_results[codec] = test_result
                continue
            
            # Validate roundtrip
            logger.info("[VALIDATE] Comparing original vs roundtrip")
            validation_result = validate_parquet_files(
                config.input_file,
                backward_result.output_file,
                f"{codec} roundtrip",
                config.validator,
                logger,
            )
            test_result.validation_result = validation_result
            
            if validation_result.passed:
                logger.success(f"Roundtrip validation PASSED for codec: {codec}")
            else:
                logger.error(f"Roundtrip validation FAILED for codec: {codec}")
                logger.raw(f"  Reason: {validation_result.message}")
            
            # Log total time
            logger.metric("Total time", format_time(test_result.total_time))
            
            codec_results[codec] = test_result
            logger.raw("")
        
        # =====================================================================
        # Step 3: Print summary
        # =====================================================================
        print_summary(config, logger, input_size, baseline_result, codec_results)
        
        # Return exit code based on results
        all_passed = all(r.passed for r in codec_results.values())
        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())