"""
Subprocess runner for the parquet_io_chunk binary.

Handles execution, timing, and output capture of the C++ compression tool.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .config import CompressionCodec, Config
from .file_utils import get_file_size
from .logging_utils import ColoredLogger


@dataclass
class ConversionResult:
    """
    Result of a parquet_io_chunk conversion operation.
    
    Attributes:
        success: Whether the conversion completed successfully.
        elapsed_seconds: Time taken for the conversion.
        output_size: Size of the output file in bytes.
        output_file: Path to the output file.
        stdout: Standard output from the process.
        stderr: Standard error from the process.
        return_code: Process return code.
    """
    
    success: bool
    elapsed_seconds: float
    output_size: int
    output_file: Path
    stdout: str
    stderr: str
    return_code: int
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000


class ParquetIORunner:
    """
    Runner for the parquet_io_chunk C++ binary.
    
    Handles building command-line arguments, executing the binary,
    capturing output, and measuring execution time.
    """
    
    def __init__(
        self,
        binary_path: Path,
        logger: Optional[ColoredLogger] = None,
    ):
        """
        Initialize the runner.
        
        Args:
            binary_path: Path to the parquet_io_chunk executable.
            logger: Optional logger for output.
        """
        self.binary_path = binary_path
        self.logger = logger
    
    def _build_command(
        self,
        input_file: Path,
        output_file: Path,
        encoding: str,
        compression: CompressionCodec,
        batch_size: Optional[int] = None,
        enable_v2_headers: bool = False,
        enable_stats: bool = False,
        skip_validation: bool = True,
    ) -> List[str]:
        """
        Build the command-line arguments for parquet_io_chunk.
        
        Args:
            input_file: Input Parquet file path.
            output_file: Output Parquet file path.
            encoding: Encoding specification.
            compression: Compression codec to use.
            batch_size: Optional batch size for parallelization.
            enable_v2_headers: Enable Parquet V2 headers.
            enable_stats: Enable page-level statistics.
            skip_validation: Skip validation after conversion.
            
        Returns:
            List of command-line arguments.
        """
        cmd = [
            str(self.binary_path),
            str(input_file),
            str(output_file),
            encoding,
            str(compression),
        ]
        
        if skip_validation:
            cmd.append("--skip-validation")
        
        if batch_size is not None:
            cmd.append(f"--batch-size={batch_size}")
        
        if enable_v2_headers:
            cmd.append("--enable-v2-headers")
        
        if enable_stats:
            cmd.append("--enable-stats")
        
        return cmd
    
    def run_conversion(
        self,
        input_file: Path,
        output_file: Path,
        encoding: str,
        compression: CompressionCodec,
        description: str = "",
        batch_size: Optional[int] = None,
        enable_v2_headers: bool = False,
        enable_stats: bool = False,
        skip_validation: bool = True,
    ) -> ConversionResult:
        """
        Run a Parquet conversion operation.
        
        Args:
            input_file: Input Parquet file path.
            output_file: Output Parquet file path.
            encoding: Encoding specification.
            compression: Compression codec to use.
            description: Description for logging.
            batch_size: Optional batch size for parallelization.
            enable_v2_headers: Enable Parquet V2 headers.
            enable_stats: Enable page-level statistics.
            skip_validation: Skip validation after conversion.
            
        Returns:
            ConversionResult with timing and status information.
        """
        cmd = self._build_command(
            input_file=input_file,
            output_file=output_file,
            encoding=encoding,
            compression=compression,
            batch_size=batch_size,
            enable_v2_headers=enable_v2_headers,
            enable_stats=enable_stats,
            skip_validation=skip_validation,
        )
        
        if self.logger:
            self.logger.info(f"Running: {description}")
            self.logger.raw(f"  Command: {' '.join(cmd)}", console=False)
        
        # Execute with timing
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            
            elapsed = time.perf_counter() - start_time
            output_size = get_file_size(output_file)
            
            success = result.returncode == 0 and output_file.exists() and output_size > 0
            
            # Log output to file
            if self.logger and result.stdout:
                self.logger.raw(result.stdout, console=False)
            if self.logger and result.stderr:
                self.logger.raw(result.stderr, console=False)
            
            if success:
                if self.logger:
                    self.logger.success(f"Conversion completed: {description}")
            else:
                if self.logger:
                    self.logger.error(f"Conversion failed: {description}")
                    if result.stderr:
                        self.logger.raw(f"  Error: {result.stderr.strip()}")
            
            return ConversionResult(
                success=success,
                elapsed_seconds=elapsed,
                output_size=output_size,
                output_file=output_file,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
            
        except FileNotFoundError:
            elapsed = time.perf_counter() - start_time
            error_msg = f"Binary not found: {self.binary_path}"
            if self.logger:
                self.logger.error(error_msg)
            
            return ConversionResult(
                success=False,
                elapsed_seconds=elapsed,
                output_size=0,
                output_file=output_file,
                stdout="",
                stderr=error_msg,
                return_code=-1,
            )
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            error_msg = f"Execution error: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            
            return ConversionResult(
                success=False,
                elapsed_seconds=elapsed,
                output_size=0,
                output_file=output_file,
                stdout="",
                stderr=error_msg,
                return_code=-1,
            )
    
    def run_baseline_conversion(
        self,
        config: Config,
    ) -> ConversionResult:
        """
        Run a baseline conversion using the configuration.
        
        Args:
            config: Configuration object.
            
        Returns:
            ConversionResult for the baseline conversion.
        """
        output_file = config.output_dir / config.get_baseline_filename()
        
        return self.run_conversion(
            input_file=config.input_file,
            output_file=output_file,
            encoding=config.encoding_spec,
            compression=config.baseline_codec,
            description=f"Baseline: {config.baseline_codec}",
            batch_size=config.batch_size,
            enable_v2_headers=config.enable_v2_headers,
            enable_stats=config.enable_stats,
            skip_validation=True,
        )
    
    def run_forward_conversion(
        self,
        config: Config,
        codec: CompressionCodec,
    ) -> ConversionResult:
        """
        Run a forward (GPU codec) compression.
        
        Args:
            config: Configuration object.
            codec: GPU codec to use.
            
        Returns:
            ConversionResult for the forward conversion.
        """
        output_file = config.output_dir / config.get_forward_filename(codec)
        
        return self.run_conversion(
            input_file=config.input_file,
            output_file=output_file,
            encoding=config.encoding_spec,
            compression=codec,
            description=f"Forward: Original -> {codec}",
            batch_size=config.batch_size,
            enable_v2_headers=config.enable_v2_headers,
            enable_stats=config.enable_stats,
            skip_validation=True,
        )
    
    def run_backward_conversion(
        self,
        config: Config,
        codec: CompressionCodec,
        forward_file: Path,
    ) -> ConversionResult:
        """
        Run a backward (standard codec) conversion.
        
        Args:
            config: Configuration object.
            codec: GPU codec being tested (for filename).
            forward_file: Path to the forward-converted file.
            
        Returns:
            ConversionResult for the backward conversion.
        """
        output_file = config.output_dir / config.get_backward_filename(codec)
        
        return self.run_conversion(
            input_file=forward_file,
            output_file=output_file,
            encoding=config.encoding_spec,
            compression=config.baseline_codec,
            description=f"Backward: {codec} -> {config.baseline_codec}",
            batch_size=config.batch_size,
            enable_v2_headers=config.enable_v2_headers,
            enable_stats=config.enable_stats,
            skip_validation=True,
        )