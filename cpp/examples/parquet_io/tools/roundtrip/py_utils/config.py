"""
Configuration module for Parquet compression roundtrip verification.

Contains dataclasses, enums, and constants for codec configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class CompressionCodec(str, Enum):
    """Supported compression codecs for Parquet files."""
    
    # Standard codecs
    NONE = "NONE"
    SNAPPY = "SNAPPY"
    ZSTD = "ZSTD"
    LZ4 = "LZ4"
    
    # GPU-specific codecs (nvCOMP)
    CASCADED = "CASCADED"
    BITCOMP = "BITCOMP"
    GDEFLATE = "GDEFLATE"
    ANS = "ANS"
    
    def __str__(self) -> str:
        return self.value


class EncodingType(str, Enum):
    """Supported encoding types for Parquet columns."""
    
    PLAIN = "PLAIN"
    DICTIONARY = "DICTIONARY"
    DELTA_BINARY_PACKED = "DELTA_BINARY_PACKED"
    DELTA_LENGTH_BYTE_ARRAY = "DELTA_LENGTH_BYTE_ARRAY"
    DELTA_BYTE_ARRAY = "DELTA_BYTE_ARRAY"
    BYTE_STREAM_SPLIT = "BYTE_STREAM_SPLIT"
    
    def __str__(self) -> str:
        return self.value


class ValidatorType(str, Enum):
    """Validation method selection."""
    
    AUTO = "auto"      # Try DuckDB first, fallback to PyArrow
    DUCKDB = "duckdb"
    PYARROW = "pyarrow"
    
    def __str__(self) -> str:
        return self.value


# Default configuration constants
DEFAULT_GPU_CODECS: List[CompressionCodec] = [
    CompressionCodec.CASCADED,
    CompressionCodec.BITCOMP,
    CompressionCodec.GDEFLATE,
    CompressionCodec.ANS,
]

BASELINE_CODEC: CompressionCodec = CompressionCodec.SNAPPY

DEFAULT_ENCODING: EncodingType = EncodingType.DELTA_BINARY_PACKED

DEFAULT_BATCH_SIZE: int = 2

DEFAULT_BINARY_PATH: str = "./build/parquet_io_chunk"


@dataclass
class Config:
    """
    Configuration for the Parquet compression roundtrip verifier.
    
    Attributes:
        input_file: Path to the source Parquet file (required).
        output_dir: Directory for output files (auto-generated if not provided).
        log_file: Path to log file (auto-generated if not provided).
        encoding_spec: Encoding specification (column-specific or default).
        batch_size: Parallelization level for row group processing.
        enable_v2_headers: Enable Parquet V2 data page headers.
        enable_stats: Enable page-level statistics.
        skip_validation: Skip validation after each conversion.
        binary_path: Path to the parquet_io_chunk executable.
        gpu_codecs: List of GPU codecs to test.
        baseline_codec: Codec to use for baseline comparison.
        validator: Validation method to use.
    """
    
    input_file: Path
    output_dir: Optional[Path] = None
    log_file: Optional[Path] = None
    encoding_spec: str = str(DEFAULT_ENCODING)
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_v2_headers: bool = False
    enable_stats: bool = False
    skip_validation: bool = False
    binary_path: Path = Path(DEFAULT_BINARY_PATH)
    gpu_codecs: List[CompressionCodec] = field(default_factory=lambda: list(DEFAULT_GPU_CODECS))
    baseline_codec: CompressionCodec = BASELINE_CODEC
    validator: ValidatorType = ValidatorType.AUTO
    
    def __post_init__(self):
        """Validate and convert paths after initialization."""
        if isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        if isinstance(self.binary_path, str):
            self.binary_path = Path(self.binary_path)
    
    def validate(self) -> None:
        """
        Validate configuration and raise ValueError if invalid.
        
        Raises:
            ValueError: If required files don't exist or configuration is invalid.
            FileNotFoundError: If input file or binary doesn't exist.
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"parquet_io_chunk binary not found at: {self.binary_path}"
            )
        
        if not self.binary_path.is_file():
            raise ValueError(f"Binary path is not a file: {self.binary_path}")
        
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got: {self.batch_size}")
    
    def get_baseline_filename(self) -> str:
        """Generate filename for baseline output."""
        stem = self.input_file.stem
        return f"{self.baseline_codec}-BASELINE-{stem}.parquet"
    
    def get_forward_filename(self, codec: CompressionCodec) -> str:
        """Generate filename for forward compression output."""
        stem = self.input_file.stem
        return f"{codec}-FORWARD-{stem}.parquet"
    
    def get_backward_filename(self, codec: CompressionCodec) -> str:
        """Generate filename for backward compression output."""
        stem = self.input_file.stem
        return f"{codec}-BACKWARD-{stem}.parquet"