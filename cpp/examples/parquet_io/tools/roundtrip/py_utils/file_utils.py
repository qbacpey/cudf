"""
File utility functions for Parquet compression verification.

Provides functions for file size calculation, human-readable formatting,
and directory management.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_file_size(filepath: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        File size in bytes, or 0 if file doesn't exist.
    """
    try:
        return filepath.stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def human_readable_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes.
        
    Returns:
        Human-readable string (e.g., "1.5 GB", "256 MB").
    """
    if size_bytes <= 0:
        return "0 B"
    
    units = [
        (1024 ** 4, "TB"),
        (1024 ** 3, "GB"),
        (1024 ** 2, "MB"),
        (1024, "KB"),
        (1, "B"),
    ]
    
    for threshold, unit in units:
        if size_bytes >= threshold:
            value = size_bytes / threshold
            if value >= 100:
                return f"{value:.0f} {unit}"
            elif value >= 10:
                return f"{value:.1f} {unit}"
            else:
                return f"{value:.2f} {unit}"
    
    return f"{size_bytes} B"


def generate_timestamp_string(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a timestamp string for filenames.
    
    Args:
        format_str: strftime format string.
        
    Returns:
        Formatted timestamp string.
    """
    return datetime.now().strftime(format_str)


def create_timestamped_dir(
    base_name: str = "roundtrip_test",
    parent: Optional[Path] = None,
) -> Path:
    """
    Create a directory with timestamp suffix.
    
    Args:
        base_name: Base name for the directory.
        parent: Parent directory (defaults to current directory).
        
    Returns:
        Path to the created directory.
    """
    timestamp = generate_timestamp_string()
    dir_name = f"{base_name}_{timestamp}"
    
    if parent:
        dir_path = parent / dir_name
    else:
        dir_path = Path.cwd() / dir_name
    
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def generate_log_filename(
    output_dir: Path,
    prefix: str = "verification",
) -> Path:
    """
    Generate a timestamped log filename.
    
    Args:
        output_dir: Directory for the log file.
        prefix: Prefix for the log filename.
        
    Returns:
        Path to the log file.
    """
    timestamp = generate_timestamp_string()
    return output_dir / f"{prefix}_{timestamp}.log"


def calculate_compression_ratio(
    original_size: int,
    compressed_size: int,
) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_size: Original file size in bytes.
        compressed_size: Compressed file size in bytes.
        
    Returns:
        Compression ratio (compressed/original), or 0.0 if invalid.
    """
    if original_size <= 0 or compressed_size <= 0:
        return 0.0
    return compressed_size / original_size


def calculate_space_savings(
    original_size: int,
    compressed_size: int,
) -> float:
    """
    Calculate space savings percentage.
    
    Args:
        original_size: Original file size in bytes.
        compressed_size: Compressed file size in bytes.
        
    Returns:
        Space savings as percentage (0-100), or 0.0 if invalid.
    """
    if original_size <= 0:
        return 0.0
    return (1.0 - compressed_size / original_size) * 100.0


def format_ratio(ratio: float) -> str:
    """
    Format compression ratio for display.
    
    Args:
        ratio: Compression ratio value.
        
    Returns:
        Formatted string (e.g., "0.45x").
    """
    if ratio <= 0:
        return "N/A"
    return f"{ratio:.4f}x"


def format_savings(savings: float) -> str:
    """
    Format space savings for display.
    
    Args:
        savings: Space savings percentage.
        
    Returns:
        Formatted string (e.g., "55.00%").
    """
    if savings <= 0:
        return "N/A"
    return f"{savings:.2f}%"


def format_time(seconds: float) -> str:
    """
    Format time duration for display.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted string (e.g., "1.23s").
    """
    if seconds < 0:
        return "N/A"
    return f"{seconds:.2f}s"


def ensure_parent_exists(filepath: Path) -> None:
    """
    Ensure the parent directory of a file exists.
    
    Args:
        filepath: Path to a file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
