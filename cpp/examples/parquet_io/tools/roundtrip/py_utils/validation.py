"""
Validation module for Parquet file comparison.

Provides validation using DuckDB (primary) and PyArrow (fallback)
to verify data integrity after compression roundtrips.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .config import ValidatorType
from .logging_utils import ColoredLogger


class ValidationStatus(str, Enum):
    """Validation result status."""
    
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class ValidationResult:
    """
    Result of a validation operation.
    
    Attributes:
        status: Validation status (PASS, FAIL, SKIPPED, ERROR).
        method: Validation method used (duckdb, pyarrow, none).
        message: Descriptive message about the result.
        rows_file1: Number of rows in first file (if available).
        rows_file2: Number of rows in second file (if available).
        differences: Number of differing rows found (if applicable).
    """
    
    status: ValidationStatus
    method: str
    message: str
    rows_file1: Optional[int] = None
    rows_file2: Optional[int] = None
    differences: Optional[int] = None
    
    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASS


def _check_duckdb_available() -> bool:
    """Check if DuckDB is available."""
    try:
        import duckdb
        return True
    except ImportError:
        return False


def _check_pyarrow_available() -> bool:
    """Check if PyArrow is available."""
    try:
        import pyarrow.parquet
        return True
    except ImportError:
        return False


def validate_with_duckdb(
    file1: Path,
    file2: Path,
    logger: Optional[ColoredLogger] = None,
) -> ValidationResult:
    """
    Validate two Parquet files using DuckDB.
    
    Performs bidirectional EXCEPT queries to find any rows present
    in one file but not the other.
    
    Args:
        file1: Path to the first Parquet file (original).
        file2: Path to the second Parquet file (roundtrip result).
        logger: Optional logger for output.
        
    Returns:
        ValidationResult with the outcome.
    """
    if not _check_duckdb_available():
        return ValidationResult(
            status=ValidationStatus.SKIPPED,
            method="duckdb",
            message="DuckDB not available",
        )
    
    try:
        import duckdb
        
        conn = duckdb.connect(":memory:")
        
        # Load both tables
        conn.execute(f"CREATE TABLE original AS SELECT * FROM read_parquet('{file1}')")
        conn.execute(f"CREATE TABLE roundtrip AS SELECT * FROM read_parquet('{file2}')")
        
        # Get row counts
        rows1 = conn.execute("SELECT COUNT(*) FROM original").fetchone()[0]
        rows2 = conn.execute("SELECT COUNT(*) FROM roundtrip").fetchone()[0]
        
        if rows1 != rows2:
            conn.close()
            return ValidationResult(
                status=ValidationStatus.FAIL,
                method="duckdb",
                message=f"Row count mismatch: {rows1} vs {rows2}",
                rows_file1=rows1,
                rows_file2=rows2,
            )
        
        # Find differences using EXCEPT
        conn.execute("""
            CREATE TABLE diff_original AS 
            SELECT * FROM original 
            EXCEPT 
            SELECT * FROM roundtrip
        """)
        
        conn.execute("""
            CREATE TABLE diff_roundtrip AS
            SELECT * FROM roundtrip
            EXCEPT
            SELECT * FROM original
        """)
        
        diff_original = conn.execute("SELECT COUNT(*) FROM diff_original").fetchone()[0]
        diff_roundtrip = conn.execute("SELECT COUNT(*) FROM diff_roundtrip").fetchone()[0]
        
        conn.close()
        
        total_differences = diff_original + diff_roundtrip
        
        if total_differences == 0:
            return ValidationResult(
                status=ValidationStatus.PASS,
                method="duckdb",
                message=f"Tables are identical ({rows1} rows)",
                rows_file1=rows1,
                rows_file2=rows2,
                differences=0,
            )
        else:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                method="duckdb",
                message=f"Found {total_differences} differences",
                rows_file1=rows1,
                rows_file2=rows2,
                differences=total_differences,
            )
            
    except Exception as e:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            method="duckdb",
            message=f"DuckDB error: {str(e)}",
        )


def validate_with_pyarrow(
    file1: Path,
    file2: Path,
    logger: Optional[ColoredLogger] = None,
) -> ValidationResult:
    """
    Validate two Parquet files using PyArrow.
    
    Loads both files as tables and checks for equality.
    
    Args:
        file1: Path to the first Parquet file (original).
        file2: Path to the second Parquet file (roundtrip result).
        logger: Optional logger for output.
        
    Returns:
        ValidationResult with the outcome.
    """
    if not _check_pyarrow_available():
        return ValidationResult(
            status=ValidationStatus.SKIPPED,
            method="pyarrow",
            message="PyArrow not available",
        )
    
    try:
        import pyarrow.parquet as pq
        
        table1 = pq.read_table(str(file1))
        table2 = pq.read_table(str(file2))
        
        rows1 = table1.num_rows
        rows2 = table2.num_rows
        
        if rows1 != rows2:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                method="pyarrow",
                message=f"Row count mismatch: {rows1} vs {rows2}",
                rows_file1=rows1,
                rows_file2=rows2,
            )
        
        if table1.equals(table2):
            return ValidationResult(
                status=ValidationStatus.PASS,
                method="pyarrow",
                message=f"Tables are identical ({rows1} rows)",
                rows_file1=rows1,
                rows_file2=rows2,
                differences=0,
            )
        else:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                method="pyarrow",
                message="Tables have different content",
                rows_file1=rows1,
                rows_file2=rows2,
            )
            
    except Exception as e:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            method="pyarrow",
            message=f"PyArrow error: {str(e)}",
        )


def validate_parquet_files(
    file1: Path,
    file2: Path,
    description: str = "",
    validator: ValidatorType = ValidatorType.AUTO,
    logger: Optional[ColoredLogger] = None,
) -> ValidationResult:
    """
    Validate two Parquet files for equality.
    
    Uses the specified validator or auto-selects based on availability.
    
    Args:
        file1: Path to the first Parquet file (original).
        file2: Path to the second Parquet file (roundtrip result).
        description: Description of the validation for logging.
        validator: Which validation method to use.
        logger: Optional logger for output.
        
    Returns:
        ValidationResult with the outcome.
    """
    if logger:
        logger.info(f"Validating: {description}")
    
    # Check files exist
    if not file1.exists():
        return ValidationResult(
            status=ValidationStatus.ERROR,
            method="none",
            message=f"File not found: {file1}",
        )
    
    if not file2.exists():
        return ValidationResult(
            status=ValidationStatus.ERROR,
            method="none",
            message=f"File not found: {file2}",
        )
    
    # Use specified validator or auto-select
    if validator == ValidatorType.DUCKDB:
        return validate_with_duckdb(file1, file2, logger)
    elif validator == ValidatorType.PYARROW:
        return validate_with_pyarrow(file1, file2, logger)
    else:  # AUTO
        # Try DuckDB first
        result = validate_with_duckdb(file1, file2, logger)
        if result.status != ValidationStatus.SKIPPED:
            return result
        
        # Fallback to PyArrow
        result = validate_with_pyarrow(file1, file2, logger)
        if result.status != ValidationStatus.SKIPPED:
            return result
        
        # No validator available
        return ValidationResult(
            status=ValidationStatus.ERROR,
            method="none",
            message="No validation method available (install duckdb or pyarrow)",
        )