#!/bin/bash
# filepath: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/GPUFileFormat-cudf/cpp/examples/parquet_io/tools/roundtrip/compare_parquet_duckdb.sh

# =============================================================================
# Simple Parquet Comparison Script using DuckDB
# =============================================================================
#
# Compares two parquet files for data equality using anti-join logic
#
# Usage: ./compare_parquet_duckdb.sh <file1.parquet> <file2.parquet>
# Output: prints row counts, symmetric differences, and final identical/different verdict.
#
# =============================================================================

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <file1.parquet> <file2.parquet>"
    exit 1
fi

FILE1="$1"
FILE2="$2"

if [ ! -f "$FILE1" ]; then
    echo "Error: File not found: $FILE1"
    exit 1
fi

if [ ! -f "$FILE2" ]; then
    echo "Error: File not found: $FILE2"
    exit 1
fi

echo "Comparing:"
echo "  File 1: $FILE1"
echo "  File 2: $FILE2"
echo ""

# Run DuckDB comparison
duckdb -c "
-- Create temporary tables
CREATE TABLE t1 AS SELECT * FROM read_parquet('${FILE1}');
CREATE TABLE t2 AS SELECT * FROM read_parquet('${FILE2}');

-- Show basic info
SELECT 'File 1 rows: ' || COUNT(*)::VARCHAR FROM t1;
SELECT 'File 2 rows: ' || COUNT(*)::VARCHAR FROM t2;

-- Find differences using EXCEPT (anti-join equivalent)
CREATE TABLE only_in_t1 AS SELECT * FROM t1 EXCEPT SELECT * FROM t2;
CREATE TABLE only_in_t2 AS SELECT * FROM t2 EXCEPT SELECT * FROM t1;

SELECT 'Rows only in File 1: ' || COUNT(*)::VARCHAR FROM only_in_t1;
SELECT 'Rows only in File 2: ' || COUNT(*)::VARCHAR FROM only_in_t2;

-- Final verdict
SELECT CASE 
    WHEN (SELECT COUNT(*) FROM only_in_t1) = 0 AND (SELECT COUNT(*) FROM only_in_t2) = 0
    THEN '✓ FILES ARE IDENTICAL'
    ELSE '✗ FILES DIFFER'
END AS result;
"

echo ""
echo "Comparison complete."