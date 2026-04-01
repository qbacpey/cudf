#!/bin/bash
# filepath: /home/qba/01_Sys_Hiwi/04_GPUFileFormat-cudf/GPUFileFormat-cudf/cpp/examples/parquet_io/tools/roundtrip/compare_parquet_pyarrow.sh

# =============================================================================
# Simple Parquet Comparison Script using PyArrow
# =============================================================================
#
# Compares two parquet files for data equality using PyArrow and Pandas
#
# Usage: ./compare_parquet_pyarrow.sh <file1.parquet> <file2.parquet>
# Output: prints schema/row diff details and exits 0 on identical content, 1 otherwise.
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

python3 << EOF
import pyarrow.parquet as pq
import pandas as pd
import sys

try:
    # Read both files
    table1 = pq.read_table("${FILE1}")
    table2 = pq.read_table("${FILE2}")
    
    df1 = table1.to_pandas()
    df2 = table2.to_pandas()
    
    print(f"File 1 rows: {len(df1)}")
    print(f"File 2 rows: {len(df2)}")
    
    # Check schema
    if table1.schema != table2.schema:
        print(f"\n⚠ Schema mismatch!")
        print(f"File 1 schema: {table1.schema}")
        print(f"File 2 schema: {table2.schema}")
    
    # Find differences using merge (anti-join equivalent)
    # Rows in df1 but not in df2
    merged = df1.merge(df2, how='outer', indicator=True)
    only_in_file1 = merged[merged['_merge'] == 'left_only']
    only_in_file2 = merged[merged['_merge'] == 'right_only']
    
    print(f"Rows only in File 1: {len(only_in_file1)}")
    print(f"Rows only in File 2: {len(only_in_file2)}")
    
    # Show sample differences if any
    if len(only_in_file1) > 0:
        print(f"\nSample rows only in File 1 (first 5):")
        print(only_in_file1.head().to_string())
    
    if len(only_in_file2) > 0:
        print(f"\nSample rows only in File 2 (first 5):")
        print(only_in_file2.head().to_string())
    
    # Final verdict
    if len(only_in_file1) == 0 and len(only_in_file2) == 0:
        print("\n✓ FILES ARE IDENTICAL")
        sys.exit(0)
    else:
        print("\n✗ FILES DIFFER")
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

result=$?
echo ""
echo "Comparison complete."
exit $result