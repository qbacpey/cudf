# Parquet Encoding Search Summary

- Input: /home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/CUDF-0003.roundtrip.parquet
- Binary: /home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/build/parquet_io_chunk
- Batch size: 2
- Search scope: fastlane-eligible
- Compressions: NONE, SNAPPY, ZSTD

## Best Result

- Compression: ZSTD
- Output size: 9.12 GB
- Output file: /home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/encoding_search_tpch100/best_validated_ZSTD.parquet
- Log file: /home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/encoding_search_tpch100/best_validated_ZSTD.log
- Validation in run: True

## Reproduction Command

/home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/build/parquet_io_chunk /home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/CUDF-0003.roundtrip.parquet /home/qchen/GPUFileFormat-cudf/cpp/examples/parquet_io/encoding_search_tpch100/best_validated_ZSTD.parquet l_orderkey:DELTA_BINARY_PACKED,l_partkey:DELTA_BINARY_PACKED,l_suppkey:DELTA_BINARY_PACKED,l_linenumber:DELTA_BINARY_PACKED,l_quantity:DICTIONARY,l_extendedprice:DICTIONARY,l_discount:DICTIONARY,l_tax:DICTIONARY,l_returnflag:DICTIONARY,l_linestatus:DICTIONARY,l_shipdate:DICTIONARY,l_commitdate:DICTIONARY,l_receiptdate:DICTIONARY,l_shipinstruct:FASTLANES_BITPACK,l_shipmode:FASTLANES_BITPACK ZSTD --batch-size=2

## Best Encoding Map

- l_orderkey: DELTA_BINARY_PACKED
- l_partkey: DELTA_BINARY_PACKED
- l_suppkey: DELTA_BINARY_PACKED
- l_linenumber: DELTA_BINARY_PACKED
- l_quantity: DICTIONARY
- l_extendedprice: DICTIONARY
- l_discount: DICTIONARY
- l_tax: DICTIONARY
- l_returnflag: DICTIONARY
- l_linestatus: DICTIONARY
- l_shipdate: DICTIONARY
- l_commitdate: DICTIONARY
- l_receiptdate: DICTIONARY
- l_shipinstruct: FASTLANES_BITPACK
- l_shipmode: FASTLANES_BITPACK
