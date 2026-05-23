# How to produce a FastLanes-encoded Parquet file with cuDF

cuDF's Parquet writer ships two FastLanes encodings that you can request
per-column via `cudf::io::column_encoding`:

| Encoding name              | Physical type | Encode locality |
| -------------------------- | ------------- | --------------- |
| `FASTLANE_BITPACK_RAW`     | INT32         | CPU encode      |
| `FASTLANES_DELTA_BINARY`   | INT64         | GPU encode      |

A third encoding, `FASTLANE_BITPACK_SPLIT64`, is **deprecated** and hard-
refused at write time (see Limitations).

## 1. Build cuDF + the `parquet_io_chunk` example

From the cuDF repo root:

    cmake --build cpp/build --target cudf -j 8
    cmake --install cpp/build
    cmake --build cpp/examples/parquet_io/build -j 8

The example binary lands at:

    cpp/examples/parquet_io/build/parquet_io_chunk

## 2. Produce a FastLanes-encoded file

`parquet_io_chunk` takes a per-column encoding map. The map MUST cover
every column in the input schema; columns that are not eligible for
FastLanes (strings, decimals, etc.) should fall back to `DICTIONARY` or
another standard encoding. Example for TPC-H `lineitem_sf1.parquet`:

    cd cpp/examples/parquet_io

    build/parquet_io_chunk \
        artifacts/tpch100/lineitem_sf1.parquet \
        out_fastlanes.parquet \
        "l_orderkey:FASTLANES_DELTA_BINARY,\
         l_partkey:FASTLANES_DELTA_BINARY,\
         l_suppkey:FASTLANES_DELTA_BINARY,\
         l_linenumber:FASTLANES_DELTA_BINARY,\
         l_quantity:DICTIONARY,\
         l_extendedprice:DICTIONARY,\
         l_discount:DICTIONARY,\
         l_tax:DICTIONARY,\
         l_returnflag:DICTIONARY,\
         l_linestatus:DICTIONARY,\
         l_shipdate:FASTLANE_BITPACK_RAW,\
         l_commitdate:FASTLANE_BITPACK_RAW,\
         l_receiptdate:FASTLANE_BITPACK_RAW,\
         l_shipinstruct:DICTIONARY,\
         l_shipmode:DICTIONARY,\
         l_comment:DICTIONARY" \
        SNAPPY \
        --batch-size=8 --enable-v2-headers

Notable flags:

- `--enable-v2-headers` -- use Parquet V2 data-page headers. Recommended for
  FastLanes since the encoded payload format is V2-friendly.
- `--batch-size=N`      -- number of row groups processed in parallel.
- `--skip-validation`   -- skip the built-in round-trip read used to verify
  the just-written file. Leave this off if you want the writer to confirm
  decodability before exiting.
- `--enable-log` / `--log-file=PATH` -- write a per-row-group log.

Available compression codecs: `NONE, AUTO, SNAPPY, LZ4, ZSTD, CASCADED,
BITCOMP, GDEFLATE, ANS`.

## 3. Validate against a baseline

To check that the FastLanes file decodes to the same data as a non-
FastLanes equivalent, generate a `DELTA_BINARY_PACKED` baseline and run
the included roundtrip checker:

    build/parquet_io_chunk \
        artifacts/tpch100/lineitem_sf1.parquet \
        out_baseline.parquet \
        "<same column map but with DELTA_BINARY_PACKED instead of FASTLANES_*>" \
        SNAPPY --batch-size=8 --enable-v2-headers

    python3 tools/roundtrip/parquet_io_roundtrip_check.py \
        --input out_baseline.parquet \
        --compare-other out_fastlanes.parquet \
        --validator cudf --allow-cudf-fallback

Expected output:

    PASS: parquet content is equal.

For the lineitem_sf1 spec above, the FastLanes file is ~15% smaller than
the DELTA baseline at SNAPPY compression
(132.7 MB vs 156.7 MB on lineitem_sf1).

## 4. Current limitations

- **`FASTLANE_BITPACK_SPLIT64` is deprecated and rejected at write time.**
  Requesting it throws `CUDF_FAIL("FASTLANE_BITPACK_SPLIT64 is deprecated
  due to a known sub-vector page-padding bug; use FASTLANES_DELTA_BINARY
  or DELTA_BINARY_PACKED for INT64 columns.")`. The decoder still
  recognizes the on-disk encoding so legacy files keep reading.
- **Type eligibility.**
  - `FASTLANE_BITPACK_RAW` only applies to columns whose Parquet physical
    type is INT32 (so int8/uint8/int16/uint16/int32/uint32, plus
    `date32[day]` and `duration` types backed by INT32).
  - `FASTLANES_DELTA_BINARY` only applies to columns whose Parquet
    physical type is INT64 (int64/uint64 and INT64-backed logical types).
  - Any other column type silently falls back to the encoder's default;
    a `[warning]` log line is emitted.
- **Every column must appear in the encoding map** when using
  `parquet_io_chunk`. There is no implicit fallback for missing columns.
- **Decimal128 columns** do not currently use FastLanes; assign them
  `DICTIONARY` (or another supported encoding) in the map.
- **No partial-file mixing of SPLIT64 and other encodings.** Since SPLIT64
  is refused before any rows are written, attempting it will abort the
  entire write, not just one column.

## 5. Reading the file back

A FastLanes-encoded file reads back with stock cuDF -- no special API:

    import cudf
    df = cudf.read_parquet("out_fastlanes.parquet")

`pyarrow` will list the per-page encoding as `UNKNOWN` because the
FastLanes on-disk encoding values are cuDF extensions; this is expected
and does not affect data correctness when read via cuDF.