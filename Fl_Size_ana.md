Now I have enough hard data to write a real analysis. Here is what's actually going on.

# FastLanes size analysis on `lineitem_sf1`

## TL;DR

My earlier "≈15 % size reduction" claim was measured against a **naïve baseline** that put `DELTA_BINARY_PACKED` on every numeric column — including ones with 2–11 distinct values where dictionary encoding obviously wins. Against the **tuned BEST baseline** (your spec), FastLanes' real, per-column contribution is:

| Variant (SNAPPY, V2 headers) | Total size | vs BEST baseline |
| --- | ---: | ---: |
| BEST baseline (DELTA + DICT) | 156.87 MiB | — |
| BEST + FastLanes on **all int64** (forced everywhere) | 159.34 MiB | **+1.6 % (worse)** |
| BEST + FastLanes on int64 **and** dates | 158.39 MiB | **+1.0 % (worse)** |
| BEST + FastLanes only on columns that **actually win** | **154.39 MiB** | **−1.58 % (better)** |

So the honest answer is: **FastLanes wins by ~1.5 % at most on `lineitem`** against a careful tuned baseline, not 15 %. The earlier 15 % was 90 % "DELTA was a bad choice on low-cardinality cols" and 10 % "FastLanes is good."

---

## Per-column attribution (where the bytes actually come from)

Using SNAPPY:

| Column | Distinct | BEST encoding | BEST size | FL encoding | FL size | Δ % |
|---|---:|---|---:|---|---:|---:|
| `l_orderkey` (sorted, monotonic) | 1.5 M | DELTA | **2.18 MB** | NATIVE64 | 6.38 MB | **+192 %** ⚠ |
| `l_partkey` (random) | 200 K | DELTA | 14.68 MB | NATIVE64 | **13.73 MB** | **−6.5 %** ✓ |
| `l_suppkey` (random) | 10 K | DELTA | 11.39 MB | NATIVE64 | **10.73 MB** | **−5.9 %** ✓ |
| `l_linenumber` | 7 | DICT | 1.80 MB | DICT | 1.80 MB | ±0 |
| `l_quantity` (decimal) | 50 | DICT | 4.57 MB | DICT | 4.57 MB | ±0 (not eligible) |
| `l_extendedprice` (decimal) | 934 K | DELTA | 18.56 MB | DELTA | 18.56 MB | ±0 (not eligible) |
| `l_discount` / `l_tax` (decimal) | 9–11 | DICT | 6.11 MB | DICT | 6.11 MB | ±0 (not eligible) |
| `l_returnflag` / `l_linestatus` | 2–3 | DICT | 2.26 MB | DICT | 2.26 MB | ±0 (string) |
| `l_shipdate` | 2 526 | DICT | 9.55 MB | RAW32 | **9.21 MB** | **−3.5 %** ✓ |
| `l_commitdate` | 2 466 | DICT | 9.53 MB | RAW32 | **9.21 MB** | **−3.4 %** ✓ |
| `l_receiptdate` | 2 554 | DICT | 9.55 MB | RAW32 | **9.21 MB** | **−3.5 %** ✓ |
| `l_shipinstruct` / `l_shipmode` / `l_comment` | string | DICT/DELTA_LENGTH | 74.24 MB | (unchanged) | 74.24 MB | ±0 |

Removing SNAPPY (pure encoding signal) confirms the same story for the FL-eligible columns: `l_partkey −4.5 %`, `l_suppkey −3.9 %`, dates `−1.3 %` each.

---

## Why FastLanes wins on `l_partkey/l_suppkey` but **loses 3× on `l_orderkey`**

Both encodings start the same way: subtract the previous value, then bit-pack the deltas. The crucial difference is the **packing granularity**:

```
DELTA_BINARY_PACKED:   blocks of 128 values, each with its own bit-width header
FASTLANES_DELTA_BINARY: fixed groups of 1024 values, ONE bit-width per group
```

- **`l_orderkey` is sorted-and-mostly-+1.** Deltas are nearly all 1, with occasional gaps of 7 (because orderkeys skip). DELTA's 128-element blocks can pack the all-1 runs at **1–2 bits/value**, with the occasional 3-bit block for the gap. FastLanes' 1024-element groups can't isolate the gap — a *single* value of 7 in a group forces the whole 1024-element group to ≥ 3 bits, killing the savings.
- **`l_partkey/l_suppkey` are random keys.** Deltas are large and noisy (no clusters of small deltas). Now DELTA's per-block headers and varint bit-widths are pure overhead, while FastLanes' bulk 1024-element pack is just leaner.
- **Date columns** (`l_shipdate` etc.) have ~2 500 distinct values within a tight range. DICTIONARY stores indices in 12-bit RLE and the keys table; FastLanes RAW32 packs the raw date32 values in ~13 bits with no indirection — a small but consistent win.

So the rule of thumb is:

> **FastLanes beats DELTA only when within-group delta variance is high (i.e. random, not sorted/monotonic). FastLanes beats DICTIONARY only when the value range is small enough to bit-pack tightly (≤ ~16 bits) on integers.**

---

## What you should report to your supervisor

1. **The headline number depends entirely on the baseline you compare to.** Calling FastLanes "15 % smaller than DELTA-everywhere" is technically true but misleading; the real gain over a well-tuned column-by-column encoding plan is **≤ 2 %** on `lineitem`.
2. **FastLanes is *not* a universal win for integer columns.** It is actively worse on sorted, monotonically-increasing keys (`l_orderkey`-shaped data), which is a very common pattern in OLAP datasets. A heuristic / per-column chooser is required.
3. **The dates win (−3 % each)** is the most reliable benefit we observed: FastLanes RAW32 consistently beats RLE_DICTIONARY for mid-cardinality bounded integers/dates.

---

## Suggested follow-up experiments

I deliberately did **not** kick off the long managed-memory runs against your `0003-20251024-184914.parquet` file — that file isn't in this workspace, and the runs would take much longer and require `LIBCUDF_USE_MANAGED_MEMORY=ON`. But the experiments worth running next, in order of value, are:

1. **Per-column "winner picker" sweep on SF10 / SF100.** Repeat the per-column table above for `lineitem_sf10.parquet` and `lineitem_sf100.parquet` to see whether the column-level winners flip at scale (they shouldn't, but it's worth verifying that the date win stays at ~3 %).
2. **Combine FastLanes with nvCOMP** (`CASCADED`, `BITCOMP`, `ZSTD`) instead of SNAPPY. FastLanes' bit-packed output has highly repetitive structure that `BITCOMP` / `CASCADED` can squeeze further, and this is where FastLanes might genuinely pull ahead.
3. **Try the `0003-*` file** (when you regenerate it on this host). With `LIBCUDF_USE_MANAGED_MEMORY=ON ./build/parquet_io_chunk … FASTLANES_DELTA_BINARY … BITCOMP`. The wider, less-structured columns in that workload are exactly where FastLanes' bulk packing has the most headroom over DELTA.
4. **A monotonic-key adapter** (future engineering work): inside the FastLanes encoder, detect monotonic groups and fall back to a tiny per-mini-block header like DELTA does. That would close the `l_orderkey` regression and likely turn the −1.6 % into something closer to −5 % overall.

If you want, I can run #1 (SF10 sweep) now and add it to the table — it's just another 30 s of GPU time on this box.