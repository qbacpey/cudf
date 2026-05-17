# FastLanes Parquet Notes

## Current assumptions

- The parquet FastLanes path is only intended for flat, non-null `INT32` and `UINT32` data.
- Page-local normalization is used before bit-packing:
  - `UINT32`: encode `value - page_min`
  - `INT32`: encode signed deltas from `page_min`, then store them as `uint32_t`
- The internal FastLanes page header stores `min_value` so decode can reconstruct the original data.

## Trade-offs

- The host-side write path keeps type handling intentionally simple and queries the leaf type
  page-by-page instead of adding more metadata plumbing.
- If a normalized page still needs `32` bits, the writer fails fast.
- Per-page fallback to another parquet encoding is intentionally out of scope for this version.
- The expected workload is practical cuDF datasets rather than pathological full-range pages.

## Tail-page reservation behavior

- Generic parquet page sizing is based on row/fragment bytes and can be small for tiny tail pages.
- FastLanes page encoding is vector-based and pads each page to 1024-value vectors.
- Result: a tail page with very few rows can still produce a relatively large encoded blob.
- To avoid under-allocation, the writer reserves FastLanes bytes with a conservative bound:
  - `header_size + (num_vectors * 1024 * max_fastlanes_bitwidth / 8)`
  - where `num_vectors = ceil(num_leaf_values / 1024)`

## Why the bound is conservative

- The current bound uses a fixed upper bitwidth for the signed INT32 path.
- This constant is a safety guard, not a compression tuning choice.
- The code also validates at encode time that encoded blob size does not exceed reserved page size,
  and fails fast with a clear message if violated.
