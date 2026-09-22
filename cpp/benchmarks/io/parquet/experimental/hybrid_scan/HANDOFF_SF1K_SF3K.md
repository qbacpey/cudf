# Hands-off: run hybrid-scan multifile vs single-file benchmark at SF100 / SF1K / SF3K

You are an LLM agent on a machine with 8x NVIDIA RTX 6000 (48 GB each) and a large RAID array.
Your job: build the benchmark from a published branch, generate TPC-H lineitem at three scale
factors, run the single-file vs multi-file hybrid scan comparison, and produce a results table +
plot. Work autonomously; the steps below are battle-tested on a GH200 node.

IMPORTANT machine notes:
- Use ONE GPU only (`export CUDA_VISIBLE_DEVICES=0`). This is a single-GPU benchmark. Do NOT
  shard files across the 8 GPUs and do NOT run sweeps in parallel on different GPUs — they would
  share RAID bandwidth and page cache, contaminating the I/O you are measuring.
- RTX 6000 (Ada, sm_89) — build with `-DCMAKE_CUDA_ARCHITECTURES=NATIVE` (handled below).
- GDS may or may not be available; it does not matter. All reference numbers were produced with
  `KVIKIO_COMPAT_MODE=ON` (host-bounce I/O path, no GDS), which also avoids a libcufile teardown
  segfault at process exit. Keep it set so both machines use the identical I/O path.
- Absolute times will differ from the GH200 baseline (RTX 6000 has ~960 GB/s memory bandwidth vs
  ~4 TB/s HBM3). Only compare RATIOS between arms on the same machine, never absolute times
  across machines.

---

## 0. Context (what you are measuring and why)

The benchmark compares three ways to read N parquet files with cuDF's experimental hybrid scan:

- `SINGLE_SEQ` — a sequential loop of single-file `hybrid_scan_reader`s (1 thread, 1 stream)
- `SINGLE_POOL` — a `BS::thread_pool` of single-file readers, one forked CUDA stream per thread
- `MULTIFILE` — one `hybrid_scan_multifile` reader over all N sources (1 thread, 1 stream)

Headline metric is **wall-clock** throughput (`rows_per_sec_wall`). Do NOT report nvbench's
"GPU Time": kvikio device reads are not stream-ordered, so CUDA-event timing misses the I/O and
shows a physically impossible flat ~120 ms for MULTIFILE. The benchmark computes
`rows_per_sec_wall` from a manually measured wall clock with `cudaDeviceSynchronize()` before
reading the clock — that is the honest number.

Prior results on another GH200 (SF100, 23 GB, 600,037,902 rows, NARROW 4-column scan, cold page
cache), wall-clock seconds — use these as a sanity baseline:

| config | N=1024 (22 MB/file) | N=2048 (11 MB) | N=4096 (5.6 MB) |
|---|---|---|---|
| SINGLE_SEQ | 14.4 | 26.7 | 51.7 |
| SINGLE_POOL@12 | 1.84 | 3.54 | 6.95 |
| SINGLE_POOL@16 | 1.88 | 3.69 | 7.27 |
| SINGLE_POOL@24 | 1.96 | 3.86 | 7.76 |
| MULTIFILE | 1.49 | 1.80 | 2.67 |

SF200 (44 GB, 1,200,018,434 rows): N=1024 (43 MB/file): SEQ 15.5, POOL@12 2.31, MF 2.55 (pool
edges ahead — crossover ≈ 30–40 MB/file); N=4096 (11 MB/file): SEQ 53.5, POOL@12 7.06, MF 3.67.

---

## 1. Preflight checks (do these first, report results)

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv   # expect 8x RTX 6000, ~49140 MiB each
df -h /raid                                             # see disk budget below
nvcc --version                                          # CUDA 13.x toolchain
docker ps                                               # a RAPIDS/cudf dev container may already exist
```

Disk budget (parquet, snappy, 128 MB row-group target), per file-count point:
- SF100  ≈ 23 GB   (row count 600,037,902)
- SF1K   ≈ 230 GB  (≈6.0 B rows — verify exactly after generation)
- SF3K   ≈ 690 GB  (≈18.0 B rows — verify exactly after generation)

Recommended sweep per SF: N ∈ {256, 1024, 4096}. If free space ≥ 2.5 TB you may keep everything;
otherwise generate → benchmark → delete one SF3K point at a time (SF3K dominates). Never delete
anything you did not create without asking.

## 2. Get the code

```bash
git clone -b hybrid/benchmark-complitetaion git@github.com:qbacpey/cudf.git cudf-bench
cd cudf-bench
```

The branch contains everything: `cpp/benchmarks/io/parquet/experimental/hybrid_scan/hybrid_scan_multifile.cpp`,
`multifile_bench_common.{hpp,cpp}`, and the `HYBRID_SCAN_MULTIFILE_NVBENCH` target registered in
`cpp/benchmarks/CMakeLists.txt`.

## 3. Build

If a cudf dev container/env already exists on the machine, reuse it. Otherwise create the conda
environment from the repo's dependency files (see cpp/conda) or use the RAPIDS dev container,
then:

```bash
cmake -S cpp -B cpp/build/latest -GNinja \
  -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=NATIVE
cmake --build cpp/build/latest --target HYBRID_SCAN_MULTIFILE_NVBENCH -j$(nproc)
```

A full libcudf build takes 1–3 h on a Grace host (use sccache if available). The benchmark target
itself links in seconds once libcudf is built. Run the binary on the host (not inside a container
without GPU/GDS passthrough), with `LD_LIBRARY_PATH` pointing at the build tree and the conda env:

```bash
B=$PWD/cpp/build/latest
export LD_LIBRARY_PATH=$B:$B/lib:<conda-env>/lib
```

## 4. Install tpchgen-cli (data generator)

Either reuse velox-testing if present
(`benchmark_data_tools/scripts/install_tpchgen_cli.sh`, binary lands in
`benchmark_data_tools/.local_installs/bin/tpchgen-cli`), or:

```bash
cargo install tpchgen-cli --locked   # needs a Rust toolchain
```

Verify: `tpchgen-cli --version`.

## 5. Generate data (long — run in background, one point at a time)

```bash
ROOT=<large-raid-dir>/tpch_multifile; mkdir -p $ROOT
tpchgen-cli -s <SF> -T lineitem --parts <N> --format=parquet \
  --parquet-row-group-bytes $((128*1024*1024)) \
  --output-dir $ROOT/sf<SF>_p<N>
```

Directory layout is contractual: the benchmark globs `$CUDF_BENCH_TPCH_ROOT/sf<SF>_p<N>/lineitem/*.parquet`
and asserts the file count equals N. `<SF>` is the bare number: 100, 1000, 3000.

Approximate generation times on a GH200: SF100 ~5 min, SF1K ~30–60 min, SF3K ~1.5–3 h.

After each generation, record the exact row count (needed to convert throughput to seconds):

```bash
python3 -c "
import pyarrow.parquet as pq, glob
fs = sorted(glob.glob('$ROOT/sf<SF>_p<N>/lineitem/*.parquet'))
print(sum(pq.read_metadata(f).num_rows for f in fs))"
```

## 6. CRITICAL: memory ceiling at SF1K/SF3K — two read modes (already in the branch)

The benchmark has two modes, selected by `CUDF_BENCH_RELEASE_MODE` (default off):

- **Full mode** — one single hybrid scan read with everything resident on the GPU. MULTIFILE: one
  fully-coalesced fetch of all column chunks, one `materialize_all_columns`, whole table returned.
  Single-file arms: per file fetch+materialize, keep every table, then one final
  `concatenate_tables`. If the dataset does not fit in GPU memory it OOMs — that is intended:
  full mode crashing is the honest "this scale does not fit on this GPU" signal.
- **Release mode** (`=1`) — bounded-memory chunked read that cannot OOM. Uses the reader's native
  chunked-read APIs: `construct_row_group_passes(pass_read_limit)` bounds each fetch+decompress
  pass, and `setup_chunking_for_all_columns(chunk_read_limit)` + `materialize_all_columns_chunk()`
  bounds each output chunk; rows are counted and released immediately, no final table is retained.
  Peak device memory is bounded by the limits, not the dataset. Release mode times the ALL phase
  only (the sub-phase arms still buffer whole files; the benchmark skips them under release mode).

NARROW output ≈ 32 B/row decoded — which mode to use:

| SF | rows | NARROW output | fits in 48 GB (RTX 6000)? |
|---|---|---|---|
| 100 | 0.6 B | ~19 GB | yes — run as-is (full mode) |
| 1000 | 6 B | ~190 GB | **no — use release mode** |
| 3000 | 18 B | ~575 GB | **no — use release mode** |

**Both modes are already implemented in the branch** (commit `01b0ecdbed`) — no patching needed.
The run script in section 7 sets `CUDF_BENCH_RELEASE_MODE` automatically per SF.

Defaults (env-tunable): MULTIFILE pass 8 GiB / chunk 1 GiB (`CUDF_BENCH_MF_PASS_BYTES`,
`CUDF_BENCH_MF_CHUNK_BYTES`); single-file arms pass 512 MiB / chunk 256 MiB
(`CUDF_BENCH_SF_PASS_BYTES`, `CUDF_BENCH_SF_CHUNK_BYTES`) — smaller because up to num_threads
readers are in flight concurrently (24 threads x ~1.3 GiB ~= 31 GiB). If a pooled run OOMs, lower
`CUDF_BENCH_SF_PASS_BYTES` first.

Release mode was validated on the GH200 at SF100 N=1024 against full mode:

| arm | full rows | release rows | full peak mem | release peak mem | full wall | release wall |
|---|---|---|---|---|---|---|
| MULTIFILE | 600,037,902 | 600,037,902 | 32.2 GB | 2.7 GB | 1.53 s | 1.89 s |
| SINGLE_SEQ | 600,037,902 | 600,037,902 | 33.6 GB | 0.03 GB | 14.45 s | 15.09 s |
| SINGLE_POOL@16 | 600,037,902 | 600,037,902 | 33.6 GB | 0.40 GB | 1.89 s | 1.96 s |

Re-run this validation on your machine first: at SF100 N=1024 with and without
`CUDF_BENCH_RELEASE_MODE=1`, every arm must report identical `total_rows` (a CSV column) in both
modes. Do not proceed to SF1K/SF3K until that holds.

Reporting caveats (state them in the writeup): in release mode the single-file arms skip the final
concatenate and MULTIFILE issues one fetch round per pass instead of a single fully-coalesced
fetch. Both differences disfavor MULTIFILE, so release-mode speedups are conservative for the
multifile thesis. Label SF1K/SF3K points "release mode".

## 7. Run the sweep

Save this as `run_scale_point.sh` (it is NOT in the repo):

```bash
#!/bin/bash
# Usage: run_scale_point.sh <SF> <N> <tag>   e.g. run_scale_point.sh 1000 1024 sf1000_p1024
set -u
SF=$1; N=$2; TAG=$3
B=<repo>/cpp/build/latest
export LD_LIBRARY_PATH=$B:$B/lib:<conda-env>/lib
export CUDF_BENCH_TPCH_ROOT=<large-raid-dir>/tpch_multifile
export CUDF_BENCH_TPCH_SF=$SF
export KVIKIO_COMPAT_MODE=ON          # avoids a libcufile teardown segfault at process exit
export CUDF_BENCHMARK_DROP_CACHE=file # drop per-file page cache between samples (cold-cache honesty)
# Bounded-memory mode: REQUIRED for SF1K/SF3K (see section 6), OFF for SF100 comparability:
#   SF=100  -> unset or 0     SF=1000/3000 -> 1
export CUDF_BENCH_RELEASE_MODE=$([ "$SF" -ge 1000 ] && echo 1 || echo 0)
OUT=<results-dir>; mkdir -p $OUT

run() { # api threads
  local api=$1 thr=$2
  timeout 1800 $B/benchmarks/HYBRID_SCAN_MULTIFILE_NVBENCH \
    -a "num_files=$N" -a "columns=NARROW" -a "phase=ALL" -a "api=$api" \
    -a "num_threads=$thr" --min-samples 3 --csv /tmp/ls_${TAG}_${api}_${thr}.csv \
    > /tmp/ls_${TAG}_${api}_${thr}.log 2>&1
  echo "  $api thr=$thr exit=$?"
}
# num_threads MUST be 1 for non-pool arms (the benchmark skips them otherwise)
run SINGLE_SEQ 1
run SINGLE_POOL 12
run SINGLE_POOL 16
run SINGLE_POOL 24
run MULTIFILE 1
```

Notes:
- Convert CSV `rows_per_sec_wall` to seconds: `wall_s = exact_row_count / rows_per_sec_wall`.
- At SF3K the SEQ arm is slow (~3 min/sample at N=4096); you may drop it to `--min-samples 2`,
  and raise the `timeout` if any arm is killed (exit 124).
- Sanity gates: (a) every arm must report the same total row count; (b) at SF100 N=1024 all arms
  should complete without error and land within an order of magnitude of the GH200 baseline
  (MF ~1.5 s) — RTX 6000 will be slower, but a >10x discrepancy means something is wrong;
  investigate before proceeding.
- If the process segfaults at exit AFTER writing results, that is the known libcufile teardown
  bug — ensure `KVIKIO_COMPAT_MODE=ON` is set.

## 8. Deliverables

1. One CSV per (SF, N) point under the results dir.
2. A summary table (wall-clock seconds) with rows = the 5 configs, columns = scale points,
   per-file size annotated, plus a "MF speedup vs best pool" row.
3. A grouped-bar plot (matplotlib, linear y-axis starting at 0, plain numbers not 10^n,
   colorblind-friendly palette: seq #D55E00, pools #0072B2/#56B4E9/#9ECAE1, multifile #009E73),
   saved at 300 dpi.
4. A short writeup: where MULTIFILE wins, how the win scales with file count and SF, and the
   crossover file size (≈30–40 MB/file on GH200; it may shift on RTX 6000 since compute and I/O
   balance differently — report what you measure, not the expectation). State clearly which
   points used release mode (see section 6).
