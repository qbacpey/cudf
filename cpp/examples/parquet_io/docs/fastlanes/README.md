# FastLanes 交付物说明（`fastlane-working` 分支）

这份文档说明 `fastlane-working` 分支里有什么、每个文件是做什么的，以及怎样在另一台机器上从头编译、测试、准备
TPC-H SF100 数据并重新跑一遍 benchmark。结论和分析在两份报告里：

- [FASTLANES_INTEGRATION_REPORT_2026-09-27.md](FASTLANES_INTEGRATION_REPORT_2026-09-27.md)：FastLanes 怎样接入
  cuDF 的 Parquet 读写，合并 26.10 时改了什么，测试覆盖情况和限制。
- [FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md](FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md)：TPC-H SF100 上
  FastLanes 和各种标准编码的大小、写入和读取速度对比。

结果的简要版本：FastLanes 在未排序的 key、数量和日期列上比最好的标准编码小 2-7%，读取也略快，但在有序 key、
低基数列上大很多，写入慢 7-12 倍。只在它占优的列上用，8 张表的文件在 SNAPPY 下小 1.1%，ZSTD 下小 0.6%。

## 1. 分支和提交

分支 `fastlane-working` 基于 cuDF `release/26.10`（上游 `eaa1e31058`）。在旧分支之上新增的提交：

| 提交 | 内容 |
| --- | --- |
| `c8533b8d3d` | 合并 `release/26.10`，解决 4 处冲突 |
| `9ed93250d2` | 把 FastLanes 代码适配到 26.10 的 API（`cuda::stream_ref`、新的 decode state 等） |
| `e0f905a0ae` | benchmark 工具：`fastlanes_encoding_bench`、驱动脚本、`parquet_io_chunk` 的新选项 |
| `0ce7f3b959` | 两份报告（英文版） |
| `9c904ca047` | 报告改为中文，并加入两个 Canvas 源文件 |
| `ad81576677` | 报告表格生成脚本 `fastlanes_bench_tables.py` |
| `ada223688e` | 驱动脚本按仓库的 ruff 规则格式化（不改行为） |
| `f72db93d34` | 驱动脚本的 `sweep_default` 步骤和启动检查，表格脚本对不完整结果的处理，`parquet_io_chunk` 帮助信息里补全编码列表 |
| 本文所在的提交 | 本 README，以及报告和 `cpp/examples/parquet_io/README.md` 里指向它的链接 |

合并前的状态保存在分支 `backup/fastlane-working-pre-26.10`（`0fae90d66c`）。

## 2. 文件清单

### 2.1 FastLanes 库代码

| 位置 | 内容 |
| --- | --- |
| `cpp/include/cudf/io/types.hpp`、`cpp/include/cudf/io/parquet_schema.hpp` | 编码枚举：`FASTLANE_BITPACK_RAW`、`FASTLANES_DELTA_BINARY`、`FASTLANE_BITPACK_SPLIT64`（已废弃），文件中的编码 ID 10-12 |
| `cpp/include/cudf/fastlanes/` | page header、大小计算、设备端 unpack、NATIVE64 kernel、编码器接口 |
| `cpp/src/fastlanes/` | RAW32 / NATIVE64 编码器和生成的 CPU kernel |
| `cpp/src/io/parquet/fastlanes_*`、`page_fastlanes_decode.cu` | Parquet 里的编码 staging 和解码 kernel |
| `writer_impl.cu`、`page_enc.cu`、`page_hdr.cu`、`reader_impl.cpp`、`parquet_gpu.hpp` | Parquet 读写路径上的挂接点 |

代码结构和读写流程见集成说明第 2 节。

### 2.2 测试

`cpp/tests/io/parquet_fastlanes_test.cpp`、`parquet_fastlanes_native64_generated_test.cu`、
`parquet_fastlanes_native64_bw37_test.cu`，对应 ctest target `PARQUET_FASTLANES_TEST`（50 个测试）。

### 2.3 示例程序和脚本（`cpp/examples/parquet_io/`）

| 文件 | 作用 |
| --- | --- |
| `fastlanes_encoding_bench.cpp` | benchmark 的核心程序：`sweep` 模式对一列试遍各种编码和 codec，`read` 模式给整个文件的读取计时 |
| `parquet_io_chunk.cpp` | 按 row group 分批重写整个 Parquet 文件，可以按列指定编码，写完默认逐个 row group 校验 |
| `fastlane_one_page_sanity_test.cpp` 等 5 个 `fastlane_*` 程序 | FastLanes 冒烟测试（只覆盖 32 位路径，见集成说明第 4 节） |
| `parquet_io_chunking_sanity.cpp` | 分块重写的冒烟测试 |
| `tools/bench/run_tpch_sf100_fastlanes_bench.py` | benchmark 驱动脚本，调用上面两个程序跑完整个测试 |
| `tools/bench/fastlanes_bench_tables.py` | 从一次 benchmark 的结果生成报告里的表格和 Canvas 数据 |
| `tools/search/`、`tools/roundtrip/`、`tools/tests/` | 之前的工具：贪心编码搜索、roundtrip 校验、page 统计等（见 `cpp/examples/parquet_io/README.md`） |

### 2.4 文档（本目录）

| 文件 | 内容 |
| --- | --- |
| `README.md` | 本文 |
| `FASTLANES_INTEGRATION_REPORT_2026-09-27.md` | 集成说明（取代 `FASTLANES_INTEGRATION_REPORT_2026-03-28.md`） |
| `FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md` | TPC-H SF100 测试报告 |
| `canvas/fastlanes-cudf-integration.canvas.tsx`、`canvas/fastlanes-tpch-sf100-benchmark.canvas.tsx` | 两份报告对应的 Cursor Canvas（可交互的页面） |
| `INT64_SNAPPY_*`、`ENCODING_SEARCH_METHOD.md`、`INT64_PAGE_STATS_WORKFLOW.md`、`fastlanes_native64_r4_generation_strategy.md` | 2026 年 3-4 月的旧文档 |

### 2.5 不在 git 里的东西

`cpp/examples/parquet_io/artifacts/` 在 `.gitignore` 里，TPC-H 数据和 benchmark 原始结果都放在这里，换机器需要
重新生成（第 5 节）。编译目录 `cpp/build/` 和 `cpp/examples/*/build/` 也不在 git 里。

## 3. 需要什么

| 项 | 要求 |
| --- | --- |
| GPU | NVIDIA GPU。这次用的是 RTX 5090 Laptop GPU（24 GB，sm_120）。显存更小的卡没有试过；整表重写的显存占用可以用 `--batch-size`、`--rgs-per-read` 调小 |
| 驱动 | 支持 CUDA 13.3（这台机器是 595.84） |
| 容器 | Docker 和 NVIDIA Container Toolkit，用来跑 RAPIDS devcontainer |
| CPU、内存 | 这台机器是 24 核、62 GB 内存。RAW32 编码在 CPU 上做，写入速度和 CPU 有关 |
| 磁盘 | 见下表，建议留 100 GB 以上 |

| 占用 | 大小 |
| --- | --- |
| 8 张表的 Parquet（`lineitem` 28.6 GB，其余 7 张共 13.4 GB） | 约 42 GB |
| 生成数据用的 DuckDB 数据库 `tpch_sf100.duckdb`（导出后可以删） | 27.6 GB |
| 整表重写的临时输出（一次一个文件，用完就删，最大的约 18 GB） | 约 20 GB |
| libcudf 编译目录 | 约 6 GB |

## 4. 环境、编译和测试

### 4.1 进入 devcontainer

另一台机器需要先拿到这个分支。它还没有 push（见第 10 节），push 之后在新机器上 clone 并切到
`fastlane-working`。然后用 `.devcontainer/cuda13.3-conda/devcontainer.json` 启动容器：

- VS Code 或 Cursor：命令面板里选 "Dev Containers: Reopen in Container"，配置选 `cuda13.3-conda`；
- 命令行：`devcontainer up --workspace-folder <仓库目录> --config .devcontainer/cuda13.3-conda/devcontainer.json`。

容器里仓库在 `~/cudf`（`/home/coder/cudf`）。如果仓库是 `git worktree` 建出来的，容器里的 git 找不到主仓库的
`.git`，需要把主仓库的 `.git` 也挂进去；用普通 clone 就没有这个问题。

编译和测试用 RAPIDS 的 wrapper（`configure-cudf-cpp`、`build-cudf-cpp`、`test-cudf-cpp`），它们会自己激活
conda 环境。直接运行 Python 脚本之前要先激活：

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate rapids
```

### 4.2 编译 libcudf 和测试

```bash
configure-cudf-cpp -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=NATIVE
build-cudf-cpp -j16 -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=NATIVE
```

`NATIVE` 只为本机的 GPU 编译，最快；要在别的 GPU 上用，得在那台机器上重新编译。并行度用 `-j16` 这类固定值：
这台机器上 `-j0` 配合 sccache-dist 展开成了几千个任务，编译卡住了。

### 4.3 运行测试

```bash
test-cudf-cpp -R '^(PARQUET_TEST|PARQUET_FASTLANES_TEST|HYBRID_SCAN_TEST|PARQUET_DELETION_VECTORS_TEST|STREAM_IO_PARQUET_TEST|STREAM_IO_HYBRID_SCAN_TEST)$'
```

期望结果：7 个 ctest 条目全部通过（其中一个是 `generate_resource_spec`），共 713 个 gtest，`PARQUET_TEST` 里
有 1 个上游自带的 skip（`ZeroColumnsPreservesRowCount`）。

### 4.4 编译示例程序

在仓库根目录执行：

```bash
LIB_BUILD_DIR=$(realpath cpp/build/latest) PARALLEL_LEVEL=16 cpp/examples/build.sh
```

脚本会编译 `cpp/examples/` 下的所有示例，`parquet_io` 的程序在 `cpp/examples/parquet_io/build/`。几点注意：

- `LIB_BUILD_DIR` 要指向实际的 libcudf 编译目录。`build.sh` 默认用 `cpp/build`，如果那一层还留着旧版本的
  `cudf-config.cmake`（这台机器上就有 26.08 的旧文件），示例会找错 libcudf。
- 程序的 RPATH 写死了编译时的 libcudf 和 conda 路径，换机器必须重新编译，不能直接拷贝二进制。
- sccache 起不来时编译会失败，见 4.6 节。

### 4.5 冒烟测试

在 `cpp/examples/parquet_io/build/` 下运行：

| 程序 | 期望的最后输出 |
| --- | --- |
| `./fastlane_one_page_sanity_test` | `FINAL RESULT: ALL TESTS PASSED` |
| `./fastlane_multi_page_sanity_test` | `FINAL RESULT: ALL TESTS PASSED` |
| `./fastlane_multi_column_test` | `FINAL RESULT: ALL TESTS PASSED` |
| `./fastlane_encode_verify_test` | `=== All Tests Complete ===` |
| `./fastlane_int64_cpu_roundtrip` | `Final result: ALL CASES PASSED` |
| `./parquet_io_chunking_sanity <输入.parquet> <输出.parquet>` | `Pipeline complete. ...` |

### 4.6 sccache 起不来怎么办

容器里编译时，`CMAKE_C_COMPILER_LAUNCHER` 等变量指向 sccache，sccache 用的远程缓存需要凭证。凭证过期或者没有配置时，
编译会报 `sccache: error: Server startup failed ... loading credential to sign http request` 这类错误。

- devcontainer 自带 `devcontainer-utils-creds-s3-init`、`devcontainer-utils-start-sccache`、
  `devcontainer-utils-stop-sccache` 等脚本，可以先试着重新初始化凭证（可能需要登录 GitHub）。libcudf 本身的编译也
  靠 sccache 加速，不用 sccache 编译整个 libcudf 会慢很多（这里没有试过）。
- 示例程序可以直接不用 sccache 编译，这台机器上试过：

  ```bash
  rm -rf cpp/examples/parquet_io/build
  env BASH_ENV= CMAKE_C_COMPILER_LAUNCHER= CMAKE_CXX_COMPILER_LAUNCHER= CMAKE_CUDA_COMPILER_LAUNCHER= \
    LIB_BUILD_DIR=$(realpath cpp/build/latest) PARALLEL_LEVEL=16 cpp/examples/build.sh
  ```

  `BASH_ENV` 要一起清掉：容器里的 `/etc/bash.bash_env` 会在脚本启动时把 launcher 重新设回 sccache。编译目录里
  已经记住了 launcher，所以要先删掉 `build/`。示例程序不多，不用 sccache 也只要半分钟左右。

## 5. 准备 TPC-H SF100 数据

benchmark 默认从 `cpp/examples/parquet_io/artifacts/tpch100/sf100/<表名>.parquet` 读 8 张表。这台机器上的数据是用
DuckDB 的 tpch 扩展生成的（本机 DuckDB 为 v1.1.3），再用 `COPY` 导出成 SNAPPY 压缩的 Parquet，row group 是 DuckDB
默认的 122,880 行：

```bash
cd cpp/examples/parquet_io/artifacts/tpch100
duckdb tpch_sf100.duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf = 100);"
mkdir -p sf100
for t in lineitem orders partsupp part customer supplier nation region; do
  duckdb -readonly tpch_sf100.duckdb -c "COPY $t TO 'sf100/$t.parquet' (FORMAT parquet, COMPRESSION snappy)"
done
```

- `INSTALL tpch` 要联网下载扩展。
- SF100 的生成比较耗时间和内存；内存不够时可以用 `dbgen` 的 `children`、`step` 参数分批生成，具体见 DuckDB
  tpch 扩展的文档。
- 这台机器上 `lineitem` 是之前单独导出的 `../lineitem_sf100.parquet`，`sf100/lineitem.parquet` 是指向它的软链接，
  效果一样。

导出后在 rapids 环境里检查一下（在 `artifacts/tpch100` 目录下）：

```bash
python - <<'EOF'
import collections, pyarrow.parquet as pq
types = collections.Counter()
for t in ["lineitem", "orders", "partsupp", "part", "customer", "supplier", "nation", "region"]:
    f = pq.ParquetFile(f"sf100/{t}.parquet")
    types.update(str(x.type) for x in f.schema_arrow)
    print(t, f.metadata.num_rows, f.metadata.num_row_groups)
print(dict(types))
EOF
```

应该得到：

| 表 | 行数 | row group 数 |
| --- | ---: | ---: |
| lineitem | 600,037,902 | 4,884 |
| orders | 150,000,000 | 1,221 |
| partsupp | 80,000,000 | 652 |
| part | 20,000,000 | 163 |
| customer | 15,000,000 | 123 |
| supplier | 1,000,000 | 9 |
| nation | 25 | 1 |
| region | 5 | 1 |

类型合计是 `int64` 12 列、`int32` 7 列、`date32[day]` 4 列、`string` 29 列、`decimal128(15, 2)` 9 列。如果用别的
工具生成数据（比如把 key 写成 int32），FastLanes 能编码的列会不一样，结果就不能直接和报告比。

## 6. 运行 benchmark

以下命令都在容器里、激活 rapids 环境之后，在 `cpp/examples/parquet_io` 目录下执行。

先用三张小表确认流程能跑通（这台机器上不到 1 分钟）：

```bash
python tools/bench/run_tpch_sf100_fastlanes_bench.py \
  --run-dir artifacts/fastlanes_smoke \
  --tables nation,region,supplier \
  --steps sweep,sweep_default,tables,summarize
```

完整运行，得到报告里的全部数据：

```bash
python tools/bench/run_tpch_sf100_fastlanes_bench.py \
  --run-dir artifacts/fastlanes_bench_sf100_$(date +%Y%m%d) \
  --steps sweep,sweep_default,tables,ablation,summarize \
  > artifacts/driver_$(date +%Y%m%d).log 2>&1
```

完整运行要几十分钟，最好放到后台（`nohup ... &`），或者在宿主机上用 `docker exec -d <容器名> bash -lc '...'`
启动，这样终端断开也不受影响。

### 6.1 各步骤

| 步骤 | 做什么 | 输出（在 `<run-dir>/03_raw/` 下） | 这台机器上的耗时 |
| --- | --- | --- | --- |
| `sweep` | 23 个适用列 × 5 种编码 × 3 种 codec，每种预热 1 次、计时 3 次，page 正好 20,480 行 | `sweep.csv` | 约 15 分钟 |
| `sweep_default` | 同样的测试，但用 cuDF 默认的 5,000 行 page fragment（page 为 20,000 行），给报告 3.6 节的布局对比用；默认不跑 | `sweep_default_fragments.csv` | 约 15 分钟 |
| `tables` | 8 张表 × SNAPPY/ZSTD × 4 种编码方案，用 `parquet_io_chunk` 重写，用到 FastLanes 的方案逐个 row group 校验，再计时读取 | `tables.csv`、`reads.csv` | 约 15 分钟 |
| `ablation` | `l_partkey`、`l_shipdate` 在三种 page 布局下对比 FastLanes 和最优标准编码（SNAPPY） | `ablation.csv` | 约 3 分钟 |
| `summarize` | 汇总成 JSON 和 markdown | `../02_machine/summary.json`、`../01_human/summary.md` | 不到 1 秒 |

`tables` 要用 `sweep` 的结果来决定每列的编码，所以要先跑 `sweep`。四种方案：`cudf-default`（全部用 cuDF 默认），
`best-standard`（每个适用列用最小的标准编码），`fastlanes-all`（适用列全用 FastLanes），`best-with-fastlanes`
（每个适用列用最小的编码，FastLanes 也参与比较）。FastLanes 不支持的列在所有方案里都用 cuDF 默认编码。

### 6.2 选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-dir` | `artifacts/tpch100/sf100` | 8 张表的目录 |
| `--run-dir` | `artifacts/fastlanes_bench_sf100_<今天>` | 结果目录 |
| `--scratch-dir` | `<run-dir>/03_raw/cases` | 整表重写的临时输出，每个文件读完就删 |
| `--bench-bin` | `build/fastlanes_encoding_bench` | |
| `--chunk-bin` | `build/parquet_io_chunk` | |
| `--tables` | 全部 8 张表 | 逗号分隔 |
| `--steps` | `sweep,tables,ablation,summarize` | 逗号分隔，可选 `sweep`、`sweep_default`、`tables`、`ablation`、`summarize` |
| `--warmup` / `--repeats` | 1 / 3 | 预热和计时的次数 |
| `--batch-size` / `--rgs-per-read` | 64 / 64 | 传给 `parquet_io_chunk`；显存不够时调小 |
| `--hang-timeout` | 600 | 单次读或写超过这么多秒就记为卡住 |
| `--keep-outputs` | 关 | 保留整表重写的输出文件 |

脚本启动时会检查步骤名、数据文件和两个程序是否存在，缺了会直接报错退出。

### 6.3 中断和续跑

每个步骤都只往 CSV 里追加结果，已经记录的组合会跳过，所以中断以后用同样的命令再跑一次就能接着做。
`fastlanes_encoding_bench` 里有看门狗：单次读或写超过 `--hang-timeout` 秒，会记一条错误结果并以返回码 3 退出，
驱动脚本接着跑剩下的组合。`parquet_io_chunk` 没有看门狗，驱动脚本给它的超时是 300 秒加上每 GB 输入 120 秒。

### 6.4 输出目录

```text
<run-dir>/
  01_human/summary.md           每列最优编码和整表合计（英文）
  02_machine/environment.json   GPU、CUDA、commit、布局参数
  02_machine/summary.json       所有结果汇总
  03_raw/sweep.csv              逐列测试（对齐布局）
  03_raw/sweep_default_fragments.csv   逐列测试（cuDF 默认布局，只有跑了 sweep_default 才有）
  03_raw/tables.csv             整表重写：大小、重写时间、是否校验通过
  03_raw/reads.csv              整表读取计时
  03_raw/ablation.csv           三种 page 布局的对比
  03_raw/logs/                  每次调用的日志
  03_raw/cases/                 整表重写的临时文件（默认用完即删）
```

`sweep.csv` 里主要的列：

| 列 | 含义 |
| --- | --- |
| `requested` / `resolved` | 请求的编码 / 实际用的编码（`FASTLANES` 对 INT64 解析为 `FASTLANES_DELTA_BINARY`，对 INT32 和日期解析为 `FASTLANE_BITPACK_RAW`） |
| `bytes`、`bits_per_value` | 这一列编码后的文件大小，以及每个值平均占几个 bit |
| `write_ms_*`、`read_ms_*`、`write_gbps`、`read_gbps` | 写入、读取时间（中位数、最小、最大）和吞吐（按内存中的列数据算） |
| `page_encodings`、`unexpected_pages` | footer 里各编码的 page 数，以及和请求不一致的 page 数（不为 0 说明 cuDF 悄悄换了编码，比如 DICTIONARY 被换成 DELTA_BINARY_PACKED） |
| `validated`、`error` | 解码结果是否和源数据一致；出错或卡住时的错误信息 |

`tables.csv` 里 `validated` 为 1 表示校验通过，0 表示失败，-1 表示没有用 FastLanes、跳过了校验；`same_as` 不为空
表示这个方案和另一个方案的编码完全相同，直接沿用了那个方案的结果。

## 7. 生成报告表格和 Canvas 数据

```bash
python tools/bench/fastlanes_bench_tables.py artifacts/fastlanes_bench_sf100_<日期> > tables.md
python tools/bench/fastlanes_bench_tables.py artifacts/fastlanes_bench_sf100_<日期> --canvas-ts > canvas_data.ts
```

- 这个脚本只用 Python 标准库，需要 Python 3.10 以上，不需要 GPU，宿主机上也能跑。
- markdown 输出和报告的各节一一对应。报告里只有一张表是手工改过的："标准编码始终更小的列"按类型分了组，加了
  "特点"列，小表的数值取了整。用 2026-09-27 的结果重新生成，其余表格和报告完全一致。
- `--canvas-ts` 输出 `COLUMNS`、`BITS`、`SPEED`、`AGGREGATE`、`PLAN_TOTALS`、`TABLE_ROWS`、`ABLATION` 七个常量，
  替换 `canvas/fastlanes-tpch-sf100-benchmark.canvas.tsx` 里的同名常量即可。
- 结果不完整时也能用：只跑了几张表，或者有组合出错、卡住，脚本会在输出开头列出没有计入的列和表。
- 查看 Canvas：Cursor 只显示 `~/.cursor/projects/<工作区>/canvases/` 下的 `.canvas.tsx` 文件，把 `canvas/` 里的
  文件复制过去就能在 Cursor 里打开。

结果的可重复性：在这台机器上用 nation、region、supplier 重跑了一遍，和原来的结果相比，只有 DICTIONARY 加 ZSTD
的几个结果差了几十个字节，其余大小全部逐字节相同。差异应该来自 DICTIONARY 的字典顺序：它由 GPU 上并行插入的哈希表
决定，每次可能不同。时间在这台笔记本上每次有 2% 左右的波动。

## 8. 单独使用这两个程序

### 8.1 `fastlanes_encoding_bench`

```bash
# 对 orders.o_custkey 试遍 5 种编码，只用 SNAPPY
./build/fastlanes_encoding_bench sweep \
  --input=artifacts/tpch100/sf100/orders.parquet --table=orders --columns=o_custkey \
  --compressions=SNAPPY --output=/tmp/o_custkey.csv

# 给一个或多个文件的读取计时
./build/fastlanes_encoding_bench read --input=a.parquet,b.parquet --label=test --output=/tmp/reads.csv
```

常用选项：`--encodings`（默认 5 种都测，`FASTLANES` 会按列的类型自动选模式）、`--compressions`、`--combos`
（只跑指定的 `编码/codec` 组合）、`--row-group-rows`（122,880）、`--page-rows`（20,480）、`--fragment-rows`
（默认等于 `--page-rows`，0 表示 cuDF 默认的 fragment）、`--warmup`、`--repeats`、`--no-validate`、
`--hang-timeout-s`。完整说明见 `--help`。

### 8.2 `parquet_io_chunk`

```bash
# 用 FastLanes 重写 nation：两个 key 列用 RAW32，其余列保持 cuDF 默认
./build/parquet_io_chunk artifacts/tpch100/sf100/nation.parquet /tmp/nation_fl.parquet \
  "n_nationkey:FASTLANE_BITPACK_RAW,n_name:DEFAULT,n_regionkey:FASTLANE_BITPACK_RAW,n_comment:DEFAULT" SNAPPY \
  --enable-v2-headers --max-page-rows=20480 --page-fragment-rows=20480
```

- 第三个参数可以是一个编码（所有列都用它），也可以是 `列名:编码` 的列表。用列表时必须列出每一列，不想改的列写
  `DEFAULT`，漏了会报 `No encoding specified for column ...`。
- FastLanes 的两个编码：`FASTLANE_BITPACK_RAW` 用于 INT32 和日期列，`FASTLANES_DELTA_BINARY` 用于 INT64 列。
  不适用的列会打印警告并回退到默认编码。
- 想让 page 正好是某个行数，`--max-page-rows` 和 `--page-fragment-rows` 要设成同一个值，最好是 1024 的倍数
  （FastLanes 会把 page 补齐到 1024 个值的整数倍）。只设 `--max-page-rows` 时，page 由 5,000 行的 fragment 拼成。
- 写完默认会读回来逐个 row group 和源文件比对，`--skip-validation` 可以关掉。
- 用 FastLanes 写出来的文件只有这个 cuDF 分支能读，其他 Parquet reader（比如 pyarrow）不认识这些编码。

## 9. 已知问题和注意事项

- 在 RTX 5090 Laptop GPU（sm_120）上，用 chunked reader 并设置 `pass_read_limit` 读 ZSTD 文件时，cuDF 调用的
  `nvcompBatchedZstdDecompressGetTempSizeSync` 对某些输入一直不返回。benchmark 已经改成按 row group 分批用
  `read_parquet` 读，避开了这个问题。别的工具读 ZSTD 文件卡住时，先考虑是不是这个原因。这是 cuDF/nvCOMP 上游的
  问题，和 FastLanes 无关。
- FastLanes 目前只支持扁平、不含 null 的列。写入路径不检查 null，有 null 的列不要用 FastLanes。
- INT64 的 FastLanes Parquet 读写路径没有 gtest 覆盖，只有 benchmark 做了端到端校验（见集成说明第 4 节）。
- 提交前看一眼 `git status`，不要把 `artifacts/` 以外的大文件加进去。这次就出现过 "Stage All" 把 5 GB 的临时文件
  放进暂存区的情况。

## 10. 这台机器的状态（2026-09-28）

这一节只描述写这份文档时这台机器上的情况，换机器后不适用。

| 项 | 状态 |
| --- | --- |
| worktree | `/home/qic/cudf-fastlane-isolated`（主仓库在 `/home/qic/cudf`） |
| devcontainer | `qic-rapids-cudf-fastlane-isolated-26.10-cuda13.3-conda`，还在运行，可以 `docker stop` |
| TPC-H 数据 | `artifacts/tpch100/`：SF1、SF10、SF100 的 DuckDB 数据库和 `lineitem` Parquet，`sf100/` 下是 8 张表 |
| benchmark 结果 | `artifacts/fastlanes_bench_sf100_20260927/`（报告用的就是这次的数据） |
| 远程分支 | 还没有 push。`private/fastlane-working` 上有 27 个旧提交是 rebase 之前的版本，本地历史里都有对应的提交，所以 `git push --force-with-lease private fastlane-working` 不会丢东西 |
| 本地改动 | `.devcontainer/cuda13.3-conda/devcontainer.json` 里有本地的挂载配置，设了 `skip-worktree` 所以不会被提交（`git update-index --no-skip-worktree <文件>` 可以取消）；`Fl_Size_ana.md` 是之前的笔记，没有纳入 git |
| 编译目录 | `cpp/build/` 最外层还有 26.08 的旧 `cudf-config.cmake`，编译示例时要用 `LIB_BUILD_DIR` 指定 `cpp/build/latest` |
