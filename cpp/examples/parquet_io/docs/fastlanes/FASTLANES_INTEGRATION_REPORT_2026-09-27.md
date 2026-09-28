# FastLanes 在 cuDF 中的集成说明（2026-09-27）

本文说明 `fastlane-working` 分支上的 FastLanes bit-packing 编码是怎样接入 cuDF Parquet 读写路径的，
以及把分支合到 cuDF 26.10 release 分支时改了什么。TPC-H SF100 的测试结果见
[FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md](FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md)。
本文取代 [FASTLANES_INTEGRATION_REPORT_2026-03-28.md](FASTLANES_INTEGRATION_REPORT_2026-03-28.md)，
旧文档描述的是 INT64 早期的 split-32 布局（现在对应已废弃的 `FASTLANE_BITPACK_SPLIT64`）。
对应的 Cursor Canvas 源文件在 `canvas/fastlanes-cudf-integration.canvas.tsx`。在另一台机器上编译、测试和复现
benchmark 的步骤见同目录的 [README.md](README.md)。

## 1. 概要

- FastLanes 加了什么：三种 Parquet page 编码（文件中的编码 ID 为 10、11、12）。列通过常规的按列编码设置
  启用：INT64 列用 `FASTLANES_DELTA_BINARY`（在 GPU 上打包），INT32 和日期列用 `FASTLANE_BITPACK_RAW`
  （在 CPU 上打包）。`FASTLANE_BITPACK_SPLIT64` 已废弃，只保留解码。读取不需要额外选项，但其他 Parquet
  reader 读不了这些 page。
- 代码在哪：全部在 Parquet 模块里。`writer_impl.cu` 做类型检查，`page_enc.cu` 选编码、算 page 大小，
  `EncodePages` 里加了一个 staging 步骤来生成每个 FastLanes page 的数据；读取时由 `page_hdr.cu` 选 kernel，
  `reader_impl.cpp` 启动每个 page 一个 warp 的解码 kernel。压缩、统计信息、page header 和文件组装都还是
  cuDF 原来的代码。
- 26.10 合并：`fastlane-working` 现在基于 `release/26.10`，依次是一个解决了 4 处冲突的合并提交
  （`c8533b8d3d`），一个 26.10 API 适配提交（`9ed93250d2`：`cuda::stream_ref`、组合式 decode state、
  `cuda::` functor），之后是 benchmark 工具（`e0f905a0ae`）和这两份报告。合并前的状态保存在
  `backup/fastlane-working-pre-26.10`。
- 验证：在 26.10 devcontainer 和 RTX 5090 Laptop GPU 上，6 个 Parquet 测试 target 共 713 个测试全部
  通过（其中 1 个是上游自带的 skip），包括 50 个 FastLanes 测试，FastLanes 示例程序也都通过。TPC-H SF100
  benchmark 写出的所有 FastLanes 文件，解码后都和源数据一致。INT64 路径目前只有这个 benchmark 做过端到端
  检查，没有 gtest 用 FastLanes 写 INT64 列（见第 4 节）。
- 主要限制：只支持扁平、不含 null 的列；写入慢（约 1 GB/s）；编码 ID 不在 Parquet 标准里；FOR 编码在
  有序 key 上很吃亏。只在合适的列上用 FastLanes 时，TPC-H SF100 文件在 SNAPPY 下小 1.1%，ZSTD 下小 0.6%。

## 2. FastLanes 怎样接入 cuDF

FastLanes 以三种新的 Parquet page 编码的形式加入，Parquet 模块以外没有改动：列通过现有的按列编码设置启用，
writer 输出的是普通 Parquet page，只是 value 部分放的是 FastLanes 数据；reader 识别新的编码 ID，用专门的
kernel 解码这些 page。

```mermaid
flowchart LR
  subgraph writePath [写入路径]
    SetEnc["column_in_metadata.set_encoding"] --> SchemaCheck["writer_impl.cu：schema 校验"]
    SchemaCheck --> MaskSel["page_enc.cu：data_encoding_for_col"]
    MaskSel --> Reserve["InitEncoderPages：预留 page 空间"]
    Reserve --> EncDispatch["EncodePages：每种编码一个 stream"]
    EncDispatch --> Stage["fastlanes_encode_stage：收集、编码、校验"]
    Stage --> CopyK["gpuEncodeCpuPages + finish_page_encode"]
  end
  subgraph readPath [读取路径]
    PageHdr["page_hdr.cu：kernel_mask_for_page"] --> Dispatch["reader_impl.cpp：decode_page_data"]
    Dispatch --> DecK["page_fastlanes_decode.cu：每个 page 一个 warp"]
  end
  CopyK --> PqPage["Parquet page：128 字节 FastLanes header + 1024 值的 vector"]
  PqPage --> PageHdr
```

### 2.1 对外接口

| 项 | 位置 | 取值 |
| --- | --- | --- |
| 编码请求 | `cpp/include/cudf/io/types.hpp` 中的 `cudf::io::column_encoding` | `FASTLANE_BITPACK_RAW`、`FASTLANES_DELTA_BINARY`、`FASTLANE_BITPACK_SPLIT64`（已废弃） |
| 文件中的编码 ID | `cpp/include/cudf/io/parquet_schema.hpp` 中的 `cudf::io::parquet::Encoding` | 10、11、12（`NUM_ENCODINGS` 改为 13） |

启用方式和其他编码一样：

```cpp
cudf::io::table_input_metadata metadata(table);
metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_DELTA_BINARY);  // INT64
metadata.column_metadata[1].set_encoding(cudf::io::column_encoding::FASTLANE_BITPACK_RAW);    // INT32 / date
```

读取不需要任何选项，`cudf::io::read_parquet` 能识别编码 10-12。其他 Parquet reader 不认识这些 ID（pyarrow
显示为 `UNKNOWN`），所以 FastLanes 文件只能用 cuDF 读。

适用范围在 `writer_impl.cu` 构建 schema 时检查一次，在设备端的 `data_encoding_for_col` 里再检查一次：

| 编码 | 物理类型 | 逻辑类型 | 在哪编码 |
| --- | --- | --- | --- |
| `FASTLANE_BITPACK_RAW` | INT32 | int8/16/32、uint8/16/32、date32、decimal32、duration | CPU |
| `FASTLANES_DELTA_BINARY` | INT64 | int64、uint64（不能带 decimal、timestamp 或 time 注解） | GPU |
| `FASTLANE_BITPACK_SPLIT64` | INT64 | 写入时直接报错（`CUDF_FAIL`）；仍可解码 | - |

不符合条件的请求会打印警告并回退到默认编码；嵌套（list）列在设备端回退（`RAW` 回退到 PLAIN，
`DELTA_BINARY` 回退到 DELTA_BINARY_PACKED）。

### 2.2 写入路径

1. **Schema 校验**（`writer_impl.cu` 中的 `construct_parquet_schema_tree`）：按上表检查每个 FastLanes
   请求的物理类型和逻辑类型。
2. **选 kernel mask**（`page_enc.cu` 中的 `data_encoding_for_col`，设备端代码）：运行时确认列是扁平的，
   再把请求映射到 `encode_kernel_mask::FASTLANE_BITPACK_RAW`（bit 6）或 `FASTLANES_DELTA_BINARY`（bit 7）。
3. **算 page 大小**（`InitEncoderPages`）：FastLanes 会把每个 page 补齐到整数个 1024 值的 vector，所以
   page buffer 按最坏情况预留 128 字节 header + `ceil(n / 1024) * 1024` 个值，每个值 32 bit（RAW）或
   64 bit（NATIVE64），再加上 payload 前面的 level 字节。
4. **分发**（`EncodePages`）：和标准编码一样，每个用到的编码 bit 有自己 fork 出来的 stream。FastLanes 的
   bit 调用 `fastlanes_encode_stage::run_fastlanes_raw32_encode` 或 `run_fastlanes_native64_encode`
   （`fastlanes_page_encoder_*.cuh`）。这些头文件直接 include 进 `page_enc.cu`，这样才能启动里面的模板
   kernel。
5. **Staging**（每类 FastLanes 编码分别做一遍）：
   - 把 `EncPage` 数组拷到 host，挑出属于这一类的 page；
   - `gpuGatherSinglePageTyped` 把每个 page 的值收集到连续的 buffer，类型转换和 PLAIN 路径相同（int8/16
     提升、timestamp 缩放）；
   - 编码：RAW32 把每个 page 下载到 host，减去 page 最小值，在 CPU 上用生成的 FastLanes `pack` kernel
     打包 1024 值的 vector，再把结果上传。NATIVE64 在 GPU 上做最小值归约和打包，每种 bit 宽度有单独的
     kernel（`native64_cuda_kernels.inl`）；
   - 一次性读回所有 page header（每类只同步一次），检查 header，并确认数据没有超出预留空间。
6. **组装 page**：`gpuEncodePageLevels` 写 definition/repetition level，`gpuEncodeCpuPages` 把 FastLanes
   数据拷进 page，把 `page.encoding` 设成 10 或 11，然后调用原有的 `finish_page_encode` 写 page header、
   统计信息和压缩输入。压缩、column chunk 元数据（包括 `encoding_stats`）和文件组装都是 cuDF 原有代码。

### 2.3 文件中的 page 格式

FastLanes page 就是普通的 Parquet data page（V1 或 V2 header，标准 RLE level）。它的 value 部分由一个
128 字节的 FastLanes header 和后面打包好的数据组成：

| 偏移 | 大小 | 字段 | 含义 |
| --- | --- | --- | --- |
| 0 | 1 | `bitwidth_lo` | 每个值的 bit 数（RAW32、NATIVE64），或低位部分的 bit 数（SPLIT64） |
| 1 | 1 | `bitwidth_hi` | 高位部分的 bit 数（仅 SPLIT64） |
| 2 | 1 | `pre_delta` | 布局策略标志，按编码分别校验 |
| 3 | 1 | `reserved_flags` | 必须为 0 |
| 4 | 4 | `original_count` | page 中的值个数 |
| 8 | 4 | `padded_count` | `original_count` 向上取整到 1024 的倍数 |
| 12 | 4 | `body_size` | payload 字节数 |
| 16 | 4 | `min_value_lo` | page 最小值（低 32 bit） |
| 20 | 4 | `min_value_hi` | page 最小值（高 32 bit） |
| 24 | 104 | padding | 全 0 |
| 128 | `body_size` | payload | `padded_count / 1024` 个 vector，每个 `1024 * bitwidth / 8` 字节 |

目前启用的两种编码都是 FOR（frame-of-reference）编码：每个值存成 `value - page_minimum`，每个 page
选一个固定的 bit 宽度，按 FastLanes 交错的 1024 值 vector 顺序排列。`FASTLANES_DELTA_BINARY` 名字里虽然有
delta，但并不对相邻值做差分（它的 `pre_delta` 策略是关的，kernel 打包的是 `value - base`）。这对有序 key
影响很大，详见 benchmark 报告。

### 2.4 读取路径

1. **选 kernel**（`page_hdr.cu` 中的 `kernel_mask_for_page`）：编码 10/11/12 对应 `decode_kernel_mask`
   的 bit 27/28/29；`parquet_gpu.hpp` 中的 `is_supported_encoding` 接受这三个编码。
2. **分发**（`reader_impl.cpp` 中的 `decode_page_data`）：统计要用的 kernel，每个 kernel fork 一个
   stream，启动 `decode_fastlanes_raw32` / `_split64` / `_native64`。
3. **解码**（`page_fastlanes_decode.cu`）：每个 page 一个 warp（32 个线程）。
   - 被裁剪（pruned）的 page 直接返回，它们的 null mask 和 offset 由 host 端修正。
   - `setup_local_page_info` 做 cuDF 通用的 page 初始化（level 解码、行范围、输出指针），结果放在
     `full_page_decode_state` 里。
   - 检查 page：列必须是扁平的，page 里不能有 null，输出宽度要受支持，header 要和编码一致。
   - 对每个 1024 值的 vector：先把打包的 word 经 shared memory 中转（payload 可能不对齐），再解包
     （32 位用生成的 `unpack_device`，64 位用按 bit 宽度区分的 native64 解码），加回 page 最小值，然后按
     page 的行偏移直接写进输出列，同时处理 `skip_rows` / `num_rows`。

### 2.5 代码分布

| 层 | 文件 | 行数 | 作用 |
| --- | --- | ---: | --- |
| 公共枚举 | `cpp/include/cudf/io/types.hpp`、`parquet_schema.hpp` | ~20 | 编码请求和文件中的 ID |
| FastLanes 核心头文件 | `cpp/include/cudf/fastlanes/` | 5,738 | page header、大小计算、生成的设备端 unpack、NATIVE64 kernel、编码器 API |
| 生成的 CPU kernel | `cpp/src/fastlanes/{pack,unpack,ffor,unffor,transpose,unrsum}.cpp` 及封装 | 109,723 | FastLanes 参考 kernel（生成代码） |
| 编码器 | `cpp/src/fastlanes/encode_*.cu`、`encode_common.hpp`、`native64_host.cu` | 759 | RAW32（CPU）、NATIVE64（GPU）、SPLIT64（旧版） |
| Parquet 衔接代码 | `cpp/src/io/parquet/fastlanes_*`、`page_fastlanes_decode.cu` | 1,287 | 适用性判断、编码 staging、解码 kernel |
| Parquet 挂接点 | `writer_impl.cu`、`page_enc.cu`、`page_hdr.cu`、`reader_impl.cpp`、`parquet_gpu.hpp`、`reader_impl_preprocess_utils.cu` | 约 500 行改动 | 校验、大小计算、分发 |
| 测试 | `cpp/tests/io/parquet_fastlanes_*` | 2,865 | `PARQUET_FASTLANES_TEST`（50 个测试） |
| 工具 | `cpp/examples/parquet_io/` | - | `parquet_io_chunk`、`fastlanes_encoding_bench`、sanity 程序、搜索和 roundtrip 脚本 |

构建配置：`cpp/CMakeLists.txt` 把 13 个 FastLanes 源文件和 `page_fastlanes_decode.cu` 加进 `cudf` 库；
`cpp/tests/CMakeLists.txt` 定义 `PARQUET_FASTLANES_TEST`。

## 3. 合并 26.10 时改了什么

### 3.1 Git 操作

- worktree `/home/qic/cudf-fastlane-isolated` 上是 `fastlane-working` 分支（它包含 `fastlane-isolated`
  的全部内容：FastLanes 文件相同，另外多四个较新的提交）。这个 worktree 的管理目录之前已经从主仓库里被
  prune 掉，所以用 `git worktree add --no-checkout` + `git worktree repair` + `git reset` 重新登记
  （没有改动任何文件）。
- 合并前状态的备份：分支 `backup/fastlane-working-pre-26.10`，指向 `0fae90d66c`。
- `git merge --no-ff upstream/release/26.10`（上游 `eaa1e31058`，比分支的 merge base `3cfe15b68d` 多
  945 个提交），提交为 `c8533b8d3d`。26.10 的 API 迁移单独放在后面一个提交里，这样合并提交里只有冲突处理。

### 3.2 冲突处理

| 文件 | 冲突 | 处理方式 |
| --- | --- | --- |
| `cpp/CMakeLists.txt` | 上游在分支添加 `src/io/parquet/fastlanes` 的位置加了 `cudf_cuda_embed` / `cudf_fragments` include 目录 | 保留上游的内容；去掉分支的 include 目录（这个目录根本不存在） |
| `cpp/src/io/parquet/parquet_gpu.hpp` | 上游把 `decode_kernel_mask` 的 bit 26 给了 `DICT_INT32`，FastLanes 也用了这个 bit | 保留 `DICT_INT32 = 1 << 26`；FastLanes 的 RAW / DELTA_BINARY / SPLIT64 挪到 bit 27 / 28 / 29 |
| `cpp/src/io/parquet/page_enc.cu` | 上游重写了 page 大小计算（`lvl_size`、`MAX_PARQUET_PAGE_SIZE`、`PAGE_SIZE_OVERFLOW` 错误） | 用上游的计算方式，再叠加 FastLanes 的最坏情况预留；因为 level 字节在 payload 前面，FastLanes 的上界现在也算上 `rle_pad + lvl_size` |
| `cpp/examples/parquet_io/README.md` | 两边都新增了这个文件 | 上游的示例文档在前，后面接 FastLanes 工具说明 |

两边都改过的另外 11 个文件（`writer_impl.cu`、`reader_impl.cpp`、`page_hdr.cu`、`types.hpp`、
`parquet_schema.hpp`、测试的 CMake 文件等）都自动合并成功；FastLanes 新增的 `Encoding`、`column_encoding`
和 `encode_kernel_mask` 取值在上游仍然没有被占用。

### 3.3 26.10 要求的 API 迁移

| 26.10 的变化 | 对 FastLanes 的影响 | 修改 |
| --- | --- | --- |
| `rmm::cuda_stream_view` 被标记为 `[[deprecated]]`；cuDF 用 `-Werror` 编译，并已改用 `cuda::stream_ref`（`fork_streams`、`get_default_stream`） | FastLanes 所有 stream 参数都编译不过 | 16 个文件改用 `cuda::stream_ref`（`.value()` 改为 `.get()`），包括编码器 API、staging 头文件、解码启动函数、测试和示例 |
| `page_state_s` 拆成了组合式状态（`full_page_decode_state`，含 `setup` / `stream` / `nesting` / `output_cvt`）；`update_list_offsets_for_pruned_pages` 被删除，被裁剪的 page 改由 host 端处理 | FastLanes 解码 kernel 编译不过 | kernel 改用 `full_page_decode_state` 并调整字段；被裁剪的 page 和上游所有解码器一样在初始化前退出 |
| cuDF 用 `cuda::minimum` / `cuda::maximum` / `cuda::std::logical_or` 代替 Thrust 的 functor | NATIVE64 的归约 | 改用 `cuda::` functor |
| GCC 14 把使用 `[[deprecated]]` 枚举值当作错误，而 `nv_diag_suppress` 对 host 编译器不起作用 | `writer_impl.cu` 中拒绝 SPLIT64 的 `case` | 在这个 case 标签外面加 `#pragma GCC diagnostic` push/ignore/pop |

FastLanes 路径里的 kernel 启动现在都会检查 `cudaGetLastError()`，staging 拷贝都用 `CUDF_CUDA_TRY`，和上游
的 `EncodePages` 一致。

## 4. 验证

环境：RAPIDS 26.10 的 `cuda13.3-conda` devcontainer（CUDA 13.3.73、GCC 14.4、nvCOMP 5.3.0.16），RTX 5090
Laptop GPU（sm_120）。libcudf 和测试用 `configure-cudf-cpp`、`build-cudf-cpp` 按本机架构做 Release 构建，
测试用 `test-cudf-cpp` 运行。示例程序用 `cpp/examples/build.sh` 基于这次构建编译。

| 测试 target | 测试数 | 结果 |
| --- | ---: | --- |
| `PARQUET_FASTLANES_TEST` | 50 | 全部通过 |
| `PARQUET_TEST` | 551 | 550 个通过；1 个被上游 skip（`ZeroColumnsPreservesRowCount`，[cudf#22935](https://github.com/NVIDIA/cudf/issues/22935)）；另有 4 个上游测试处于禁用状态 |
| `HYBRID_SCAN_TEST` | 101 | 全部通过 |
| `PARQUET_DELETION_VECTORS_TEST` | 6 | 全部通过 |
| `STREAM_IO_PARQUET_TEST` | 4 | 全部通过 |
| `STREAM_IO_HYBRID_SCAN_TEST` | 1 | 全部通过 |

`PARQUET_TEST`、`HYBRID_SCAN_TEST` 和 deletion vector 测试都会经过 reader 的解码分发逻辑，FastLanes 的
kernel mask bit 就是在这里重新编号的。`STREAM_IO_*` 检查 Parquet 路径是否跑在调用方传入的 stream 上。

`PARQUET_FASTLANES_TEST` 覆盖的内容：

- `ParquetCpuEncoderTest`（40 个）：`FASTLANE_BITPACK_RAW` 的 Parquet 写读往返，覆盖所有适用的 INT32 逻辑
  类型（int8/16/32、uint8/16/32、date32、decimal32、duration），以及 1024 值补齐边界、不同 bit 宽度的多
  page 列、header 元数据、SPLIT64 解码和不支持类型的回退。
- `ParquetFastLanesNative64GeneratedTest` 和 `ParquetFastLanesNative64Bw37Test`（各 5 个）：拿 NATIVE64
  GPU 编码器和 CPU 参考实现对比，覆盖各种 bit 宽度，包括 37 bit 边界附近的随机输入和特意构造的输入。这些测试
  直接调用编码器，不经过 Parquet writer 和 reader。

`cpp/examples/parquet_io` 下的示例程序（全部通过）：`fastlane_one_page_sanity_test`（INT32 和 UINT32，
顺序和随机数据，各一个 page，和参考 `pack()` 逐 bit 一致），`fastlane_multi_page_sanity_test`（一个 UINT32
多 page 用例），`fastlane_multi_column_test`（INT32/UINT32 列，顺序和随机数据），
`fastlane_encode_verify_test` 和 `fastlane_int64_cpu_roundtrip`（CPU 参考 kernel），以及
`parquet_io_chunking_sanity`（对 SF100 的 `supplier` 表做分块重写）。前三个程序里的 64 位用例都被注释掉了：
它们写于 NATIVE64 之前，请求的是 SPLIT64。

端到端：TPC-H SF100 benchmark 用 FastLanes 写了 69 个单列文件（12 个 INT64 列和 11 个 INT32 列，三种
codec）和 22 个用到 FastLanes 的整表文件，解码结果全部和源数据一致。

覆盖缺口：没有 gtest 用 `FASTLANES_DELTA_BINARY` 写或读 INT64 列，所以 NATIVE64 的 Parquet 路径（包括
这次合并中移植的解码 kernel）只靠上面的 benchmark 验证过。也没有测试覆盖这几种读 FastLanes page 的情况：
`skip_rows`/`num_rows`、page 裁剪、chunked reader，以及可空列。

## 5. 限制和后续工作

- 编码 ID 不在标准里：10-12 不属于 Parquet 规范，将来可能和规范新增的编码冲突；用了这些编码的文件
  只有这个 cuDF 分支能读。
- 只支持扁平、无 null 的 page：解码器会拒绝含 null 或嵌套的 page。写入路径检查了嵌套但没有检查 null，
  而且没有测试覆盖真正带 null 的数据，所以目前不要用 FastLanes 写有 null 的列（这是读代码得出的结论，没有
  实际跑过）。
- 缺测试：应该补上 `FASTLANES_DELTA_BINARY` 的 Parquet 往返测试（多 page、尾部 page、大 bit 宽度），以及
  用行范围、page 裁剪和 chunked reader 读 FastLanes page 的测试（见第 4 节）。
- 写入慢：RAW32 在 CPU 上编码，每个 page 都要下载、在 host 上打包再上传，每个 page 同步一次 stream。
  NATIVE64 在 GPU 上编码，但仍有按 page 的归约和内存分配。在 TPC-H SF100 上，FastLanes 写入约 1 GB/s，
  DELTA_BINARY_PACKED 是 6-12 GB/s；如果 page 只有 5,000 行，还会再慢 40-60%。
- 只有 FOR，每个 page 一个 bit 宽度。有序或聚集的 key 相邻差值很小，DELTA_BINARY_PACKED（每个 miniblock
  单独选 bit 宽度）压得好得多：在 TPC-H 的 key 上是每值 0.03-5 bit，FastLanes 要 13-17 bit。生成的
  FastLanes 代码里已经有 FastLanes 做 delta 编码用的 transpose 和前缀和（`unrsum`）kernel，但现在的编码器
  没有用上。
- 常量 page 每个值也要 1 bit：`encode_common.hpp` 里的 `compute_bitwidth` 最小返回 1，所以常量列
  （TPC-H 的 `o_shippriority`）每值 1.07 bit，DICTIONARY 只要 0.02 bit。
- Page 补齐：每个 page 都补齐到整数个 1024 值的 vector。cuDF 按完整的 page fragment 组 page（定长列
  默认 5000 行一个 fragment），所以默认的 page 是 5000 或 20000 行，FastLanes 会多出约 2.4% 的补齐开销；把
  `max_page_fragment_size` 设成 1024 的倍数就没有这部分开销了（benchmark 报告里有具体数字）。
- 不支持 decimal：DECIMAL64 列（TPC-H 里很多）不能用 FastLanes。
- SPLIT64 已废弃：因为子 vector 的 page 补齐有 bug，写入时直接拒绝；为了能读已有的文件，保留了解码。
- 预留空间偏保守：Page buffer 按每值 32 或 64 bit 预留，比编码后的实际大小大，会抬高 writer 的峰值内存。
