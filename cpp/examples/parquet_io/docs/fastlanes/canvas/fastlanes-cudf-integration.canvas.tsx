import {
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

type Step = { title: string; file: string; note: string; fastlanes?: boolean };

const writePath: Step[] = [
  { title: "请求编码", file: "column_in_metadata::set_encoding", note: "FASTLANE_BITPACK_RAW 或 FASTLANES_DELTA_BINARY" },
  { title: "校验 schema", file: "writer_impl.cu", note: "类型白名单；SPLIT64 直接拒绝", fastlanes: true },
  { title: "选 kernel mask", file: "page_enc.cu data_encoding_for_col", note: "bit 6（RAW）/ bit 7（DELTA_BINARY）", fastlanes: true },
  { title: "预留 page", file: "InitEncoderPages", note: "128 B header + 补齐后的 vector + level", fastlanes: true },
  { title: "分发", file: "EncodePages", note: "每种编码 fork 一个 stream" },
  { title: "收集 + 编码", file: "fastlanes_page_encoder_*.cuh", note: "RAW32 在 CPU 上，NATIVE64 在 GPU 上", fastlanes: true },
  { title: "组装 page", file: "gpuEncodeCpuPages + finish_page_encode", note: "level、header、统计信息、压缩" },
];

const readPath: Step[] = [
  { title: "识别编码", file: "page_hdr.cu kernel_mask_for_page", note: "ID 10/11/12 对应 mask bit 27/28/29", fastlanes: true },
  { title: "分发", file: "reader_impl.cpp decode_page_data", note: "每个用到的 kernel 一个 stream" },
  { title: "解码", file: "page_fastlanes_decode.cu", note: "每个 page 一个 warp，每 1024 个值一个 vector", fastlanes: true },
];

function Pipeline({ steps }: { steps: Step[] }) {
  const theme = useHostTheme();
  return (
    <Row gap={6} wrap align="stretch">
      {steps.map((s, i) => (
        <div key={s.title} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            style={{
              width: 168,
              padding: "8px 10px",
              borderRadius: 6,
              border: `1px solid ${s.fastlanes ? theme.accent.primary : theme.stroke.secondary}`,
              background: theme.fill.quaternary,
            }}
          >
            <Text size="small" weight="semibold">
              {i + 1}. {s.title}
            </Text>
            <Text size="small" tone="secondary">
              <Code>{s.file}</Code>
            </Text>
            <Text size="small" tone="tertiary">
              {s.note}
            </Text>
          </div>
          {i < steps.length - 1 ? (
            <Text tone="tertiary" size="small">
              {"->"}
            </Text>
          ) : null}
        </div>
      ))}
    </Row>
  );
}

export default function FastLanesIntegration() {
  const theme = useHostTheme();
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>FastLanes 在 cuDF 中的集成：已合入 release/26.10</H1>
        <Text tone="secondary">
          <Code>fastlane-working</Code> 分支合并了 rapidsai/cudf 的 <Code>release/26.10</Code>（上游{" "}
          <Code>eaa1e31058</Code>，多出 945 个提交），适配了 26.10 的 API，并在 26.10 CUDA 13.3 devcontainer、RTX
          5090 Laptop GPU（sm_120）上验证过。
        </Text>
      </Stack>

      <Row gap={32} wrap>
        <Stat value="50 / 50" label="PARQUET_FASTLANES_TEST" tone="success" />
        <Stat value="550 + 1 skip" label="PARQUET_TEST（skip 来自上游）" tone="success" />
        <Stat value="101 / 101" label="HYBRID_SCAN_TEST" tone="success" />
        <Stat value="4" label="处文本冲突" />
      </Row>

      <Divider />

      <Stack gap={10}>
        <H2>FastLanes 怎样接入 cuDF</H2>
        <Text tone="secondary">
          FastLanes 增加了三种 Parquet page 编码。列通过常规的按列编码设置启用；writer 输出普通的 Parquet page，只是
          value 部分放的是 FastLanes 数据；reader 识别新的编码 ID。高亮的步骤是 FastLanes 专有代码，其余都是 cuDF
          原有代码。
        </Text>
        <H3>写入路径</H3>
        <Pipeline steps={writePath} />
        <H3>读取路径</H3>
        <Pipeline steps={readPath} />
      </Stack>

      <Callout tone="info" title="目前启用的两种模式都是 FOR 编码">
        每个值存成 <Code>value - page_minimum</Code>，每个 page 一个 bit 宽度，按 FastLanes 交错的 1024 值 vector
        排列。<Code>FASTLANES_DELTA_BINARY</Code> 名字里有 delta，打包的却是 <Code>value - base</Code>，不是相邻值之差，
        所以在有序 key 上不如 DELTA_BINARY_PACKED。
      </Callout>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Stack gap={8}>
          <H3>编码</H3>
          <Table
            headers={["编码", "ID", "物理类型", "在哪编码"]}
            rows={[
              ["FASTLANE_BITPACK_RAW", "10", "INT32（整数、date32、decimal32、duration）", "CPU"],
              ["FASTLANES_DELTA_BINARY", "11", "INT64（int64、uint64）", "GPU"],
              ["FASTLANE_BITPACK_SPLIT64", "12", "INT64，写入时拒绝", "只解码"],
            ]}
            columnAlign={["left", "right", "left", "left"]}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Page 数据头（128 字节）</H3>
          <Table
            headers={["偏移", "字段", "含义"]}
            rows={[
              ["0", "bitwidth_lo", "每值 bit 数"],
              ["1", "bitwidth_hi", "SPLIT64 高位部分"],
              ["2", "pre_delta", "布局策略标志"],
              ["4", "original_count", "page 中的值个数"],
              ["8", "padded_count", "向上取整到 1024"],
              ["12", "body_size", "payload 字节数"],
              ["16 / 20", "min_value_lo / hi", "page 最小值（基准）"],
            ]}
            columnAlign={["right", "left", "left"]}
          />
        </Stack>
      </Grid>

      <Stack gap={8}>
        <H3>代码分布</H3>
        <Table
          headers={["层", "位置", "行数", "作用"]}
          rows={[
            ["核心头文件", "cpp/include/cudf/fastlanes/", "5,738", "page header、大小计算、生成的设备端 unpack、NATIVE64 kernel"],
            ["生成的 CPU kernel", "cpp/src/fastlanes/*.cpp", "109,723", "FastLanes 参考 pack/unpack/FFOR kernel"],
            ["编码器", "cpp/src/fastlanes/encode_*.cu", "759", "RAW32（CPU）、NATIVE64（GPU）、SPLIT64（旧版）"],
            ["Parquet 衔接代码", "cpp/src/io/parquet/fastlanes_*、page_fastlanes_decode.cu", "1,287", "适用性判断、编码 staging、解码 kernel"],
            ["Parquet 挂接点", "writer_impl.cu、page_enc.cu、page_hdr.cu、reader_impl.cpp", "约 500", "校验、大小计算、分发"],
            ["测试", "cpp/tests/io/parquet_fastlanes_*", "2,865", "PARQUET_FASTLANES_TEST（50 个测试）"],
          ]}
          columnAlign={["left", "left", "right", "left"]}
        />
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>合并 26.10 时改了什么</H2>
        <Table
          headers={["文件", "冲突或问题", "处理"]}
          rows={[
            ["cpp/CMakeLists.txt", "上游在一个失效的 FastLanes include 旁边加了 JIT include 目录", "保留上游；去掉并不存在的 src/io/parquet/fastlanes"],
            ["parquet_gpu.hpp", "decode_kernel_mask 的 bit 26 被上游的 DICT_INT32 占用", "FastLanes 的 RAW / DELTA_BINARY / SPLIT64 挪到 bit 27 / 28 / 29"],
            ["page_enc.cu", "上游重写了 page 大小计算和大小检查", "用上游的计算，再加上 FastLanes 预留，现在也算上 level 字节"],
            ["examples README", "两边都新增了这个文件", "上游文档在前，后面接 FastLanes 工具说明"],
            ["16 个 FastLanes 文件", "rmm::cuda_stream_view 已废弃，且开了 -Werror", "改用 cuda::stream_ref"],
            ["page_fastlanes_decode.cu", "page_state_s 被组合式 decode state 取代", "改用 full_page_decode_state；被裁剪的 page 和上游一样提前退出"],
            ["writer_impl.cu", "GCC 14 不接受已废弃的 SPLIT64 case 标签", "局部关闭 -Wdeprecated-declarations"],
          ]}
          rowTone={[undefined, "warning", "warning", undefined, "warning", "warning", undefined]}
        />
        <Text size="small" tone="tertiary">
          fastlane-working 上的提交：c8533b8d3d（合并）、9ed93250d2（26.10 API 适配）、e0f905a0ae（benchmark
          工具）、0ce7f3b959（报告）。合并前的状态备份在 backup/fastlane-working-pre-26.10。
        </Text>
      </Stack>

      <Stack gap={10}>
        <H2>验证</H2>
        <Table
          headers={["检查项", "结果"]}
          rows={[
            ["PARQUET_FASTLANES_TEST", "50 / 50 通过"],
            ["PARQUET_TEST", "550 个通过，1 个 skip（上游的 ZeroColumnsPreservesRowCount）"],
            ["HYBRID_SCAN_TEST", "101 / 101 通过"],
            ["PARQUET_DELETION_VECTORS_TEST", "6 / 6 通过"],
            ["STREAM_IO_PARQUET_TEST、STREAM_IO_HYBRID_SCAN_TEST", "4 / 4、1 / 1 通过"],
            ["FastLanes 示例程序（5 个）和 parquet_io_chunking_sanity", "全部通过"],
            ["lineitem SF1 往返，7 列用 FastLanes", "49 / 49 个 row group 一致"],
            ["TPC-H SF100：69 个单列文件和 22 个用到 FastLanes 的整表文件", "解码结果全部和源数据一致"],
            ["INT64 FASTLANES_DELTA_BINARY 经过 Parquet writer 和 reader", "没有 gtest，只有 SF100 benchmark 覆盖"],
          ]}
          rowTone={["success", "success", "success", "success", "success", "success", "success", "success", "warning"]}
        />
        <Text size="small" tone="secondary">
          另外还没有测试覆盖：用 skip_rows / num_rows、page 裁剪、chunked reader 读 FastLanes page，以及可空列。
          NATIVE64 的 gtest 只拿 GPU 编码器和 CPU 参考实现对比，不经过 Parquet。
        </Text>
        <Text size="small" tone="tertiary">
          环境：RAPIDS 26.10 devcontainer（cuda13.3-conda），CUDA 13.3.73，GCC 14.4，RMM 26.10，nvCOMP 5.3.0.16；
          用 build-cudf-cpp 和 sccache-dist 做 sm_120 的 Release 构建。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>限制</H2>
        <Grid columns={2} gap={12}>
          <Text size="small">
            <Text weight="semibold">只有 cuDF 能读。</Text>编码 ID 10-12 不在 Parquet 规范里，其他 reader 会报未知编码。
          </Text>
          <Text size="small">
            <Text weight="semibold">只支持扁平、无 null 的 page。</Text>解码器拒绝含 null 的 page；writer 不检查
            null（读代码得出的结论，没有实测）。
          </Text>
          <Text size="small">
            <Text weight="semibold">写入慢。</Text>TPC-H SF100 上约 1 GB/s，DELTA_BINARY_PACKED 是 6-12 GB/s；RAW32
            每个 page 都在 host 上打包，每个 page 同步一次。
          </Text>
          <Text size="small">
            <Text weight="semibold">只有 FOR。</Text>有序 key 每值 13-17 bit，DELTA_BINARY_PACKED 只要 0.03-5 bit；
            常量 page 每值也要 1 bit。
          </Text>
          <Text size="small">
            <Text weight="semibold">1024 值补齐。</Text>page 行数最好是 1024 的倍数（设置
            max_page_fragment_size）；不支持 DECIMAL64。
          </Text>
          <Text size="small">
            <Text weight="semibold">按列使用才有收益。</Text>只在 FastLanes 最小的列上用，SF100 文件在 SNAPPY 下小
            1.1%，ZSTD 下小 0.6%。
          </Text>
        </Grid>
      </Stack>

      <div style={{ height: 1, background: theme.stroke.tertiary }} />
      <Text size="small" tone="tertiary">
        完整说明：cpp/examples/parquet_io/docs/fastlanes/FASTLANES_INTEGRATION_REPORT_2026-09-27.md。Benchmark：同目录下的
        FASTLANES_TPCH_SF100_BENCHMARK_2026-09-27.md，以及 fastlanes-tpch-sf100-benchmark canvas。
      </Text>
    </Stack>
  );
}
