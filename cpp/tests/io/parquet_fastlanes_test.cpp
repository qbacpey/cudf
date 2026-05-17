#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/fastlanes/common.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda_runtime_api.h>

#include "parquet_common.hpp"

#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

class ParquetCpuEncoderTest : public cudf::test::BaseFixture {
 protected:
  void SetUp() override { clear_cuda_error_state(); }

  void TearDown() override
  {
    clear_cuda_error_state();
  }

 private:
  static void clear_cuda_error_state()
  {
    // Clear sticky async CUDA errors so one failed test does not cascade.
    (void)cudaDeviceSynchronize();
    (void)cudaGetLastError();
  }
};

namespace {

template <typename T>
void write_single_column(std::vector<T> const& values,
                         std::string const& file_name,
                         size_t max_page_rows  = 0,
                         size_t max_page_bytes = 0)
{
  (void)cudaGetLastError();

  auto const filepath = file_name;

  cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end());
  cudf::table_view input({col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("col0");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  auto cleanup = [&]() { std::remove(filepath.c_str()); };

  try {
    auto builder =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(metadata)
        .write_v2_headers(true);
    if (max_page_rows > 0) { builder.max_page_size_rows(max_page_rows); }
    if (max_page_bytes > 0) { builder.max_page_size_bytes(max_page_bytes); }

    cudf::io::write_parquet(builder);
  } catch (...) {
    (void)cudaGetLastError();
    cleanup();
    throw;
  }
  (void)cudaGetLastError();
}

template <typename T>
void roundtrip_single_column(std::vector<T> const& values,
                             std::string const& file_name,
                             size_t max_page_rows  = 0,
                             size_t max_page_bytes = 0)
{
  auto const filepath = file_name;
  auto cleanup        = [&]() { std::remove(filepath.c_str()); };

  write_single_column(values, filepath, max_page_rows, max_page_bytes);

  cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end());
  cudf::table_view input({col});

  try {
    auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result  = cudf::io::read_parquet(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

void roundtrip_with_metadata(cudf::table_view input,
                             cudf::io::table_input_metadata metadata,
                             std::string const& file_name,
                             size_t max_page_rows  = 0,
                             size_t max_page_bytes = 0)
{
  (void)cudaGetLastError();

  auto const filepath = file_name;
  auto cleanup        = [&]() { std::remove(filepath.c_str()); };

  try {
    auto builder =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(metadata)
        .write_v2_headers(true);
    if (max_page_rows > 0) { builder.max_page_size_rows(max_page_rows); }
    if (max_page_bytes > 0) { builder.max_page_size_bytes(max_page_bytes); }

    cudf::io::write_parquet(builder);

    auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result  = cudf::io::read_parquet(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
  } catch (...) {
    (void)cudaGetLastError();
    cleanup();
    throw;
  }

  (void)cudaGetLastError();
  cleanup();
}

cudf::io::parquet::Encoding first_data_page_encoding(std::string const& file_name)
{
  auto const source = cudf::io::datasource::create(file_name);
  cudf::io::parquet::FileMetaData fmd;
  read_footer(source, &fmd);

  if (fmd.row_groups.empty() || fmd.row_groups.front().columns.empty()) {
    throw std::runtime_error("Parquet file has no row-group/column metadata to inspect encoding");
  }

  auto const& first_chunk = fmd.row_groups.front().columns.front().meta_data;
  if (first_chunk.data_page_offset <= 0) {
    throw std::runtime_error("Invalid Parquet data_page_offset when inspecting encoding");
  }

  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::PageHeader), 0});

  if (ph.type == cudf::io::parquet::PageType::DATA_PAGE_V2) {
    return ph.data_page_header_v2.encoding;
  }
  return ph.data_page_header.encoding;
}

cudf::io::parquet::Type first_data_column_physical_type(std::string const& file_name)
{
  auto const source = cudf::io::datasource::create(file_name);
  cudf::io::parquet::FileMetaData fmd;
  read_footer(source, &fmd);

  if (fmd.row_groups.empty() || fmd.row_groups.front().columns.empty()) {
    throw std::runtime_error("Parquet file has no row-group/column metadata to inspect type");
  }

  return fmd.row_groups.front().columns.front().meta_data.type;
}

void expect_first_data_page_encoding(std::string const& file_name,
                                     cudf::io::parquet::Encoding expected_encoding)
{
  EXPECT_EQ(first_data_page_encoding(file_name), expected_encoding);
}

void expect_first_data_column_physical_type(std::string const& file_name,
                                            cudf::io::parquet::Type expected_type)
{
  EXPECT_EQ(first_data_column_physical_type(file_name), expected_type);
}

template <typename T, typename SourceT = T>
void write_single_column_typed_expect_physical_and_encoding(
  std::vector<SourceT> const& values,
  std::string const& file_name,
  cudf::io::parquet::Type expected_physical_type,
  cudf::io::parquet::Encoding encoding_to_check,
  bool expect_encoding,
  bool set_decimal_precision = false,
  int32_t decimal_precision  = 9,
  size_t max_page_rows       = 0,
  size_t max_page_bytes      = 0)
{
  auto const filepath = file_name;
  auto cleanup        = [&]() { std::remove(filepath.c_str()); };

  cudf::test::fixed_width_column_wrapper<T, SourceT> col(values.begin(), values.end());
  cudf::table_view input({col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("col0");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
  if (set_decimal_precision) {
    metadata.column_metadata[0].set_decimal_precision(decimal_precision);
  }

  try {
    auto builder =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(metadata)
        .write_v2_headers(true);
    if (max_page_rows > 0) { builder.max_page_size_rows(max_page_rows); }
    if (max_page_bytes > 0) { builder.max_page_size_bytes(max_page_bytes); }

    cudf::io::write_parquet(builder);

    expect_first_data_column_physical_type(filepath, expected_physical_type);
    if (expect_encoding) {
      EXPECT_EQ(first_data_page_encoding(filepath), encoding_to_check);
    } else {
      EXPECT_NE(first_data_page_encoding(filepath), encoding_to_check);
    }
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

template <typename T>
void roundtrip_single_column_expect_encoding(std::vector<T> const& values,
                                             std::string const& file_name,
                                             cudf::io::parquet::Encoding expected_encoding,
                                             size_t max_page_rows  = 0,
                                             size_t max_page_bytes = 0)
{
  auto const filepath = file_name;
  auto cleanup        = [&]() { std::remove(filepath.c_str()); };

  write_single_column(values, filepath, max_page_rows, max_page_bytes);
  expect_first_data_page_encoding(filepath, expected_encoding);

  cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end());
  cudf::table_view input({col});

  try {
    auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result  = cudf::io::read_parquet(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

template <typename T, typename SourceT>
void roundtrip_single_column_typed_expect_encoding(std::vector<SourceT> const& values,
                                                   std::string const& file_name,
                                                   cudf::io::parquet::Encoding expected_encoding,
                                                   size_t max_page_rows  = 0,
                                                   size_t max_page_bytes = 0)
{
  auto const filepath = file_name;
  auto cleanup        = [&]() { std::remove(filepath.c_str()); };

  cudf::test::fixed_width_column_wrapper<T, SourceT> col(values.begin(), values.end());
  cudf::table_view input({col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("col0");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  try {
    auto builder =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(metadata)
        .write_v2_headers(true);
    if (max_page_rows > 0) { builder.max_page_size_rows(max_page_rows); }
    if (max_page_bytes > 0) { builder.max_page_size_bytes(max_page_bytes); }

    cudf::io::write_parquet(builder);

    expect_first_data_page_encoding(filepath, expected_encoding);

    auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result  = cudf::io::read_parquet(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

template <typename T, typename SourceT>
void roundtrip_single_column_typed_expect_not_encoding(
  std::vector<SourceT> const& values,
  std::string const& file_name,
  cudf::io::parquet::Encoding unexpected_encoding,
  size_t max_page_rows  = 0,
  size_t max_page_bytes = 0)
{
  auto const filepath = file_name;
  auto cleanup        = [&]() { std::remove(filepath.c_str()); };

  cudf::test::fixed_width_column_wrapper<T, SourceT> col(values.begin(), values.end());
  cudf::table_view input({col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("col0");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  try {
    auto builder =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(metadata)
        .write_v2_headers(true);
    if (max_page_rows > 0) { builder.max_page_size_rows(max_page_rows); }
    if (max_page_bytes > 0) { builder.max_page_size_bytes(max_page_bytes); }

    cudf::io::write_parquet(builder);

    EXPECT_NE(first_data_page_encoding(filepath), unexpected_encoding);

    auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result  = cudf::io::read_parquet(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

// void roundtrip_two_columns(std::vector<int32_t> const& values_i32,
//                            std::vector<uint32_t> const& values_u32,
//                            std::string const& file_name,
//                            size_t max_page_rows,
//                            size_t max_page_bytes)
// {
//   (void)cudaGetLastError();

//   auto const filepath = file_name;

//   cudf::test::fixed_width_column_wrapper<int32_t> col_i32(values_i32.begin(), values_i32.end());
//   cudf::test::fixed_width_column_wrapper<uint32_t> col_u32(values_u32.begin(), values_u32.end());
//   cudf::table_view input({col_i32, col_u32});

//   cudf::io::table_input_metadata metadata(input);
//   metadata.column_metadata[0].set_name("i32_col");
//   metadata.column_metadata[1].set_name("u32_col");
//   metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
//   metadata.column_metadata[1].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

//   auto cleanup = [&]() { std::remove(filepath.c_str()); };

//   try {
//     auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
//                       .metadata(metadata)
//                       .write_v2_headers(true)
//                       .max_page_size_rows(max_page_rows)
//                       .max_page_size_bytes(max_page_bytes);

//     cudf::io::write_parquet(out_opts);

//     auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
//     auto result  = cudf::io::read_parquet(in_opts);

//     CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
//   } catch (...) {
//     (void)cudaGetLastError();
//     cleanup();
//     throw;
//   }
//   (void)cudaGetLastError();
//   cleanup();
// }

}  // namespace

TEST_F(ParquetCpuEncoderTest, SimpleInt32SinglePage)
{
  std::vector<int32_t> values(2050);
  std::iota(values.begin(), values.end(), 2);
  roundtrip_single_column(values, "test_cpu_encoder_int32.parquet");
}

TEST_F(ParquetCpuEncoderTest, SimpleInt32MultiplePages)
{
  std::vector<int32_t> values(50000);
  std::iota(values.begin(), values.end(), 4);
  roundtrip_single_column(values, "test_cpu_encoder_int32_multi_page.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, SimpleInt32SinglePageNegative)
{
  std::vector<int32_t> values(2050);
  std::iota(values.begin(), values.end(), -1025);
  values[2]    = 0;
  values[3]    = 1;
  roundtrip_single_column(values, "test_cpu_encoder_int32_negative.parquet");
}


TEST_F(ParquetCpuEncoderTest, FastLanesInt32TinySizesAndPaddingBoundaries)
{
  std::vector<int32_t> const test_sizes = {
    1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2047, 2048,
    2049};

  for (auto const n : test_sizes) {
    std::vector<int32_t> values(n);
    std::iota(values.begin(), values.end(), 0);
    std::string file_name = "test_fastlanes_i32_size_" + std::to_string(n) + ".parquet";
    roundtrip_single_column(values, file_name, 1024, 4096);
  }
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt32MultiPageDifferentBitwidths)
{
  // 3 logical pages of 1024 rows each with very different magnitude distributions.
  std::vector<int32_t> values(3 * 1024, 0);

  // Page 0: tiny values -> low bitwidth
  for (int i = 0; i < 1024; ++i) { values[i] = i % 16; }

  // Page 1: medium values -> higher bitwidth
  for (int i = 0; i < 1024; ++i) { values[1024 + i] = (1 << 20) + i; }

  // Page 2: negatives present, but page-local normalization should still compress it.
  for (int i = 0; i < 1024; ++i) { values[2048 + i] = -2000 + (i % 97); }

  roundtrip_single_column(values, "test_fastlanes_i32_multi_page_bw.parquet", 1024, 4096);
}

// TEST_F(ParquetCpuEncoderTest, FastLanesMultiColumnMultiPage)
// {
//   std::vector<int32_t> values_i32(4097, 0);
//   std::vector<uint32_t> values_u32(4097, 0);

//   for (size_t i = 0; i < values_i32.size(); ++i) {
//     values_i32[i] = (i % 3 == 0) ? -static_cast<int32_t>(i) : static_cast<int32_t>(i * 7);
//     values_u32[i] = (i % 5 == 0) ? std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(i)
//                                  : static_cast<uint32_t>(i * 13);
//   }
// TODO: 这个函数的实现也有问题，得处理一下
//   roundtrip_two_columns(
//     values_i32, values_u32, "test_fastlanes_multi_col_multi_page.parquet", 512, 4096);
// }


TEST_F(ParquetCpuEncoderTest, Int32WithSpecificValuesLarge)
{
  // Keep a larger mixed-pattern case as a stress test.
  std::vector<int32_t> values(1000);
  std::iota(values.begin(), values.end(), 0);
  // 这个只有一个Vector所以不会干扰
  values[0]   = -1;
  values[500] = 12345;
  values[999] = -12345;
  roundtrip_single_column(values, "test_cpu_encoder_specific_large.parquet", 256, 4096);
}

TEST_F(ParquetCpuEncoderTest, FallbackForUnsupportedType)
{
  std::vector<double> values(1000);
  std::iota(values.begin(), values.end(), 0.0);

  auto const filepath = "test_cpu_encoder_unsupported_type.parquet";
  auto cleanup        = [&]() { std::remove(filepath); };

  try {
    cudf::test::fixed_width_column_wrapper<double> col(values.begin(), values.end());
    cudf::table_view input({col});

    cudf::io::table_input_metadata metadata(input);
    metadata.column_metadata[0].set_name("col0");
    metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

    auto builder =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
        .metadata(metadata)
        .write_v2_headers(true);

    cudf::io::write_parquet(builder);

    auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    auto result  = cudf::io::read_parquet(in_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, result.tbl->view());
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
}

TEST_F(ParquetCpuEncoderTest, Int32WithSpecificValues)
{
  // Small-vector edge case with negatives and mixed magnitudes.
  // std::vector<int32_t> values(1000);
  // TODO: 其实这么一看出问题的测试用例都是编码过后的大小还不如未编码过后的，所以第一，元素过少的话，第二，元素过大（位宽过大）的话，似乎都会出问题
  // 所以我觉得最好的策略是如果发现编码过后的数据比未编码过后的数据还大的话，干脆Fallback成其他编码算了
  std::vector<int32_t> values(1000);
  // std::vector<int32_t> values = {0, -1, 1, 100, -100, 2, -2, 12345, -12345, 42};
  std::iota(values.begin(), values.end(), 0);
  values.push_back(1);
  values.push_back(-1);
  values.push_back(100);
  values.push_back(-100);
  values.push_back(2);
  values.push_back(-2);
  values.push_back(12345);
  values.push_back(-12345);
  values.push_back(42);
  roundtrip_single_column(values, "test_cpu_encoder_specific.parquet");
}

TEST_F(ParquetCpuEncoderTest, FastLanesUInt32LogicalTypeForcedBitpack)
{
  std::vector<uint32_t> values(2049, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = std::numeric_limits<uint32_t>::max() - 4096u + static_cast<uint32_t>(i % 251);
  }

  values[0]        = std::numeric_limits<uint32_t>::max() - 17u;
  values[1]        = std::numeric_limits<uint32_t>::max() - 16u;
  values[1023]     = std::numeric_limits<uint32_t>::max() - 3u;
  values[1024]     = std::numeric_limits<uint32_t>::max() - 200u;
  values[2048]     = std::numeric_limits<uint32_t>::max() - 1u;

  roundtrip_single_column_typed_expect_encoding<uint32_t, uint32_t>(
    values,
    "test_fastlanes_u32_logical_type.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesUInt32BoundaryAcrossSignedSplitForcedBitpack)
{
  // Keep unsigned boundary coverage while requiring FastLanes for UINT32.
  std::vector<uint32_t> values(2048, 0);

  for (size_t i = 0; i < 1024; ++i) {
    values[i] = static_cast<uint32_t>(i % 257);
  }
  for (size_t i = 1024; i < values.size(); ++i) {
    values[i] = std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(i % 257);
  }

  values[0]    = 0u;
  values[1]    = static_cast<uint32_t>(std::numeric_limits<int32_t>::max() - 1);
  values[1023] = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  values[1024] = static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u;
  values[1025] = std::numeric_limits<uint32_t>::max() - 1u;
  values[2047] = std::numeric_limits<uint32_t>::max();

  roundtrip_single_column_typed_expect_encoding<uint32_t, uint32_t>(
    values,
    "test_fastlanes_u32_signed_split_boundary.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt32NegativeCastingAndExtremes)
{
  std::vector<int32_t> values(2050, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = -250000 + static_cast<int32_t>((i * 37) % 500000);
  }

  values[0]    = -123456789;
  values[1]    = -1;
  values[2]    = 0;
  values[3]    = 1;
  values[1024] = 123456789;
  values[2049] = -123456700;

  // Print out 2048 and 2049 to verify the intended edge case is present in the test data.
  std::cout << "Value at 2048: " << values[2048] << " (0x" << std::hex << values[2048] << std::dec << ")\n";
  std::cout << "Value at 2049: " << values[2049] << " (0x" << std::hex << values[2049] << std::dec << ")\n";

  roundtrip_single_column(values, "test_fastlanes_i32_negative_edges.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt32HighBitwidthFullPages)
{
  // Diagnostic: same value distribution style as the failing case, but exactly two full pages.
  // If this passes while tiny-tail variants fail, the issue is page-size planning for tails,
  // not high bitwidth by itself.
  std::vector<int32_t> values(2048, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = -250000 + static_cast<int32_t>((i * 37) % 500000);
  }

  values[0]    = -123456789;
  values[1]    = -1;
  values[2]    = 0;
  values[3]    = 1;
  values[1024] = 123456789;
  values[2047] = -123456700;

  roundtrip_single_column(values, "test_fastlanes_i32_high_bw_full_pages.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt32HighBitwidthTinyTailPage)
{
  // Diagnostic: high-bitwidth data with a tiny last page (2050 = 1024 + 1024 + 2).
  // This directly exercises the tail-page reservation path.
  std::vector<int32_t> values(2050, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = -250000 + static_cast<int32_t>((i * 37) % 500000);
  }

  values[0]    = -123456789;
  values[1]    = -1;
  values[2]    = 0;
  values[3]    = 1;
  values[1024] = 123456789;
  values[2049] = -123456700;

  roundtrip_single_column(values, "test_fastlanes_i32_high_bw_tiny_tail.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesHeaderRoundTripPreservesMinValue)
{
  auto const min_value = fastlanes::int32_to_u32_bits(-1234567);
  auto blob            = fastlanes::PageHeader::serialize(
    17, fastlanes::TypeCastMode::SIGNED_REINTERPRET, 100, 1024, min_value, nullptr, 0);

  auto const header = fastlanes::PageHeader::deserialize(blob.data());

  EXPECT_EQ(header.bitwidth, 17);
  EXPECT_EQ(header.cast_mode, fastlanes::TypeCastMode::SIGNED_REINTERPRET);
  EXPECT_EQ(header.original_count, 100);
  EXPECT_EQ(header.padded_count, 1024);
  EXPECT_EQ(header.body_size, 0);
  EXPECT_EQ(header.min_value, min_value);
  EXPECT_EQ(fastlanes::u32_bits_to_int32(header.min_value), -1234567);
}

TEST_F(ParquetCpuEncoderTest, FastLanesHeaderRejectsLegacyUnsignedCastMode)
{
  auto blob = fastlanes::PageHeader::serialize(
    7, static_cast<fastlanes::TypeCastMode>(0), 64, 1024, 0u, nullptr, 0);

  auto const header = fastlanes::PageHeader::deserialize(blob.data());

  EXPECT_FALSE(fastlanes::is_valid_cast_mode(header.cast_mode));
  EXPECT_FALSE(fastlanes::is_valid_cast_mode(static_cast<uint8_t>(header.cast_mode)));
  EXPECT_TRUE(fastlanes::is_valid_cast_mode(fastlanes::TypeCastMode::SIGNED_SAFE));
  EXPECT_TRUE(fastlanes::is_valid_cast_mode(fastlanes::TypeCastMode::SIGNED_REINTERPRET));
}

TEST_F(ParquetCpuEncoderTest, FastLanesMixedEncodingsWithDate32LogicalType)
{
  constexpr int num_rows = 4099;

  std::vector<int64_t> l_orderkey(num_rows);
  std::vector<int64_t> l_partkey(num_rows);
  std::vector<int64_t> l_linenumber(num_rows);
  std::vector<int32_t> l_returnflag(num_rows);
  std::vector<int32_t> l_linestatus(num_rows);
  std::vector<cudf::timestamp_D::rep> l_shipdate(num_rows);
  std::vector<int32_t> l_receiptdate(num_rows);
  std::vector<int32_t> l_shipmode(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    l_orderkey[i]   = static_cast<int64_t>(i) * 17;
    l_partkey[i]    = static_cast<int64_t>(i % 250000);
    l_linenumber[i] = static_cast<int64_t>((i % 7) + 1);

    l_returnflag[i] = i % 3;
    l_linestatus[i] = i % 2;
    l_shipmode[i]   = i % 7;

    l_shipdate[i]    = 19000 + (i % 3650);
    l_receiptdate[i] = 19030 + (i % 3650);
  }

  cudf::test::fixed_width_column_wrapper<int64_t> col_orderkey(l_orderkey.begin(),
                                                                l_orderkey.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_partkey(l_partkey.begin(), l_partkey.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_linenumber(l_linenumber.begin(),
                                                                  l_linenumber.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_returnflag(l_returnflag.begin(),
                                                                  l_returnflag.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_linestatus(l_linestatus.begin(),
                                                                  l_linestatus.end());
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> col_shipdate(
    l_shipdate.begin(), l_shipdate.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_receiptdate(l_receiptdate.begin(),
                                                                   l_receiptdate.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_shipmode(l_shipmode.begin(), l_shipmode.end());

  cudf::table_view input({col_orderkey,
                          col_partkey,
                          col_linenumber,
                          col_returnflag,
                          col_linestatus,
                          col_shipdate,
                          col_receiptdate,
                          col_shipmode});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("l_orderkey");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::DELTA_BINARY_PACKED);
  metadata.column_metadata[1].set_name("l_partkey");
  metadata.column_metadata[1].set_encoding(cudf::io::column_encoding::DELTA_BINARY_PACKED);
  metadata.column_metadata[2].set_name("l_linenumber");
  metadata.column_metadata[2].set_encoding(cudf::io::column_encoding::DICTIONARY);
  metadata.column_metadata[3].set_name("l_returnflag");
  metadata.column_metadata[3].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
  metadata.column_metadata[4].set_name("l_linestatus");
  metadata.column_metadata[4].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
  metadata.column_metadata[5].set_name("l_shipdate");
  metadata.column_metadata[5].set_encoding(cudf::io::column_encoding::DICTIONARY);
  metadata.column_metadata[6].set_name("l_receiptdate");
  metadata.column_metadata[6].set_encoding(cudf::io::column_encoding::DELTA_BINARY_PACKED);
  metadata.column_metadata[7].set_name("l_shipmode");
  metadata.column_metadata[7].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  roundtrip_with_metadata(
    input, metadata, "test_fastlanes_mixed_with_date32_logical.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDate32LogicalTypeForcedBitpack)
{
  constexpr int num_rows = 2050;
  std::vector<cudf::timestamp_D::rep> values(num_rows);
  for (int i = 0; i < num_rows; ++i) { values[i] = 18000 + (i % 4000); }

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> col(
    values.begin(), values.end());
  cudf::table_view input({col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("date32_col");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  roundtrip_with_metadata(input, metadata, "test_fastlanes_date32_forced_bitpack.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDate32LogicalTypeForcedBitpackNegativePreEpoch)
{
  constexpr int num_rows = 3077;
  std::vector<cudf::timestamp_D::rep> values(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    values[i] = static_cast<cudf::timestamp_D::rep>((i % 4000) - 2100);
  }
  values[0]          = static_cast<cudf::timestamp_D::rep>(-1);
  values[1]          = static_cast<cudf::timestamp_D::rep>(0);
  values[2]          = static_cast<cudf::timestamp_D::rep>(1);
  values[num_rows - 1] = static_cast<cudf::timestamp_D::rep>(-3650);

  roundtrip_single_column_typed_expect_encoding<cudf::timestamp_D>(
    values,
    "test_fastlanes_date32_negative_forced_bitpack.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDecimal32LogicalTypeForcedBitpack)
{
  constexpr int num_rows = 3073;
  std::vector<numeric::decimal32> values;
  values.reserve(num_rows);

  auto const scale = numeric::scale_type{0};
  for (int i = 0; i < num_rows; ++i) {
    auto const raw = (i % 2 == 0) ? (100000 + i * 3) : -(50000 + i * 5);
    values.emplace_back(raw, scale);
  }

  cudf::test::fixed_width_column_wrapper<numeric::decimal32> col(values.begin(), values.end());
  cudf::table_view input({col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("decimal32_col").set_decimal_precision(9);
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  roundtrip_with_metadata(
    input, metadata, "test_fastlanes_decimal32_forced_bitpack.parquet", 1024, 4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationMillisLogicalTypeForcedBitpack)
{
  constexpr int num_rows = 3075;
  std::vector<cudf::duration_ms::rep> values(num_rows);
  for (int i = 0; i < num_rows; ++i) { values[i] = static_cast<cudf::duration_ms::rep>(i * 11); }

  roundtrip_single_column_typed_expect_encoding<cudf::duration_ms>(
    values,
    "test_fastlanes_duration_ms_forced_bitpack.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationMillisLogicalTypeForcedBitpackNegative)
{
  constexpr int num_rows = 3075;
  std::vector<cudf::duration_ms::rep> values(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    auto const shifted = static_cast<int64_t>(i % 2048) - 1024;
    values[i]          = static_cast<cudf::duration_ms::rep>(shifted * 11);
  }
  values[0]            = static_cast<cudf::duration_ms::rep>(-1);
  values[1]            = static_cast<cudf::duration_ms::rep>(0);
  values[2]            = static_cast<cudf::duration_ms::rep>(1);
  values[num_rows - 1] = static_cast<cudf::duration_ms::rep>(-86400000);

  roundtrip_single_column_typed_expect_encoding<cudf::duration_ms>(
    values,
    "test_fastlanes_duration_ms_negative_forced_bitpack.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationSecondsLogicalTypeForcedBitpackNonNegative)
{
  constexpr int num_rows = 2051;
  std::vector<cudf::duration_s::rep> input_seconds(num_rows);
  std::vector<cudf::duration_ms::rep> expected_millis(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    auto const seconds = static_cast<cudf::duration_s::rep>(i);
    input_seconds[i]   = seconds;
    expected_millis[i] = static_cast<cudf::duration_ms::rep>(seconds * 1000);
  }

  cudf::test::fixed_width_column_wrapper<cudf::duration_s, cudf::duration_s::rep> input_col(
    input_seconds.begin(), input_seconds.end());
  cudf::table_view input({input_col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("time_millis_seconds_col");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  auto const filepath = "test_fastlanes_duration_s_forced_bitpack_roundtrip.parquet";
  auto cleanup        = [&]() { std::remove(filepath); };

  auto builder = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
                   .metadata(metadata)
                   .write_v2_headers(true)
                   .max_page_size_rows(1024)
                   .max_page_size_bytes(4096);
  cudf::io::write_parquet(builder);

  expect_first_data_page_encoding(filepath, cudf::io::parquet::Encoding::FASTLANES_BITPACK);

  auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result  = cudf::io::read_parquet(in_opts);

  cudf::test::fixed_width_column_wrapper<cudf::duration_ms, cudf::duration_ms::rep> expected_col(
    expected_millis.begin(), expected_millis.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result.tbl->view().column(0));

  cleanup();
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationSecondsLogicalTypeForcedBitpackNegative)
{
  constexpr int num_rows = 2051;
  std::vector<cudf::duration_s::rep> input_seconds(num_rows);
  std::vector<cudf::duration_ms::rep> expected_millis(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    auto const seconds  = static_cast<cudf::duration_s::rep>((i % 2048) - 1024);
    input_seconds[i]    = seconds;
    expected_millis[i]  = static_cast<cudf::duration_ms::rep>(seconds * 1000);
  }

  input_seconds[0]      = static_cast<cudf::duration_s::rep>(-1);
  input_seconds[1]      = static_cast<cudf::duration_s::rep>(0);
  input_seconds[2]      = static_cast<cudf::duration_s::rep>(1);
  input_seconds[num_rows - 1] = static_cast<cudf::duration_s::rep>(-86400);
  expected_millis[0]    = static_cast<cudf::duration_ms::rep>(-1000);
  expected_millis[1]    = static_cast<cudf::duration_ms::rep>(0);
  expected_millis[2]    = static_cast<cudf::duration_ms::rep>(1000);
  expected_millis[num_rows - 1] = static_cast<cudf::duration_ms::rep>(-86400000);

  cudf::test::fixed_width_column_wrapper<cudf::duration_s, cudf::duration_s::rep> input_col(
    input_seconds.begin(), input_seconds.end());
  cudf::table_view input({input_col});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("time_millis_seconds_col_negative");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  auto const filepath = "test_fastlanes_duration_s_negative_forced_bitpack_roundtrip.parquet";
  auto cleanup        = [&]() { std::remove(filepath); };

  auto builder = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
                   .metadata(metadata)
                   .write_v2_headers(true)
                   .max_page_size_rows(1024)
                   .max_page_size_bytes(4096);
  cudf::io::write_parquet(builder);

  expect_first_data_page_encoding(filepath, cudf::io::parquet::Encoding::FASTLANES_BITPACK);

  auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result  = cudf::io::read_parquet(in_opts);

  cudf::test::fixed_width_column_wrapper<cudf::duration_ms, cudf::duration_ms::rep> expected_col(
    expected_millis.begin(), expected_millis.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col, result.tbl->view().column(0));

  cleanup();
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt8LogicalTypeForcedBitpack)
{
  std::vector<int8_t> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int8_t>((i % 127) - 63);
  }
  roundtrip_single_column_expect_encoding(values,
                                          "test_fastlanes_i8_forced_bitpack_roundtrip.parquet",
                                          cudf::io::parquet::Encoding::FASTLANES_BITPACK,
                                          1024,
                                          4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt16LogicalTypeForcedBitpack)
{
  std::vector<int16_t> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int16_t>((i * 11) % 32767 - 16384);
  }
  roundtrip_single_column_expect_encoding(values,
                                          "test_fastlanes_i16_forced_bitpack_roundtrip.parquet",
                                          cudf::io::parquet::Encoding::FASTLANES_BITPACK,
                                          1024,
                                          4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesUInt8LogicalTypeForcedBitpack)
{
  std::vector<uint8_t> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint8_t>(i % 251);
  }
  roundtrip_single_column_expect_encoding(
    values,
    "test_fastlanes_u8_forced_bitpack_roundtrip.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesUInt16LogicalTypeForcedBitpack)
{
  std::vector<uint16_t> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint16_t>((i * 37) % 65535);
  }
  roundtrip_single_column_expect_encoding(
    values,
    "test_fastlanes_u16_forced_bitpack_roundtrip.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt32PhysicalLogicalTypeSupportMatrix)
{
  auto const expected_physical = cudf::io::parquet::Type::INT32;
  auto const expected_encoding = cudf::io::parquet::Encoding::FASTLANES_BITPACK;

  std::vector<int8_t> i8_values(2050);
  std::vector<uint8_t> u8_values(2050);
  std::vector<int16_t> i16_values(2050);
  std::vector<uint16_t> u16_values(2050);
  std::vector<int32_t> i32_values(2050);
  std::vector<uint32_t> u32_values(2050);
  std::vector<cudf::timestamp_D::rep> date32_values(2050);
  std::vector<numeric::decimal32> decimal32_values;
  std::vector<cudf::duration_ms::rep> duration_ms_values(2050);
  std::vector<cudf::duration_s::rep> duration_s_values(2050);

  decimal32_values.reserve(2050);

  for (int i = 0; i < 2050; ++i) {
    i8_values[i]         = static_cast<int8_t>((i % 127) - 63);
    u8_values[i]         = static_cast<uint8_t>(i % 251);
    i16_values[i]        = static_cast<int16_t>((i * 11) % 32767 - 16384);
    u16_values[i]        = static_cast<uint16_t>((i * 37) % 65535);
    i32_values[i]        = static_cast<int32_t>((i * 97) - 100000);
    u32_values[i]        = std::numeric_limits<uint32_t>::max() - static_cast<uint32_t>(i * 31);
    date32_values[i]     = static_cast<cudf::timestamp_D::rep>((i % 4000) - 2000);
    duration_ms_values[i] = static_cast<cudf::duration_ms::rep>((i - 1024) * 17);
    duration_s_values[i]  = static_cast<cudf::duration_s::rep>((i % 2048) - 1024);
    decimal32_values.emplace_back((i % 2 == 0) ? (100000 + i * 3) : -(50000 + i * 5),
                                  numeric::scale_type{0});
  }

  write_single_column_typed_expect_physical_and_encoding<int8_t>(
    i8_values,
    "test_fastlanes_matrix_support_i8.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<uint8_t>(
    u8_values,
    "test_fastlanes_matrix_support_u8.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<int16_t>(
    i16_values,
    "test_fastlanes_matrix_support_i16.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<uint16_t>(
    u16_values,
    "test_fastlanes_matrix_support_u16.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<int32_t>(
    i32_values,
    "test_fastlanes_matrix_support_i32.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<uint32_t>(
    u32_values,
    "test_fastlanes_matrix_support_u32.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<cudf::timestamp_D>(
    date32_values,
    "test_fastlanes_matrix_support_date32.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<numeric::decimal32>(
    decimal32_values,
    "test_fastlanes_matrix_support_decimal32.parquet",
    expected_physical,
    expected_encoding,
    true,
    true,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<cudf::duration_ms>(
    duration_ms_values,
    "test_fastlanes_matrix_support_duration_ms.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);

  write_single_column_typed_expect_physical_and_encoding<cudf::duration_s>(
    duration_s_values,
    "test_fastlanes_matrix_support_duration_s.parquet",
    expected_physical,
    expected_encoding,
    true,
    false,
    9,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesInt32PhysicalLogicalTypeUnsupportedMatrix)
{
  auto const expected_physical = cudf::io::parquet::Type::INT32;
  auto const fastlanes_encoding = cudf::io::parquet::Encoding::FASTLANES_BITPACK;

  std::vector<cudf::duration_D::rep> duration_day_values(2050);
  for (int i = 0; i < 2050; ++i) {
    duration_day_values[i] = static_cast<cudf::duration_D::rep>((i % 4000) - 2000);
  }

  // duration_D writes as INT32 physical with TIME_MILLIS annotation, but is intentionally
  // not in the current FastLanes logical allowlist.
  write_single_column_typed_expect_physical_and_encoding<cudf::duration_D>(
    duration_day_values,
    "test_fastlanes_matrix_unsupported_duration_day.parquet",
    expected_physical,
    fastlanes_encoding,
    false,
    false,
    9,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationMicrosecondsLogicalTypeFallsBackFromFastLanes)
{
  std::vector<cudf::duration_us::rep> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<cudf::duration_us::rep>(i * 1000);
  }

  roundtrip_single_column_typed_expect_not_encoding<cudf::duration_us>(
    values,
    "test_fastlanes_duration_us_fallback_roundtrip.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest,
       FastLanesDurationMicrosecondsNegativeLogicalTypeFallsBackFromFastLanes)
{
  std::vector<cudf::duration_us::rep> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    auto const shifted = static_cast<int64_t>(i % 2048) - 1024;
    values[i]          = static_cast<cudf::duration_us::rep>(shifted * 1000);
  }
  values[0]            = static_cast<cudf::duration_us::rep>(-1);
  values[1]            = static_cast<cudf::duration_us::rep>(0);
  values[2]            = static_cast<cudf::duration_us::rep>(1);
  values[values.size() - 1] = static_cast<cudf::duration_us::rep>(-86400000000LL);

  roundtrip_single_column_typed_expect_not_encoding<cudf::duration_us>(
    values,
    "test_fastlanes_duration_us_negative_fallback_roundtrip.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationNanosecondsLogicalTypeFallsBackFromFastLanes)
{
  std::vector<cudf::duration_ns::rep> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<cudf::duration_ns::rep>(i * 1000000);
  }

  roundtrip_single_column_typed_expect_not_encoding<cudf::duration_ns>(
    values,
    "test_fastlanes_duration_ns_fallback_roundtrip.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesDurationNanosecondsNegativeLogicalTypeFallsBackFromFastLanes)
{
  std::vector<cudf::duration_ns::rep> values(4099);
  for (size_t i = 0; i < values.size(); ++i) {
    auto const shifted = static_cast<int64_t>(i % 2048) - 1024;
    values[i]          = static_cast<cudf::duration_ns::rep>(shifted * 1000000);
  }
  values[0]            = static_cast<cudf::duration_ns::rep>(-1);
  values[1]            = static_cast<cudf::duration_ns::rep>(0);
  values[2]            = static_cast<cudf::duration_ns::rep>(1);
  values[values.size() - 1] = static_cast<cudf::duration_ns::rep>(-86400000000000LL);

  roundtrip_single_column_typed_expect_not_encoding<cudf::duration_ns>(
    values,
    "test_fastlanes_duration_ns_negative_fallback_roundtrip.parquet",
    cudf::io::parquet::Encoding::FASTLANES_BITPACK,
    1024,
    4096);
}

TEST_F(ParquetCpuEncoderTest, FastLanesMixedEncodingsLowCardinalityInt32Pattern)
{
  // Mirrors real-world behavior where one low-cardinality INT32 column works while
  // others with slightly different cardinalities or distributions fail in mixed mode.
  constexpr int num_rows = 4096 * 3 + 17;

  std::vector<int64_t> l_orderkey(num_rows);
  std::vector<int64_t> l_partkey(num_rows);
  std::vector<int32_t> l_returnflag(num_rows);
  std::vector<int32_t> l_linestatus(num_rows);
  std::vector<int32_t> l_shipinstruct(num_rows);
  std::vector<int32_t> l_shipmode(num_rows);
  std::vector<cudf::timestamp_D::rep> l_shipdate(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    l_orderkey[i] = static_cast<int64_t>(i) * 37;
    l_partkey[i]  = static_cast<int64_t>((i * 11) % 1000000);

    // Similar cardinality pattern as the failing mixed-column dataset.
    l_returnflag[i]  = i % 3;
    l_linestatus[i]  = i % 2;
    l_shipinstruct[i] = i % 4;
    l_shipmode[i]    = i % 7;

    l_shipdate[i] = 18500 + (i % 2000);
  }

  cudf::test::fixed_width_column_wrapper<int64_t> col_orderkey(l_orderkey.begin(),
                                                                l_orderkey.end());
  cudf::test::fixed_width_column_wrapper<int64_t> col_partkey(l_partkey.begin(), l_partkey.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_returnflag(l_returnflag.begin(),
                                                                  l_returnflag.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_linestatus(l_linestatus.begin(),
                                                                  l_linestatus.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_shipinstruct(l_shipinstruct.begin(),
                                                                    l_shipinstruct.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col_shipmode(l_shipmode.begin(), l_shipmode.end());
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> col_shipdate(
    l_shipdate.begin(), l_shipdate.end());

  cudf::table_view input({col_orderkey,
                          col_partkey,
                          col_returnflag,
                          col_linestatus,
                          col_shipdate,
                          col_shipinstruct,
                          col_shipmode});

  cudf::io::table_input_metadata metadata(input);
  metadata.column_metadata[0].set_name("l_orderkey");
  metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::DELTA_BINARY_PACKED);
  metadata.column_metadata[1].set_name("l_partkey");
  metadata.column_metadata[1].set_encoding(cudf::io::column_encoding::DELTA_BINARY_PACKED);
  metadata.column_metadata[2].set_name("l_returnflag");
  metadata.column_metadata[2].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
  metadata.column_metadata[3].set_name("l_linestatus");
  metadata.column_metadata[3].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
  metadata.column_metadata[4].set_name("l_shipdate");
  metadata.column_metadata[4].set_encoding(cudf::io::column_encoding::DICTIONARY);
  metadata.column_metadata[5].set_name("l_shipinstruct");
  metadata.column_metadata[5].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);
  metadata.column_metadata[6].set_name("l_shipmode");
  metadata.column_metadata[6].set_encoding(cudf::io::column_encoding::FASTLANES_BITPACK);

  roundtrip_with_metadata(
    input, metadata, "test_fastlanes_mixed_low_cardinality_pattern.parquet", 1024, 4096);
}
