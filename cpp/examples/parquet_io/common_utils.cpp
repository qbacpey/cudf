/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>

/**
 * @file common_utils.cpp
 * @brief Definitions for common utilities for `parquet_io` examples
 *
 */

cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                         rmm::percent_of_free_device_memory(80)};
  }
  return rmm::mr::cuda_async_memory_resource{};
}

std::shared_ptr<rmm::mr::device_memory_resource> create_managed_memory_resource(bool is_pool_used)
{
  std::cout << "Using managed memory resource\n";
  auto managed_mr = std::make_shared<rmm::mr::managed_memory_resource>();
  if (is_pool_used) {
    // Maximum pool size set to 500 GB
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      managed_mr, rmm::percent_of_free_device_memory(100), 700ULL * 1024 * 1024 * 1024);
  }
  return managed_mr;
}

cudf::io::column_encoding get_encoding_type(std::string name)
{
  using encoding_type = cudf::io::column_encoding;

  static std::unordered_map<std::string_view, encoding_type> const map = {
    {"DEFAULT", encoding_type::USE_DEFAULT},
    {"DICTIONARY", encoding_type::DICTIONARY},
    {"PLAIN", encoding_type::PLAIN},
    {"DELTA_BINARY_PACKED", encoding_type::DELTA_BINARY_PACKED},
    {"DELTA_LENGTH_BYTE_ARRAY", encoding_type::DELTA_LENGTH_BYTE_ARRAY},
    {"DELTA_BYTE_ARRAY", encoding_type::DELTA_BYTE_ARRAY},
    {"BYTE_STREAM_SPLIT", encoding_type::BYTE_STREAM_SPLIT},
    {"DIRECT", encoding_type::DIRECT},
    {"DIRECT_V2", encoding_type::DIRECT_V2},
    {"DICTIONARY_V2", encoding_type::DICTIONARY_V2},
    {"FASTLANE_BITPACK_RAW", encoding_type::FASTLANE_BITPACK_RAW},
    {"FASTLANES_DELTA_BINARY", encoding_type::FASTLANES_DELTA_BINARY},
    {"FASTLANE_BITPACK_SPLIT64", encoding_type::FASTLANE_BITPACK_SPLIT64}
  };

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument(name +
                              " is not a valid encoding type.\n\n"
                              "Available encoding types: DEFAULT, DICTIONARY, PLAIN,\n"
                              "DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,\n"
                              "DELTA_BYTE_ARRAY\n\n");
}

std::string get_encoding_string(cudf::io::column_encoding encoding)
{
  switch (encoding) {
    case cudf::io::column_encoding::USE_DEFAULT: return "DEFAULT";
    case cudf::io::column_encoding::DICTIONARY: return "DICTIONARY";
    case cudf::io::column_encoding::PLAIN: return "PLAIN";
    case cudf::io::column_encoding::DELTA_BINARY_PACKED: return "DELTA_BINARY_PACKED";
    case cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY: return "DELTA_LENGTH_BYTE_ARRAY";
    case cudf::io::column_encoding::DELTA_BYTE_ARRAY: return "DELTA_BYTE_ARRAY";
    case cudf::io::column_encoding::BYTE_STREAM_SPLIT: return "BYTE_STREAM_SPLIT";
    case cudf::io::column_encoding::DIRECT: return "DIRECT";
    case cudf::io::column_encoding::DIRECT_V2: return "DIRECT_V2";
    case cudf::io::column_encoding::DICTIONARY_V2: return "DICTIONARY_V2";
    case cudf::io::column_encoding::FASTLANE_BITPACK_RAW: return "FASTLANE_BITPACK_RAW";
    case cudf::io::column_encoding::FASTLANES_DELTA_BINARY: return "FASTLANES_DELTA_BINARY";
    case cudf::io::column_encoding::FASTLANE_BITPACK_SPLIT64: return "FASTLANE_BITPACK_SPLIT64";
    default: return "UNKNOWN_ENCODING";
  }
}

cudf::io::compression_type get_compression_type(std::string name)
{
  using compression_type = cudf::io::compression_type;

  static std::unordered_map<std::string_view, compression_type> const map = {
    {"NONE", compression_type::NONE},
    {"AUTO", compression_type::AUTO},
    {"SNAPPY", compression_type::SNAPPY},
    {"LZ4", compression_type::LZ4},
    // {"CASCADED", compression_type::CASCADED},
    // {"DEFLATE", compression_type::DEFLATE},
    // {"GDEFLATE", compression_type::GDEFLATE},
    // {"ANS", compression_type::ANS},
    // {"BITCOMP", compression_type::BITCOMP},
    {"ZSTD", compression_type::ZSTD}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument(name +
                              " is not a valid compression type.\n\n"
                              "Available compression types: NONE, AUTO, SNAPPY,\n"
                              "LZ4, ZSTD, CASCADED, BITCOMP, GDEFLATE, ANS\n\n");
}

bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);

  // Check if the input string matches to any of the following
  return input == "ON" or input == "TRUE" or input == "YES" or input == "Y" or input == "T";
}

void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        rmm::cuda_stream_view stream)
{
  auto const tables_equal =
    cudf::tables_equal(lhs_table, rhs_table, cudf::null_equality::EQUAL, stream);
  std::cout << "Tables identical: " << std::boolalpha << tables_equal << "\n\n";
  if (not tables_equal) { throw std::logic_error("Table equality check failed"); }
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}

std::string current_date_and_time()
{
  auto const time       = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto const local_time = *std::localtime(&time);
  // Stringstream to format the date and time
  std::stringstream ss;
  ss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");
  return ss.str();
}

std::map<std::string, cudf::io::column_encoding> parse_column_encodings(std::string const& arg)
{
  std::map<std::string, cudf::io::column_encoding> encoding_map;

  // If no colon found, return empty map to indicate global encoding mode
  if (arg.find(':') == std::string::npos) { return {}; }

  std::stringstream ss(arg);
  std::string segment;

  while (std::getline(ss, segment, ',')) {
    // Trim whitespace
    segment.erase(0, segment.find_first_not_of(" \t\n\r"));
    segment.erase(segment.find_last_not_of(" \t\n\r") + 1);

    if (segment.empty()) { continue; }

    auto delimiter_pos = segment.find(':');
    if (delimiter_pos == std::string::npos) {
      throw std::runtime_error("Parse Error: Segment '" + segment + "' missing ':' separator.");
    }

    std::string col_name = segment.substr(0, delimiter_pos);
    std::string enc_str  = segment.substr(delimiter_pos + 1);

    // Trim individual parts
    col_name.erase(0, col_name.find_first_not_of(" \t\n\r"));
    col_name.erase(col_name.find_last_not_of(" \t\n\r") + 1);
    enc_str.erase(0, enc_str.find_first_not_of(" \t\n\r"));
    enc_str.erase(enc_str.find_last_not_of(" \t\n\r") + 1);

    if (col_name.empty() || enc_str.empty()) {
      throw std::runtime_error("Parse Error: Empty column name or encoding in '" + segment + "'");
    }

    encoding_map[col_name] = get_encoding_type(enc_str);
  }

  return encoding_map;
}

void apply_encodings(cudf::io::table_input_metadata& table_metadata,
                     std::map<std::string, cudf::io::column_encoding> const& encoding_map,
                     cudf::io::column_encoding default_encoding,
                     bool verbose)
{
  for (auto& col_meta : table_metadata.column_metadata) {
    std::string col_name = col_meta.get_name();
    cudf::io::column_encoding selected_encoding;

    if (encoding_map.empty()) {
      // Mode A: No map provided, use default for all
      selected_encoding = default_encoding;
    } else {
      // Mode B: Map provided, STRICT check
      auto it = encoding_map.find(col_name);
      if (it == encoding_map.end()) {
        throw std::runtime_error("Error: No encoding specified for column '" + col_name + "'");
      }
      selected_encoding = it->second;
    }

    col_meta.set_encoding(selected_encoding);

    if (verbose) {
      std::cout << "  Column '" << col_name << "' -> " << get_encoding_string(selected_encoding)
                << "\n";
    }
  }
}

bool use_managed_memory()
{
  auto const* env = std::getenv("LIBCUDF_USE_MANAGED_MEMORY");
  if (env == nullptr) { return false; }
  std::string val{env};
  std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return val == "1" or val == "ON" or val == "TRUE";
}

std::shared_ptr<rmm::mr::device_memory_resource> init_memory_resource(bool is_pool_used)
{
  if (use_managed_memory()) {
    return create_managed_memory_resource(is_pool_used);
  }
  return create_memory_resource(is_pool_used);
}