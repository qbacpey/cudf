/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <string>
#include <map>

/**
 * @file common_utils.hpp
 * @brief Common utilities for `parquet_io` examples
 *
 */

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used);

/**
 * @brief Get encoding type from the keyword
 *
 * @param name encoding keyword name
 * @return corresponding column encoding type
 */
[[nodiscard]] cudf::io::column_encoding get_encoding_type(std::string name);

/**
 * @brief Get encoding string from the encoding type
 *
 * @param encoding column encoding type
 * @return corresponding encoding string
 */
[[nodiscard]] std::string get_encoding_string(cudf::io::column_encoding encoding);

/**
 * @brief Get compression type from the keyword
 *
 * @param name compression keyword name
 * @return corresponding compression type
 */
[[nodiscard]] cudf::io::compression_type get_compression_type(std::string name);

/**
 * @brief Get boolean from they keyword
 *
 * @param input keyword affirmation string such as: Y, T, YES, TRUE, ON
 * @return true or false
 */
[[nodiscard]] bool get_boolean(std::string input);

/**
 * @brief Check if two tables are identical, throw an error otherwise
 *
 * @param lhs_table View to lhs table
 * @param rhs_table View to rhs table
 * @param stream CUDA stream to use
 */
void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream);

/**
 * @brief Returns a string containing current date and time
 *
 */
std::string current_date_and_time();

/**
 * @brief Parse column encoding specification string
 *
 * Parses a string like "col1:ENCODING1,col2:ENCODING2" into a map.
 * If the string contains no ':', returns an empty map to indicate global encoding mode.
 *
 * @param arg Command line argument string
 * @return Map of column name to encoding type (empty if global encoding)
 */
[[nodiscard]] std::map<std::string, cudf::io::column_encoding> parse_column_encodings(
  std::string const& arg);

/**
 * @brief Apply encodings to table input metadata
 *
 * If encoding_map is provided (non-empty), strictly applies per-column encoding.
 * If encoding_map is empty, applies default_encoding to all columns.
 *
 * @param table_metadata Table input metadata to modify
 * @param encoding_map Per-column encoding map (empty for global encoding)
 * @param default_encoding Default encoding if map is empty
 * @param verbose Print encoding assignments if true
 */
void apply_encodings(cudf::io::table_input_metadata& table_metadata,
                     std::map<std::string, cudf::io::column_encoding> const& encoding_map,
                     cudf::io::column_encoding default_encoding,
                     bool verbose = false);

/**
 * @brief Check environment variable for managed memory usage
 *
 * @return true if LIBCUDF_USE_MANAGED_MEMORY is set to 1/ON/TRUE
 */
[[nodiscard]] bool use_managed_memory();

/**
 * @brief Initialize memory resource based on environment and configuration
 *
 * Checks LIBCUDF_USE_MANAGED_MEMORY environment variable to decide
 * between device and managed memory.
 *
 * @param is_pool_used Whether to use memory pool
 * @return Shared pointer to memory resource
 */
[[nodiscard]] std::shared_ptr<rmm::mr::device_memory_resource> init_memory_resource(
  bool is_pool_used = true);
