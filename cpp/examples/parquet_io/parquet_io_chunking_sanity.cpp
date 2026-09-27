#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include "common_utils.hpp"
#include "io_source.hpp"
#include "timer.hpp"

#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <iostream>
#include <memory>

/**
 * @brief Process a Parquet file Row Group by Row Group with minimal memory footprint.
 *
 * Pipeline: Read RG -> Process -> Write -> Release Memory -> Next RG
 */
void process_parquet_by_row_group(std::string const& input_file, std::string const& output_file)
{
  auto stream = cudf::get_default_stream();

  bool constexpr is_pool_used = true;
  auto resource               = create_memory_resource(is_pool_used);
  cudf::set_current_device_resource(resource);

  // Step 1: Read metadata to get the number of row groups
  auto const source_info    = cudf::io::source_info(input_file);
  auto const metadata       = cudf::io::read_parquet_metadata(source_info);
  auto const num_row_groups = metadata.num_rowgroups();

  std::cout << "Total Row Groups: " << num_row_groups << std::endl;

  // Step 2: Create chunked writer (this allows appending row groups one by one)
  auto sink_info      = cudf::io::sink_info(output_file);
  auto writer_options = cudf::io::chunked_parquet_writer_options::builder(sink_info);

  // Optional: Configure writer settings
  // writer_options.compression(cudf::io::compression_type::SNAPPY);
  // writer_options.row_group_size_bytes(128 * 1024 * 1024);  // 128 MB row groups
  // writer_options.stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);

  cudf::io::chunked_parquet_writer writer(writer_options.build(), stream);

  // Step 3: Process each Row Group one at a time
  for (int rg_idx = 0; rg_idx < num_row_groups; ++rg_idx) {
    std::cout << "Processing Row Group " << rg_idx << " / " << num_row_groups << std::endl;

    // 3a. Read ONLY this specific row group
    auto read_options = cudf::io::parquet_reader_options::builder(source_info)
                          .row_groups({{rg_idx}})  // Read only row group at index rg_idx
                          .build();

    auto table_with_meta = cudf::io::read_parquet(read_options, stream);

    // 3b. Process the table (your custom logic here)
    // Example: auto processed_table = your_processing_function(table_with_meta.tbl->view());

    // 3c. Write the processed row group to output file
    writer.write(table_with_meta.tbl->view());

    // 3d. Synchronize and release memory
    stream.sync();

    // The unique_ptr (table_with_meta.tbl) goes out of scope here,
    // automatically releasing GPU memory for this row group.

    std::cout << "  Row Group " << rg_idx << " written and memory released." << std::endl;
  }

  // Step 4: Finalize the output file (writes footer)
  writer.close();

  std::cout << "Pipeline complete. Output written to: " << output_file << std::endl;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.parquet> <output.parquet>" << std::endl;
    return 1;
  }

  std::string input_file  = argv[1];
  std::string output_file = argv[2];

  process_parquet_by_row_group(input_file, output_file);

  return 0;
}