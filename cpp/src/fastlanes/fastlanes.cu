#include <cudf/fastlanes/fastlanes_encode.cuh>

// This file ensures the CUDA kernels are compiled.
// The decode_kernel is defined in the header as __global__,
// so this file just needs to include it to trigger compilation.

namespace cudf::io::parquet::detail::fastlanes_cudf {

// Explicit instantiation if needed in the future
// Currently, the inline functions in the header handle everything

}  // namespace cudf::io::parquet::detail::fastlanes_cudf