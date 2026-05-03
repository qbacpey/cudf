#include "encode_common.hpp"

#include <cstring>

namespace cudf::io::parquet::detail::fastlanes::detail {

EncodedPageResult create_empty_result(Encoding encoding, rmm::cuda_stream_view stream)
{
  EncodedPageResult result;
  result.bitwidth       = 1;
  result.cast_mode      = ::fastlanes::TypeCastMode::SIGNED_SAFE;
  result.original_count = 0;
  result.padded_count   = 0;
  result.body_size      = 0;
  result.min_value      = 0;

  switch (encoding) {
    case Encoding::FASTLANE_BITPACK_RAW:
      result.host_blob = ::fastlanes::PageHeader::serialize_scalar32(
        1, 0, 0, 0, nullptr, 0, ::fastlanes::default_pre_delta_for_raw32());
      break;
    case Encoding::FASTLANE_BITPACK_SPLIT64:
      result.host_blob = ::fastlanes::PageHeader::serialize_split32(
        1, 1, 0, 0, 0, 0, nullptr, 0, ::fastlanes::default_pre_delta_for_split64());
      break;
    case Encoding::FASTLANES_DELTA_BINARY:
      result.host_blob = ::fastlanes::PageHeader::serialize_native64(
        1, 0, 0, 0, nullptr, 0, ::fastlanes::default_pre_delta_for_native64());
      break;
    default:
      throw std::invalid_argument("FastLanesEncoder: unsupported encoding for empty-page layout");
  }

  result.device_blob = rmm::device_buffer(result.host_blob.size(), stream);
  cuda_check(cudaMemcpyAsync(result.device_blob.data(),
                             result.host_blob.data(),
                             result.host_blob.size(),
                             cudaMemcpyHostToDevice,
                             stream.value()),
             "upload empty blob");

  return result;
}

EncodedPageResult create_empty_scalar32_result(rmm::cuda_stream_view stream)
{
  return create_empty_result(Encoding::FASTLANE_BITPACK_RAW, stream);
}

EncodedPageResult create_empty_split32_result(rmm::cuda_stream_view stream)
{
  return create_empty_result(Encoding::FASTLANE_BITPACK_SPLIT64, stream);
}

EncodedPageResult create_empty_native64_result(rmm::cuda_stream_view stream)
{
  return create_empty_result(Encoding::FASTLANES_DELTA_BINARY, stream);
}

}  // namespace cudf::io::parquet::detail::fastlanes::detail

namespace cudf::io::parquet::detail::fastlanes {

FastLanesInt32Encoder::FastLanesInt32Encoder() {}
FastLanesInt64Split32Encoder::FastLanesInt64Split32Encoder() {}
FastLanesInt64NativeEncoder::FastLanesInt64NativeEncoder() {}

}  // namespace cudf::io::parquet::detail::fastlanes
