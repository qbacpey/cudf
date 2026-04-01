#pragma once

#ifndef FLS_DEBUG_HPP
#define FLS_DEBUG_HPP

#include <cudf/fastlanes/common.cuh>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <vector>

namespace fastlanes::debug {

// =============================================================================
// Terminal Color Manipulators
// =============================================================================

enum Code : uint32_t {
	FG_BLACK   = 30,
	FG_RED     = 31,
	FG_GREEN   = 32,
	FG_YELLOW  = 33,
	FG_BLUE    = 34,
	FG_MAGENTA = 35,
	FG_CYAN    = 36,
	FG_WHITE   = 37,
	FG_DEFAULT = 39,
	BG_RED     = 41,
	BG_GREEN   = 42,
	BG_BLUE    = 44,
	BG_DEFAULT = 49
};

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& reset(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_BLACK << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& black(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_BLACK << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& bold_black(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[1m\033[" << FG_BLACK << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& bold_blue(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[1m\033[" << FG_BLUE << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& red(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_RED << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& magenta(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_MAGENTA << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& yellow(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_YELLOW << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& def(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_DEFAULT << "m";
}

template <class CHAR_T, class TRAITS>
constexpr std::basic_ostream<CHAR_T, TRAITS>& green(std::basic_ostream<CHAR_T, TRAITS>& os) {
	return os << "\033[" << FG_GREEN << "m";
}

// =============================================================================
// Cast Mode String Helper
// =============================================================================

/**
 * @brief Convert TypeCastMode (as uint8_t) to a human-readable string.
 */
inline const char* cast_mode_str(uint8_t mode)
{
  switch (mode) {
    case 0: return "LEGACY_UNSIGNED_UNSUPPORTED";
    case 1: return "SIGNED_SAFE";
    case 2: return "SIGNED_REINTERPRET";
    default: return "UNKNOWN";
  }
}

inline const char* cast_mode_str(TypeCastMode mode)
{
  return cast_mode_str(static_cast<uint8_t>(mode));
}

inline const char* layout_mode_str(uint8_t mode)
{
  switch (mode) {
    case static_cast<uint8_t>(PageLayoutMode::SCALAR32): return "SCALAR32";
    case static_cast<uint8_t>(PageLayoutMode::SPLIT32): return "SPLIT32";
    case static_cast<uint8_t>(PageLayoutMode::NATIVE64): return "NATIVE64";
    default: return "UNKNOWN";
  }
}

inline const char* bitwidth_mode_str(uint8_t mode)
{
  switch (mode) {
    case static_cast<uint8_t>(BitwidthMode::SINGLE): return "SINGLE";
    case static_cast<uint8_t>(BitwidthMode::SPLIT_COMPONENTS): return "SPLIT_COMPONENTS";
    default: return "UNKNOWN";
  }
}

// =============================================================================
// Debug Print Configuration
// =============================================================================

/**
 * @brief Controls which parts of the FastLanes debug info are printed.
 *
 * Allows fine-grained control separating the page/header boundaries
 * from the full workload and item details.
 */
struct DebugPrintConfig {
  bool enable_header_boundary = false;
  bool enable_workload_detail = false;
  bool enable_vector_boundary = false;
  bool enable_debug_kernel_launch = false;
};

/**
 * @brief Get the global debug print configuration.
 *
 * This singleton provides centralized control of what gets printed.
 * It is initialized from environment variables on the first call, allowing
 * you to change debug toggles without recompiling.
 *
 * Runtime flags (0 = disable, 1 = enable):
 *   export FLS_DEBUG_HEADER=1    # Print page/header boundary debug lines
 *   export FLS_DEBUG_HEADER=0    # Do not print header lines
 *   export FLS_DEBUG_WORKLOAD=1  # Print payload/workload dumps
 *   export FLS_DEBUG_WORKLOAD=0  # Do not print workload dumps
 *   export FLS_DEBUG_VECTOR=1    # Print vector/batch boundary tracking
 *   export FLS_DEBUG_VECTOR=0    # Do not print vector boundary tracking
 *   export FLS_DEBUG_KERNEL=1    # Enable explicit debug-dump kernel launch
 *   export FLS_DEBUG_KERNEL=0    # Decode only (default, no debug kernel)
 *
 * Example (single run, no rebuild):
 *   FLS_DEBUG_KERNEL=1 FLS_DEBUG_HEADER=1 FLS_DEBUG_WORKLOAD=0 ctest --test-dir cpp/build -R PARQUET_CPU_ENCODER_TEST
 */
inline DebugPrintConfig& get_print_config()
{
  static DebugPrintConfig config;
  static bool initialized = false;
  if (!initialized) {
    if (const char* env_p = std::getenv("FLS_DEBUG_HEADER")) {
      config.enable_header_boundary = (env_p[0] == '1');
    }
    if (const char* env_p = std::getenv("FLS_DEBUG_WORKLOAD")) {
      config.enable_workload_detail = (env_p[0] == '1');
    }
    if (const char* env_p = std::getenv("FLS_DEBUG_VECTOR")) {
      config.enable_vector_boundary = (env_p[0] == '1');
    }
    if (const char* env_p = std::getenv("FLS_DEBUG_KERNEL")) {
      config.enable_debug_kernel_launch = (env_p[0] == '1');
    }
    initialized = true;
  }
  return config;
}

/**
 * @brief Check if page/header boundary debug printing is enabled.
 */
inline __host__ __device__ bool is_header_enabled()
{
#if defined(__CUDA_ARCH__)
  return false; // Or provide a device-side toggle mechanism if needed
#else
  return get_print_config().enable_header_boundary;
#endif
}

/**
 * @brief Check if full workload/detail debug printing is enabled.
 */
inline __host__ __device__ bool is_workload_enabled()
{
#if defined(__CUDA_ARCH__)
  return false;
#else
  return get_print_config().enable_workload_detail;
#endif
}

/**
 * @brief Check if vector boundary debug printing is enabled.
 */
inline __host__ __device__ bool is_vector_boundary_enabled()
{
#if defined(__CUDA_ARCH__)
  return false;
#else
  return get_print_config().enable_vector_boundary;
#endif
}

/**
 * @brief Check if host-side FastLanes debug-dump kernel launch is enabled.
 */
inline __host__ __device__ bool is_debug_kernel_enabled()
{
#if defined(__CUDA_ARCH__)
  return false;
#else
  return get_print_config().enable_debug_kernel_launch;
#endif
}

/**
 * @brief Set the debug print configuration centrally.
 *
 * @param headers   Whether to print page/header boundaries.
 * @param workload  Whether to print workload/internal details.
 * @param boundary  Whether to print vector boundary tracking (optional, default true).
 * @param debug_kernel Whether to enable the dedicated debug-dump kernel launch.
 */
inline void set_print_config(bool headers,
                             bool workload,
                             bool boundary = true,
                             bool debug_kernel = false)
{
  get_print_config().enable_header_boundary = headers;
  get_print_config().enable_workload_detail = workload;
  get_print_config().enable_vector_boundary = boundary;
  get_print_config().enable_debug_kernel_launch = debug_kernel;
}

// =============================================================================
// Page Debug Info Struct
// =============================================================================

/**
 * @brief POD struct capturing debug information for a single FastLanes page.
 *
 * Designed to work in two modes:
 * 1. **GPU decode kernel**: Filled on-device by thread 0, memcpy'd to host for printing.
 *    The inline `payload_preview` array captures the first N encoded words.
 * 2. **Host-side encoding/testing**: Filled directly from encoder metadata.
 *    The inline payload preview can be left empty; use `print_encoded_dump()` for
 *    external encoded data buffers instead.
 *
 * When cuDF page metadata is not available (e.g., standalone FastLanes tests),
 * set `has_cudf_info = false` and the cuDF section will be omitted from output.
 */
struct PageDebugInfo {
  // --- FastLanes header fields (always present when page_valid) ---
  uint8_t bitwidth;
  uint8_t cast_mode;  ///< TypeCastMode as uint8_t for device compatibility
  uint8_t layout_mode;
  uint8_t bitwidth_mode;
  uint8_t component_bitwidth_low;
  uint8_t component_bitwidth_high;
  uint32_t original_count;
  uint32_t padded_count;
  uint32_t body_size;
  uint64_t min_value;
  uint32_t min_value_low_bits;
  uint32_t min_value_high_bits;

  // --- cuDF page metadata (optional) ---
  bool has_cudf_info;
  uint32_t num_input_values;
  int32_t first_row;
  int32_t num_rows;
  int32_t dtype_len;
  int32_t dtype_len_in;
  ptrdiff_t data_size;  ///< data_end - data_start
  bool has_repetition;
  int32_t max_nesting_depth;
  uint32_t skipped_leaf_values;

  // --- Validity ---
  bool page_valid;  ///< false if the page was skipped or setup failed

  // --- Inline payload preview (for GPU kernel → host transfer) ---
  static constexpr uint32_t MAX_PAYLOAD_PREVIEW_WORDS = 4096;
  uint32_t payload_preview[MAX_PAYLOAD_PREVIEW_WORDS];
  uint32_t payload_preview_count;  ///< actual words captured (0 = no preview)
};

// =============================================================================
// Factory Functions (Host Only)
// =============================================================================

/**
 * @brief Create a PageDebugInfo from a FastLanes PageHeader (no cuDF metadata).
 */
inline PageDebugInfo make_debug_info(const PageHeader& hdr)
{
  PageDebugInfo info{};
  info.page_valid     = true;
  info.bitwidth       = hdr.bitwidth;
  info.cast_mode      = 0;
  info.layout_mode    = static_cast<uint8_t>(hdr.layout_mode);
  info.bitwidth_mode  = static_cast<uint8_t>(hdr.bitwidth_mode);
  info.component_bitwidth_low  = hdr.component_bitwidth_low;
  info.component_bitwidth_high = hdr.component_bitwidth_high;
  info.original_count = hdr.original_count;
  info.padded_count   = hdr.padded_count;
  info.body_size      = hdr.body_size;
  info.min_value      = hdr.min_value_bits();
  info.min_value_low_bits  = hdr.min_value_low_bits;
  info.min_value_high_bits = hdr.min_value_high_bits;
  info.has_cudf_info  = false;
  info.payload_preview_count = 0;
  return info;
}

/**
 * @brief Create a PageDebugInfo from raw header fields (no cuDF metadata).
 */
inline PageDebugInfo make_debug_info(uint8_t bw,
                                     TypeCastMode mode,
                                     uint32_t orig_count,
                                     uint32_t pad_count,
                                     uint32_t body_sz,
                                     uint64_t min_value)
{
  PageDebugInfo info{};
  info.page_valid     = true;
  info.bitwidth       = bw;
  info.cast_mode      = static_cast<uint8_t>(mode);
  info.layout_mode    = static_cast<uint8_t>(PageLayoutMode::SCALAR32);
  info.bitwidth_mode  = static_cast<uint8_t>(BitwidthMode::SINGLE);
  info.component_bitwidth_low  = bw;
  info.component_bitwidth_high = 0;
  info.original_count = orig_count;
  info.padded_count   = pad_count;
  info.body_size      = body_sz;
  info.min_value      = min_value;
  info.min_value_low_bits  = static_cast<uint32_t>(min_value);
  info.min_value_high_bits = static_cast<uint32_t>(min_value >> 32);
  info.has_cudf_info  = false;
  info.payload_preview_count = 0;
  return info;
}

/**
 * @brief Create a PageDebugInfo from raw header fields AND encoded payload.
 *
 * The encoded payload (of any integer word type) is reinterpreted as uint32_t
 * words and copied into the inline payload_preview buffer (up to
 * MAX_PAYLOAD_PREVIEW_WORDS). This lets print_page_debug() print everything
 * in a single call, without a separate print_encoded_dump().
 *
 * @tparam T  Element type of the encoded buffer (e.g. uint32_t, uint64_t)
 * @param bw           Bitwidth
 * @param mode         TypeCastMode
 * @param orig_count   Original (unpadded) element count
 * @param pad_count    Padded element count
 * @param body_sz      Encoded body size in bytes
 * @param encoded_data Pointer to encoded data
 * @param encoded_elems Number of T-sized elements in encoded_data
 */
template <typename T>
inline PageDebugInfo make_debug_info(uint8_t bw,
                                     TypeCastMode mode,
                                     uint32_t orig_count,
                                     uint32_t pad_count,
                                     uint32_t body_sz,
                                     uint64_t min_value,
                                     const T* encoded_data,
                                     size_t encoded_elems)
{
  PageDebugInfo info = make_debug_info(bw, mode, orig_count, pad_count, body_sz, min_value);

  // Reinterpret encoded data as uint32_t words for the preview buffer
  auto const total_bytes = encoded_elems * sizeof(T);
  auto const total_words = static_cast<uint32_t>(total_bytes / sizeof(uint32_t));
  auto const preview_count =
    total_words < PageDebugInfo::MAX_PAYLOAD_PREVIEW_WORDS
      ? total_words
      : PageDebugInfo::MAX_PAYLOAD_PREVIEW_WORDS;

  auto const* src = reinterpret_cast<const uint32_t*>(encoded_data);
  for (uint32_t i = 0; i < preview_count; ++i) {
    info.payload_preview[i] = src[i];
  }
  info.payload_preview_count = preview_count;

  return info;
}

// =============================================================================
// Printing Functions (Host Only)
// =============================================================================

/**
 * @brief Print the FastLanes page header section.
 */
inline void print_fl_header(std::ostream& os, const PageDebugInfo& info)
{
  auto const restore_fill = os.fill();
  os << "\n--- FastLanes Page Header ---\n"
     << "  bitwidth        : " << static_cast<int>(info.bitwidth) << "\n"
     << "  layout_mode     : " << static_cast<int>(info.layout_mode)
     << " (" << layout_mode_str(info.layout_mode) << ")\n"
     << "  bitwidth_mode   : " << static_cast<int>(info.bitwidth_mode)
     << " (" << bitwidth_mode_str(info.bitwidth_mode) << ")\n"
     << "  bitwidth_lo/hi  : " << static_cast<int>(info.component_bitwidth_low) << "/"
     << static_cast<int>(info.component_bitwidth_high) << "\n"
     << "  original_count  : " << info.original_count << "\n"
     << "  padded_count    : " << info.padded_count << "\n"
     << "  body_size       : " << info.body_size << " bytes\n"
     << "  min_value_raw   : 0x" << std::hex << std::setw(16) << std::setfill('0')
     << info.min_value << std::dec << "\n"
     << "  min_low/min_high: 0x" << std::hex << info.min_value_low_bits << "/0x"
     << info.min_value_high_bits << std::dec << "\n"
     << "  min_value       : ";
  if (info.has_cudf_info && info.dtype_len_in == 4) {
    os << fastlanes::u32_bits_to_int32(static_cast<uint32_t>(info.min_value));
  } else {
    os << fastlanes::u64_bits_to_int64(info.min_value);
  }
  os << "\n";
  os.fill(restore_fill);
}

/**
 * @brief Print the cuDF page metadata section (only if has_cudf_info is true).
 */
inline void print_cudf_metadata(std::ostream& os, const PageDebugInfo& info)
{
  if (!info.has_cudf_info) { return; }
  os << "\n--- cuDF Page Metadata ---\n"
     << "  num_input_values   : " << info.num_input_values << "\n"
     << "  first_row          : " << info.first_row << "\n"
     << "  num_rows           : " << info.num_rows << "\n"
     << "  dtype_len          : " << info.dtype_len << "\n"
     << "  dtype_len_in       : " << info.dtype_len_in << "\n"
     << "  data_size (bytes)  : " << info.data_size << "\n"
     << "  has_repetition     : " << (info.has_repetition ? "true" : "false") << "\n"
     << "  max_nesting_depth  : " << info.max_nesting_depth << "\n"
     << "  skipped_leaf_values: " << info.skipped_leaf_values << "\n";
}

/**
 * @brief Print a hex dump of uint32_t payload words.
 *
 * @param os         Output stream
 * @param words      Pointer to uint32_t words
 * @param count      Number of words
 * @param indent     Indentation prefix per line (default: 2 spaces)
 */
inline void print_payload_hex(std::ostream& os,
                              const uint32_t* words,
                              uint32_t count,
                              const char* indent = "  ")
{
  if (count == 0) { return; }

  os << "\n--- Encoded Payload Preview (" << count << " x 32-bit words) ---\n";

  std::ios state(nullptr);
  state.copyfmt(os);

  os << std::hex << std::setfill('0');
  for (uint32_t i = 0; i < count; ++i) {
    if (i % 8 == 0) { os << "\n" << indent << "[" << std::setw(4) << i << "] "; }
    os << std::setw(8) << words[i] << " ";
  }
  os << "\n";

  os.copyfmt(state);
}

/**
 * @brief Print complete page debug output.
 *
 * Prints: separator → FL header → cuDF metadata (if available) → inline payload preview.
 *
 * @param os           Output stream
 * @param info         Debug info struct
 * @param page_index   Page index for the banner (-1 to omit index)
 */
inline void print_page_debug(std::ostream& os,
                             const PageDebugInfo& info,
                             int page_index = -1)
{
  if (!info.page_valid) { return; }
  if (!is_header_enabled() && !is_workload_enabled()) { return; }

  os << "\n============================================================\n";
  if (page_index >= 0) {
    os << "  FastLanes Debug — Page " << page_index << "\n";
  } else {
    os << "  FastLanes Debug\n";
  }
  os << "============================================================\n";

  if (is_header_enabled()) {
    print_fl_header(os, info);
    print_cudf_metadata(os, info);
  }

  if (is_workload_enabled() && info.payload_preview_count > 0) {
    print_payload_hex(os, info.payload_preview, info.payload_preview_count);
  }

  os << "\n============================================================\n" << std::endl;
}

// =============================================================================
// Encoded Data Dump (Templated — Works for Any Integer Word Size)
// =============================================================================

/**
 * @brief Print a hex dump of an encoded data buffer.
 *
 * This is the centralized version of the `print_encoded_dump` function that was
 * previously duplicated across fastlanes_encode.cuh and test files.
 * Works for uint32_t, int32_t, uint64_t, int64_t, etc.
 *
 * @tparam T     Element type of the encoded buffer
 * @param os     Output stream
 * @param data   Pointer to encoded data
 * @param count  Number of elements
 * @param label  Human-readable label for the dump
 */
template <typename T>
inline void print_encoded_dump(std::ostream& os,
                               const T* data,
                               size_t count,
                               const char* label = "Encoded Dump")
{
  os << ">> " << label << " (" << count << " x " << (sizeof(T) * 8) << "-bit words):" << std::endl;

  std::ios state(nullptr);
  state.copyfmt(os);

  os << std::hex << std::setfill('0');
  for (size_t i = 0; i < count; ++i) {
    if (i % 8 == 0) { os << "\n[" << std::setw(4) << i << "] "; }
    using UnsignedT = std::make_unsigned_t<T>;
    os << std::setw(sizeof(T) * 2) << static_cast<uint64_t>(static_cast<UnsignedT>(data[i]))
       << " ";
  }
  os << "\n" << std::endl;

  os.copyfmt(state);
}

/// Overload taking a std::vector
template <typename T>
inline void print_encoded_dump(std::ostream& os,
                               const std::vector<T>& data,
                               const char* label = "Encoded Dump")
{
  print_encoded_dump(os, data.data(), data.size(), label);
}

/// Convenience overloads defaulting to std::cout
template <typename T>
inline void print_encoded_dump(const T* data, size_t count, const char* label = "Encoded Dump")
{
  print_encoded_dump(std::cout, data, count, label);
}

template <typename T>
inline void print_encoded_dump(const std::vector<T>& data, const char* label = "Encoded Dump")
{
  print_encoded_dump(std::cout, data.data(), data.size(), label);
}

// =============================================================================
// Encoder Debug Summary
// =============================================================================

/**
 * @brief Print a one-line encoder summary (count, padded, bitwidth, mode, first values).
 *
 * Replaces the private `print_debug_info` that was in FastLanesEncoder.
 *
 * @tparam T     Data element type
 * @param os     Output stream
 * @param count  Original element count
 * @param padded Padded element count
 * @param bw     Bitwidth
 * @param mode   TypeCastMode
 * @param data   Pointer to input data (first N values printed)
 * @param n      Number of sample values to print (default 10)
 */
template <typename T>
inline void print_encoder_summary(std::ostream& os,
                                  uint32_t count,
                                  uint64_t padded,
                                  uint8_t bw,
                                  TypeCastMode mode,
                                  const T* data,
                                  size_t n = 10)
{
  os << "[FastLanes Encode] count=" << count << ", padded=" << padded
     << ", bitwidth=" << static_cast<int>(bw) << ", mode=" << cast_mode_str(mode)
     << ", first " << n << ": ";
  for (size_t k = 0; k < std::min<size_t>(count, n); ++k) {
    os << data[k] << " ";
  }
  os << "\n";
}

/// Convenience overload defaulting to std::cout
template <typename T>
inline void print_encoder_summary(uint32_t count,
                                  uint64_t padded,
                                  uint8_t bw,
                                  TypeCastMode mode,
                                  const T* data,
                                  size_t n = 10)
{
  print_encoder_summary(std::cout, count, padded, bw, mode, data, n);
}

}  // namespace fastlanes::debug
#endif  // FLS_DEBUG_HPP

// =============================================================================
// Convenience Macros (Outside namespace, as before)
// =============================================================================

#define FLS_SHOW(a)                                                                                \
	std::cout << fastlanes::debug::yellow << "-- " << #a << ": " << (a)                             \
	          << fastlanes::debug::def << '\n';
#define FLS_LOG(m)                                                                                 \
	std::cout << fastlanes::debug::yellow << "-- " << m << fastlanes::debug::def << '\n';
#define FLS_CERR(a)                                                                                \
	std::cout << fastlanes::debug::red << "-- " << #a << ": " << (a)                                \
	          << fastlanes::debug::def << '\n';
#define FLS_SUCCESS(m)                                                                             \
	std::cout << fastlanes::debug::green << "-- " << m << fastlanes::debug::def << '\n';
#define FLS_RESULT(m)                                                                              \
	std::cout << fastlanes::debug::bold_blue << "-- " << m << fastlanes::debug::def << '\n';

template <typename T>
void PRINT(T* arr, const char* str)
{
	printf("\n ==================   %s   ================= \n ", str);
	for (int ITEM = 0; ITEM < 1024; ++ITEM) {
		if (ITEM % 128 == 0) { printf("\n"); }
		printf(" %d | ", arr[ITEM]);
	}
	printf("\n");
}
