#!/usr/bin/env Rscript

# INT64 FastLanes Page-Level Plot/Report Generator
#
# What this script does:
# 1. Reads case_summary.csv and page_stats_int64_fastlanes.csv.
# 2. Builds human-facing plots for bitwidth/bytes-value/padding/min-drift trends.
# 3. Writes a concise markdown report that references generated figures.
#
# Recommended layered output placement:
# - output_dir: <run_root>/01_human/r_report
#
# Inputs:
# - --case-summary-csv: usually <run_root>/03_raw/case_summary.csv
# - --page-stats-csv: usually <run_root>/03_raw/page_stats_int64_fastlanes.csv
#
# Outputs in --output-dir:
# - bitwidth_distribution.png
# - encoded_bytes_per_value.png
# - padding_overhead.png
# - min_value_drift.png
# - int64_fastlanes_page_report.md
#
# Usage example:
# Rscript ./tools/search/plot_int64_fastlanes_page_stats.R \
#   --case-summary-csv ./artifacts/fastlanes/snappy_int64_page_stats/run_<id>/03_raw/case_summary.csv \
#   --page-stats-csv ./artifacts/fastlanes/snappy_int64_page_stats/run_<id>/03_raw/page_stats_int64_fastlanes.csv \
#   --output-dir ./artifacts/fastlanes/snappy_int64_page_stats/run_<id>/01_human/r_report

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% args || "-h" %in% args) {
    cat(
      "Usage: Rscript plot_int64_fastlanes_page_stats.R ",
      "--case-summary-csv <csv> --page-stats-csv <csv> --output-dir <dir> ",
      "[--report-title <title>]\n",
      sep = ""
    )
    quit(status = 0)
  }

  out <- list(
    case_summary_csv = "",
    page_stats_csv = "",
    output_dir = "",
    report_title = "INT64 FastLanes Page-Level Report"
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--case-summary-csv" && i < length(args)) {
      out$case_summary_csv <- args[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--page-stats-csv" && i < length(args)) {
      out$page_stats_csv <- args[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--output-dir" && i < length(args)) {
      out$output_dir <- args[[i + 1]]
      i <- i + 2
      next
    }
    if (key == "--report-title" && i < length(args)) {
      out$report_title <- args[[i + 1]]
      i <- i + 2
      next
    }
    stop(paste("Unknown or incomplete arg:", key))
  }

  if (out$case_summary_csv == "" || out$page_stats_csv == "" || out$output_dir == "") {
    stop("Usage: Rscript plot_int64_fastlanes_page_stats.R --case-summary-csv <csv> --page-stats-csv <csv> --output-dir <dir> [--report-title <title>]")
  }

  out
}

args <- parse_args()
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

case_summary <- readr::read_csv(args$case_summary_csv, show_col_types = FALSE)
page_stats <- readr::read_csv(args$page_stats_csv, show_col_types = FALSE)

if (nrow(page_stats) == 0) {
  stop("page_stats_csv has no rows. Run extractor first and ensure FLS debug logs were captured.")
}

page_stats <- page_stats %>%
  mutate(
    case_name = factor(case_name, levels = unique(case_name)),
    column_name = factor(column_name, levels = unique(column_name)),
    page_index = as.integer(page_index),
    row_group_index = as.integer(row_group_index),
    component_bitwidth_low = as.integer(component_bitwidth_low),
    component_bitwidth_high = as.integer(component_bitwidth_high),
    encoded_bytes_per_value = as.numeric(encoded_bytes_per_value),
    padding_overhead_pct = as.numeric(padding_overhead_pct)
  )

bitwidth_long <- page_stats %>%
  select(case_name, column_name, row_group_index, page_index, component_bitwidth_low, component_bitwidth_high) %>%
  pivot_longer(
    cols = c(component_bitwidth_low, component_bitwidth_high),
    names_to = "component",
    values_to = "bitwidth"
  ) %>%
  mutate(component = recode(component,
    component_bitwidth_low = "low",
    component_bitwidth_high = "high"
  ))

p_bitwidth <- ggplot(bitwidth_long, aes(x = bitwidth, fill = component)) +
  geom_histogram(binwidth = 1, position = "identity", alpha = 0.55) +
  facet_grid(case_name ~ column_name, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title = "SPLIT32 Component Bitwidth Distribution",
    x = "Bitwidth",
    y = "Page count",
    fill = "Component"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(args$output_dir, "bitwidth_distribution.png"), p_bitwidth, width = 14, height = 8, dpi = 140)

p_bytes <- ggplot(page_stats, aes(x = page_index, y = encoded_bytes_per_value, color = case_name)) +
  geom_line(alpha = 0.5) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.7) +
  facet_wrap(~ column_name, scales = "free_y") +
  scale_y_continuous(labels = label_number(accuracy = 0.01)) +
  labs(
    title = "Encoded Bytes Per Value Across Pages",
    x = "Page index within chunk",
    y = "Encoded bytes/value",
    color = "Case"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(args$output_dir, "encoded_bytes_per_value.png"), p_bytes, width = 14, height = 8, dpi = 140)

p_padding <- ggplot(page_stats, aes(x = column_name, y = padding_overhead_pct, fill = case_name)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(
    title = "Padding Overhead by Column",
    x = "Column",
    y = "Padding overhead (%)",
    fill = "Case"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

ggsave(file.path(args$output_dir, "padding_overhead.png"), p_padding, width = 12, height = 6, dpi = 140)

min_values <- page_stats %>%
  mutate(
    min_lo_num = strtoi(sub("^0x", "", min_value_low_bits_hex), base = 16),
    min_hi_num = strtoi(sub("^0x", "", min_value_high_bits_hex), base = 16)
  ) %>%
  select(case_name, column_name, page_index, min_lo_num, min_hi_num) %>%
  pivot_longer(
    cols = c(min_lo_num, min_hi_num),
    names_to = "component",
    values_to = "min_bits"
  ) %>%
  mutate(component = recode(component, min_lo_num = "low", min_hi_num = "high"))

p_min <- ggplot(min_values, aes(x = page_index, y = min_bits, color = component)) +
  geom_line(alpha = 0.5) +
  facet_grid(case_name ~ column_name, scales = "free_y") +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title = "Page-local Min Value Drift (Raw Bits)",
    x = "Page index within chunk",
    y = "Min value bits",
    color = "Component"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(args$output_dir, "min_value_drift.png"), p_min, width = 14, height = 8, dpi = 140)

case_delta <- case_summary %>%
  transmute(
    case_name,
    status,
    output_size_bytes,
    size_delta_bytes,
    size_delta_pct,
    elapsed_seconds,
    time_delta_seconds,
    time_delta_pct
  )

report_path <- file.path(args$output_dir, "int64_fastlanes_page_report.md")
con <- file(report_path, open = "wt")
writeLines(paste0("# ", args$report_title), con)
writeLines("", con)
writeLines("## Inputs", con)
writeLines(paste0("- case summary: ", args$case_summary_csv), con)
writeLines(paste0("- page stats: ", args$page_stats_csv), con)
writeLines("", con)
writeLines("## Key Plots", con)
writeLines("- bitwidth_distribution.png", con)
writeLines("- encoded_bytes_per_value.png", con)
writeLines("- padding_overhead.png", con)
writeLines("- min_value_drift.png", con)
writeLines("", con)
writeLines("## Case-Level Delta Table", con)
writeLines("", con)
writeLines("| case_name | status | output_size_bytes | size_delta_bytes | size_delta_pct | elapsed_seconds | time_delta_seconds | time_delta_pct |", con)
writeLines("|---|---:|---:|---:|---:|---:|---:|---:|", con)
for (i in seq_len(nrow(case_delta))) {
  r <- case_delta[i, ]
  writeLines(
    paste0(
      "| ", r$case_name,
      " | ", r$status,
      " | ", r$output_size_bytes,
      " | ", r$size_delta_bytes,
      " | ", r$size_delta_pct,
      " | ", round(r$elapsed_seconds, 4),
      " | ", r$time_delta_seconds,
      " | ", r$time_delta_pct,
      " |"
    ),
    con
  )
}
writeLines("", con)
writeLines("## Notes", con)
writeLines("- Page index is the in-chunk FastLanes page ordinal.", con)
writeLines("- Row-group and column are inferred from chunk_id and parquet column count.", con)
writeLines("- Compare case deltas with page-level bitwidth and padding trends to explain SNAPPY outcomes.", con)
close(con)

cat("Wrote report assets to:", args$output_dir, "\n")
cat("-", file.path(args$output_dir, "bitwidth_distribution.png"), "\n")
cat("-", file.path(args$output_dir, "encoded_bytes_per_value.png"), "\n")
cat("-", file.path(args$output_dir, "padding_overhead.png"), "\n")
cat("-", file.path(args$output_dir, "min_value_drift.png"), "\n")
cat("-", report_path, "\n")
