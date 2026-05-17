/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <iostream>

/**
 * @brief Light-weight timer for measuring elapsed time.
 *
 * A timer object constructed from std::chrono, instrumenting at microseconds
 * precision. Can display elapsed durations at milli and micro second
 * scales. The timer starts at object construction.
 */
class timer {
 public:
  using micros = std::chrono::microseconds;
  using millis = std::chrono::milliseconds;

  timer() { reset(); }
  void reset() { start_time = std::chrono::high_resolution_clock::now(); }
  auto elapsed() const { return (std::chrono::high_resolution_clock::now() - start_time); }

  // Return elapsed time in milliseconds as double
  [[nodiscard]] double elapsed_millis() const
  {
    return std::chrono::duration<double, std::milli>(elapsed()).count();
  }

  // Return elapsed time in microseconds as double
  [[nodiscard]] double elapsed_micros() const
  {
    return std::chrono::duration<double, std::micro>(elapsed()).count();
  }

  // Return elapsed time in seconds as double
  [[nodiscard]] double elapsed_seconds() const { return std::chrono::duration<double>(elapsed()).count(); }

  void print_elapsed_micros() const
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<micros>(elapsed()).count()
              << "us\n\n";
  }
  void print_elapsed_millis() const
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<millis>(elapsed()).count()
              << "ms\n\n";
  }

 private:
  using time_point_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_t start_time;
};
