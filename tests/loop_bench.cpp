#include "utils.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

TEST(loop, singleiter) {
  const size_t N = 2'000'000;
  const size_t tries = 100;

  std::vector<float> data(N, 1.0f); // all ones, just for simplicity

  auto sw = seqr::StopWatch();

  // Warm up
  volatile float warmup = 0;
  for (size_t i = 0; i < N; ++i)
    warmup += data[i];

  std::cout << sw.split() << std::endl;
  sw.reset();

  volatile float sum = 0.0f;

  for (size_t j = 0; j < tries; j++) {
    sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      sum += data[i];
    }
  }
  std::cout << sw.split() << std::endl;
  sw.reset();

  sum = 0.0f;
  volatile size_t loop_count = 1;

  for (size_t j = 0; j < tries; j++) {

    sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      for (size_t t = 0; t < loop_count; t++) {
        sum += data[i + t];
      }
    }
  }
  std::cout << sw.split() << std::endl;
  sw.reset();

  for (size_t j = 0; j < tries; j++) {

    sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      if (loop_count == 1) {
        sum += data[i];

      } else {
        for (size_t t = 0; t < loop_count; t++) {
          sum += data[i + t];
        }
      }
    }
  }
  std::cout << sw.split() << std::endl;
  sw.reset();
  ASSERT_TRUE(true);
}