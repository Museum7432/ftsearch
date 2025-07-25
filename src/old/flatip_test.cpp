#include "FlatIndex.h"
#include "utils.h"
#include <cstddef>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include <sys/time.h>

#include <random>
#include <vector>

TEST(FlatIndex, IP) {

  auto sw = seqr::StopWatch();

  size_t d = 128, n_seq = 100;

  std::random_device rd;                                 // Seed
  std::mt19937 gen(rd());                                // Mersenne Twister RNG
  std::uniform_real_distribution<> value_dis(-1.0, 1.0); // Uniform [-1,1)

  std::uniform_int_distribution<> seq_len_dis(200, 500);

  auto index = seqr::FlatIndexIP(128);

  for (size_t i = 0; i < n_seq; i++) {
    // generate a randome seq
    size_t seq_len = seq_len_dis(gen);
    std::vector<float> rand_seq(seq_len * d);
    for (auto &v : rand_seq) {
      v = value_dis(gen);
    }

    index.add_seq(rand_seq.data(), seq_len);
  }

  EXPECT_EQ(1 + 1, 2);
  ASSERT_TRUE(true);

  std::cout << sw.split();
}
