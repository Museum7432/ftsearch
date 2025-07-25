#include "RaggedArray.h"
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>

TEST(RaggedId, t1) {
  seqr::RaggedIndexer<3> I;

  // Build a known, small ragged structure
  // Let's do 2 top-level chunks, each with 3 and 2 children, respectively
  I.open(); // chunk 0
  I.open();
  I.add(5);
  I.close(); // subchunk 0, 5 elems
  I.open();
  I.add(2);
  I.close(); // subchunk 1, 2 elems
  I.open();
  I.add(3);
  I.close(); // subchunk 2, 3 elems
  I.close();

  I.open(); // chunk 1
  I.open();
  I.add(1);
  I.close(); // subchunk 0, 1 elem
  I.open();
  I.add(4);
  I.close(); // subchunk 1, 4 elems
  I.close();

  for (auto &l : I.level_indices[0]) {
    std::cout << l << " ";
  }
  std::cout << std::endl;
  for (auto &l : I.level_indices[1]) {
    std::cout << l << " ";
  }
  std::cout << std::endl;

  // Now test
  EXPECT_EQ(I.level_size(0), 2);  // 2 chunks at level 0
  EXPECT_EQ(I.level_size(1), 5);  // 3+2=5 subchunks at level 1
  EXPECT_EQ(I.level_size(2), 15); // 5+2+3+1+4=15 elements total

  // Test mapping logic
  // First chunk maps

  EXPECT_EQ((I.map_idx<0, 1>(0)), 0);
  EXPECT_EQ((I.map_idx<0, 1>(1)), 3);

  // Mapping first subchunk of first chunk down to elements
  EXPECT_EQ((I.map_idx<1, 2>(0)), 0);
  // Second subchunk (index 1 at level 1) starts at element 5
  EXPECT_EQ((I.map_idx<1, 2>(1)), 5);

  // // Now reversing: find which subchunk element 6 belongs to (should be subchunk 1)
  // EXPECT_EQ(I.map_idx_rev(6, 2, 1), 1);
  // // find which chunk element 12 belongs to (should be chunk 1)
  // EXPECT_EQ(I.map_idx_rev(12, 2, 0), 1);

  //   EXPECT_EQ(I.map_idx(size_t idx, size_t from_level, size_t to_level), 1);

  ASSERT_TRUE(true);
}

TEST(RaggedId, t2) {

  seqr::RaggedIndexer<4> I;

  size_t t0 = std::rand() % 20;
  for (size_t i0 = 0; i0 < t0; i0++) {
    I.open();
    size_t t1 = std::rand() % 20;
    for (size_t i1 = 0; i1 < t1; i1++) {
      I.open();
      size_t t2 = std::rand() % 20;
      for (size_t i2 = 0; i2 < t2; i2++) {

        I.open();
        size_t t3 = std::rand() % 20 + 1;
        I.add(t3);
        I.close();
      }
      I.close();
    }
    I.close();
  }

  // loop over which dim
  size_t l = 2;
  std::cout << I.level_size(l) ;

  size_t last_start = 1, n_shards = 7;

  for (size_t s = 0; s < n_shards; s++) {
    auto [start_, size_] = I.get_shard<2>(0, I.level_size(l) - 2, n_shards, s);

    auto [mapped_start, mapped_size] = I.map_span<2>(start_, size_);
    std::cout << start_ << "->" << start_ + size_ - 1 << " " << size_ << std::endl;
    std::cout << "\t" << mapped_start << "->" << mapped_start + mapped_size - 1 << " " << mapped_size << std::endl;
  }

  ASSERT_TRUE(true);
}