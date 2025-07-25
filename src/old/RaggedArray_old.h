#pragma once

#include "omp.h"
#include <cstddef>
#include <tuple>
#include <vector>

namespace seqr {

// N-dimensional CSR/JaggedIndexer
// made for fast sharding and batching per level
struct RaggedIndexer {
  std::vector<std::vector<size_t>> level_indices;
  size_t current_level;
  const size_t num_dims;

  RaggedIndexer(size_t num_dims);

  // equivalent to openning and closing a bracket
  void open();
  void close();

  // return the number of entries per level (not items)
  size_t level_size(size_t d) const;

  void reset();
  // add number of item to the last layer
  // only work if current_level == num_dims - 1
  void add(size_t n);

  // only for upper level to lower level (tarting idx)
  size_t map_idx(size_t idx, size_t from_level, size_t to_level) const;

  size_t map_idx_rev(size_t idx, size_t from_level, size_t to_level) const;

  // map a span (start, size) at level from_level to level to_level
  // TODO: this won't check the input
  std::tuple<size_t, size_t> map_span(size_t from_level, size_t start_, size_t size_, size_t to_level) const;

  // divide the (start_, size_) range evenly into n_workers (only the worker_id th shard is returned)
  // the size is depend on the range mapped to the size_level
  // batching also use this function
  std::tuple<size_t, size_t> get_shard(size_t start_, size_t size_, size_t level_, size_t size_level, size_t n_workers, size_t worker_id) const;

  // loop over shards (of level level_) with optional multithreading,
  // n_shard will be ignored if use openMP and use_nthread == true,
  // divide the shards base on size_level.
  template <bool use_openMP = false, bool use_nthread = true, typename Func>
  void shard_loop(size_t start_, size_t size_, size_t level_, size_t size_level, Func f, size_t n_shard = 1) const;

  template <bool use_openMP = false, typename Func>
  void batch_loop(size_t start_, size_t size_, size_t level_, size_t size_level, Func f, size_t batch_size_) const;

  // this actually loop through the sub_level_
  // with the span mapped from (start, size) of level_,
  // and give f two indices: f(level_i, sublevel_i)
  template <typename Func>
  void sublevel_loop(size_t start_, size_t size_, size_t level_, size_t sub_level_, Func f) const;
};

// storing vectors in ragged format
struct RaggedVectors {
  const size_t d;
  std::vector<float> owned;
  float *ptr = nullptr;

  RaggedIndexer I;

  RaggedVectors(size_t vd, size_t seq_d);

  // this add n*d float into flatten_vecs
  // the adding of the vector and shape info is seperated
  // use I.open(), I.add() and I.close() to shape it
  void add(const float *xs, size_t n);

  // check if index I fit the vector
  bool ready() const;

  void reset();

  const float *data() const;
};
} // namespace seqr

namespace seqr {

template <bool use_openMP, bool use_nthread, typename Func>
void RaggedIndexer::shard_loop(size_t start_, size_t size_, size_t level_, size_t size_level, Func f, size_t n_shard) const {

  if constexpr (use_openMP) {
    if constexpr (use_nthread) {
#pragma omp parallel
      {
        const size_t num_threads = omp_get_num_threads(), thread_id = omp_get_thread_num();
        auto [shard_start, shard_size] = get_shard(start_, size_, level_, size_level, num_threads, thread_id);
        if (shard_size != 0)
          f(shard_start, shard_size);
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n_shard; i++) {
        auto [shard_start, shard_size] = get_shard(start_, size_, level_, size_level, n_shard, i);
        if (shard_size != 0)
          f(shard_start, shard_size);
      }
    }
  } else {
    for (size_t i = 0; i < n_shard; i++) {
      auto [shard_start, shard_size] = get_shard(start_, size_, level_, size_level, n_shard, i);
      if (shard_size != 0)
        f(shard_start, shard_size);
    }
  }
}

template <bool use_openMP, typename Func>
void RaggedIndexer::batch_loop(size_t start_, size_t size_, size_t level_, size_t size_level, Func f, size_t batch_size_) const {
  // map the span to size_level to get the total size
  auto [f_start_, f_size_] = map_span(level_, start_, size_, size_level);

  auto n_shard = (f_size_ + batch_size_ - 1) / batch_size_;

  // use custom n_shard
  shard_loop<use_openMP, false>(start_, size_, level_, size_level, f, n_shard);
}

template <typename Func>
void RaggedIndexer::sublevel_loop(size_t start_, size_t size_, size_t level_, size_t sub_level_, Func f) const {
  if (level_ == sub_level_) {
    // identity map
    for (size_t i = start_; i < start_ + size_; i++) {
      f(i, i, true);
    }
  } else {
    size_t sub_i = map_idx(start_, level_, sub_level_), next_sub_start;

    bool is_first = true;

    for (size_t i = start_; i < start_ + size_; i++) {
      next_sub_start = map_idx(i + 1, level_, sub_level_);

      is_first = true;
      while (sub_i < next_sub_start) {
        f(i, sub_i, is_first);
        is_first = false;
        sub_i++;
      }
    }
  }
}
}; // namespace seqr