#pragma once

#include "omp.h"
#include "utils.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace seqr {

// compile time fixed shape ragged indexer
// since we only need 5 dimensions max
template <size_t N>
struct RaggedIndexer {
  const size_t num_dims = N;

  std::array<std::vector<size_t>, N - 1> level_indices;
  size_t final_level_size;
  size_t current_level;

  RaggedIndexer();

  // equivalent to openning and closing a bracket
  void open();
  void close();

  // return the number of entries per level (not items)
  size_t level_size(size_t d) const;

  void reset();
  // add number of item to the last layer
  // only work if current_level == num_dims - 1
  void add(size_t n);

  // only for upper level to lower level (starting idx)
  template <size_t from_level, size_t to_level = N - 1>
  inline size_t map_idx(size_t idx) const;

  //   template<size_t from_level, size_t to_level>
  //   inline size_t map_idx_rev(size_t idx) const;

  template <size_t from_level, size_t to_level = N - 1>
  inline std::pair<size_t, size_t> map_span(size_t start_, size_t size_) const;

  template <size_t level_, size_t size_level_ = N - 1>
  inline std::pair<size_t, size_t> get_shard(size_t start_, size_t size_, size_t n_workers, size_t worker_id) const;

  template <size_t level_, size_t size_level_ = N - 1, bool use_openMP = false, bool use_nthread = true, typename Func>
  inline void balance_shard_loop(size_t start_, size_t size_, Func f, size_t n_shard = 1) const;

  template <size_t level_, size_t size_level_ = N - 1, bool use_openMP = false, typename Func>
  inline void batch_loop(size_t start_, size_t size_, size_t batch_size_, Func f) const;

  // loop over sub_level_ through the span mapped from (start_, size_), giving both indices (of levels), f(i, sub_i)
  template <size_t level_, size_t sub_level_ = N - 1, typename Func>
  inline void sublevel_loop(size_t start_, size_t size_, Func f) const;

  // similar to sublevel_loop but loop over level_ and give the mapped span only, f(i, sub_start, sub_size)
  template <size_t level_, size_t sub_level_ = N - 1, typename Func>
  inline void span_loop(size_t start_, size_t size_, Func f) const;
};

// storing vectors in ragged format
template <size_t N>
struct RaggedVectors {
  const size_t d;
  std::vector<float> owned;
  float *ptr = nullptr;

  RaggedIndexer<N> I;

  RaggedVectors(size_t d);

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

template <size_t N>
template <size_t from_level, size_t to_level>
inline size_t RaggedIndexer<N>::map_idx(size_t idx) const {
  static_assert(from_level <= to_level, "only for upper level to lower level");
  static_assert(to_level < N, "exceed maximum level");

  if constexpr (N == 1)
    return idx;
  else if constexpr (from_level == to_level)
    return idx;
  else
    return map_idx<from_level + 1, to_level>(level_indices[from_level][idx]);
}

template <size_t N>
template <size_t from_level, size_t to_level>
inline std::pair<size_t, size_t> RaggedIndexer<N>::map_span(size_t start_, size_t size_) const {

  if constexpr (from_level == to_level) {
    return std::make_pair(start_, size_);
  } else {
    size_t mapped_start = map_idx<from_level, to_level>(start_);
    size_t next_mapped_start = map_idx<from_level, to_level>(start_ + size_);

    return std::make_pair(mapped_start, next_mapped_start - mapped_start);
  }
}

template <size_t N>
template <size_t level_, size_t size_level_>
inline std::pair<size_t, size_t> RaggedIndexer<N>::get_shard(size_t start_, size_t size_, size_t n_workers, size_t worker_id) const {
  static_assert(level_ <= size_level_, "incorrect levels");

  if constexpr (level_ == size_level_) {
    auto [shard_start, shard_size] = get_shard_range(start_, size_, n_workers, worker_id);
    return std::make_pair(shard_start, shard_size);
  } else {

    auto [id_start_, id_size_] = map_span<level_, size_level_>(start_, size_);
    auto [id_shard_start, id_shard_size] = get_shard_range(id_start_, id_size_, n_workers, worker_id);

    // inclusive
    auto id_shard_end = id_shard_start + id_shard_size;

    auto it_begin = level_indices[level_].begin() + start_, it_end = level_indices[level_].begin() + start_ + size_;

    auto start_shard = std::lower_bound(
                           it_begin,
                           it_end,
                           id_shard_start,
                           [&](size_t candidate, size_t needle) {
                             // the candidate is value of level_indices[level_] so it map level_ + 1
                             return map_idx<level_ + 1, size_level_>(candidate) < needle;
                           }) -
                       level_indices[level_].begin();

    auto end_shard = std::lower_bound(
                         it_begin,
                         it_end,
                         id_shard_end,
                         [&](size_t candidate, size_t needle) {
                           return map_idx<level_ + 1, size_level_>(candidate) < needle;
                         }) -
                     level_indices[level_].begin();

    // just to be sure
    if (worker_id == 0)
      start_shard = start_;

    if (worker_id + 1 == n_workers)
      end_shard = start_ + size_;

    return std::make_pair(start_shard, end_shard - start_shard);
  }
}

template <size_t N>
template <size_t level_, size_t size_level_, bool use_openMP, bool use_nthread, typename Func>
inline void RaggedIndexer<N>::balance_shard_loop(size_t start_, size_t size_, Func f, size_t n_shard) const {

  if constexpr (use_openMP) {
    if constexpr (use_nthread) {
#pragma omp parallel
      {
        const size_t num_threads = omp_get_num_threads(), thread_id = omp_get_thread_num();

        auto [shard_start, shard_size] = get_shard<level_, size_level_>(start_, size_, num_threads, thread_id);
        if (shard_size != 0)
          f(shard_start, shard_size);
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n_shard; i++) {
        auto [shard_start, shard_size] = get_shard<level_, size_level_>(start_, size_, n_shard, i);
        if (shard_size != 0)
          f(shard_start, shard_size);
      }
    }
  } else {
    for (size_t i = 0; i < n_shard; i++) {
      auto [shard_start, shard_size] = get_shard<level_, size_level_>(start_, size_, n_shard, i);
      if (shard_size != 0)
        f(shard_start, shard_size);
    }
  }
}

template <size_t N>
template <size_t level_, size_t size_level_, bool use_openMP, typename Func>
inline void RaggedIndexer<N>::batch_loop(size_t start_, size_t size_, size_t batch_size_, Func f) const {

  // map the span to size_level to get the total size
  auto [f_start_, f_size_] = map_span<level_, size_level_>(start_, size_);

  auto n_shard = (f_size_ + batch_size_ - 1) / batch_size_;

  balance_shard_loop<level_, size_level_, use_openMP, false>(start_, size_, f, n_shard);
}

template <size_t N>
template <size_t level_, size_t sub_level_, typename Func>
inline void RaggedIndexer<N>::sublevel_loop(size_t start_, size_t size_, Func f) const {
  if constexpr (level_ == sub_level_) {
    // identity map
    for (size_t i = start_; i < start_ + size_; i++) {
      f(i, i, true);
    }
  } else {
    size_t sub_i = map_idx<level_, sub_level_>(start_), next_sub_start;
    bool is_first = true;

    for (size_t i = start_; i < start_ + size_; i++) {
      next_sub_start = map_idx<level_, sub_level_>(i + 1);

      is_first = true;
      while (sub_i < next_sub_start) {
        f(i, sub_i, is_first);
        is_first = false;
        sub_i++;
      }
    }
  }
}

template <size_t N>
template <size_t level_, size_t sub_level_, typename Func>
inline void RaggedIndexer<N>::span_loop(size_t start_, size_t size_, Func f) const {
  static_assert(sub_level_ >= level_, "sub_level_ must be >= level_");

  if constexpr (level_ == sub_level_) {
    for (size_t i = start_; i < start_ + size_; i++) {
      f(i, i, 1);
    }
  } else {

    size_t sub_i = 0, next_sub_i = map_idx<level_, sub_level_>(start_);

    for (size_t i = start_; i < start_ + size_; i++) {
      // update next_sub_i
      sub_i = next_sub_i;
      next_sub_i = map_idx<level_, sub_level_>(i + 1);

      f(i, sub_i, next_sub_i - sub_i);
    }
  }
}

template <size_t N>
RaggedIndexer<N>::RaggedIndexer() {
  reset();
}

template <size_t N>
void RaggedIndexer<N>::reset() {
  for (auto &l : level_indices) {
    l.clear();
    l = {0};
  }
  current_level = 0;
  final_level_size = 0;
}

template <size_t N>
void RaggedIndexer<N>::open() {
  if (current_level + 1 >= N)
    throw std::runtime_error("exceed maximum bracket depth");
  // if there is a previous level, extend the last element
  if (current_level > 0) {
    level_indices[current_level - 1].back()++;
  }
  // add a new entry to the current level then move to the next one
  level_indices[current_level].push_back(level_indices[current_level].back());
  current_level++;
}

template <size_t N>
void RaggedIndexer<N>::close() {
  if (current_level <= 0)
    throw std::runtime_error("exceed minimum bracket depth");

  if (current_level == N - 1) {
    const auto &final_l = level_indices.back();

    if (final_l[final_l.size() - 1] == final_l[final_l.size() - 2]) {
      throw std::runtime_error("empty bracket is not allowed");
    }
  }
  // moving back 1 level
  current_level--;
}

template <size_t N>
void RaggedIndexer<N>::add(size_t n) {
  if (current_level + 1 != N)
    throw std::runtime_error("only add items at last level");
  // if we are at the last level, just add to the count
  if constexpr (N > 1)
    level_indices[current_level - 1].back() += n;
  final_level_size += n;
}

template <size_t N>
inline size_t RaggedIndexer<N>::level_size(size_t d) const {
  if (d >= N)
    throw std::runtime_error("exceed maximum level");

  if (d + 1 == N)
    return final_level_size;

  return level_indices[d].size() - 1;
}

template <size_t N>
RaggedVectors<N>::RaggedVectors(size_t d) : d(d) {}

template <size_t N>
void RaggedVectors<N>::add(const float *xs, size_t n) {
  if (ptr != nullptr)
    throw std::runtime_error("can't add items to a non own array");
  if (n == 0)
    return;
  owned.insert(owned.end(), xs, xs + n * d);
}

template <size_t N>
void RaggedVectors<N>::reset() {
  I.reset();
  owned.clear();
}

template <size_t N>
bool RaggedVectors<N>::ready() const {
  // no idea...
  if (ptr != nullptr)
    return true;

  if (I.level_size(I.num_dims - 1) == owned.size() / d)
    return true;

  return false;
}

template <size_t N>
const float *RaggedVectors<N>::data() const {
  if (ptr != nullptr)
    return ptr;
  return owned.data();
}

} // namespace seqr