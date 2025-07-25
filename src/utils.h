#pragma once

#include "cblas.h"
#include "half.hpp"
#include "omp.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ctime>
#include <string>
#include <tuple>
#include <vector>

namespace seqr {

std::pair<size_t, size_t> get_shard_range(size_t start_, size_t size_, size_t n_workers, size_t worker_id);

// divide start_, size into batches of f(shard_start, shard_size)
// openMP is disable by default
template <bool use_openMP = false, typename Func>
inline void batch_loop(size_t start_, size_t size_, size_t batch_size_, Func f);

template <bool use_openMP = false, bool use_nthread = true, typename Func>
inline void shard_loop(size_t start_, size_t size_, Func f, size_t n_shard = 0);

// a simple for loop, expect f(i)
template <bool use_openMP = false, typename Func>
inline void for_loop(size_t start_, size_t size_, Func f);

void tranpose(const float *A, float *B, const size_t m, const size_t n);

// row major matrix multiplication, no tranpose
// A: (m, k)
// B: (k, n)
// C: (m, n)
void rmmm(const float *A, const float *B, float *C, size_t m, size_t n, size_t k);

// Xs is transpose
// Q: (nq, d)
// Xs: (nv, d)
// C: (nq, nv)
void similarity_dot_product(const float *Q, const float *Xs, float *S, const size_t nq, const size_t nv, const size_t d);

// similarity_dot_product but the output is tranposed
// C: (nv, nq)
void similarity_dot_product_trans_out(const float *Q, const float *Xs, float *S, const size_t nq, const size_t nv, const size_t d);

// cache size is number of vector
void similarity_dot_product_trans_out_fp16(const float *Q, const half_float::half *Xs, float *S, const size_t nq, const size_t nv, const size_t d, float *fp32_cache_buffer, size_t cache_size);

void random_fill(float *A, const size_t n, size_t seed = 123);

void random_fill_fp16(half_float::half *A, const size_t n, size_t seed = 123);

// for benchmarking
struct StopWatch {
  struct timespec start;

  StopWatch();

  void reset();

  // return ms since last reset
  float split() const;

  float split_print(const std::string &message, bool reset_ = true);
};

inline void fp16_load(float *dst, const half_float::half *src, const size_t n);

template <typename T>
double estimate_vector_size_GB(std::vector<T> &arr);

} // namespace seqr

namespace seqr {

template <typename T>
double estimate_vector_size_GB(std::vector<T> &arr) {
  const constexpr size_t type_size_bytes = sizeof(T);

  double total_bytes = static_cast<double>(arr.size()) * type_size_bytes;

  double bytes_in_gib = 1024.0 * 1024.0 * 1024.0;

  return total_bytes / bytes_in_gib;
}

inline void fp16_load(float *dst, const half_float::half *src, const size_t n) {
  for (size_t i = 0; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <bool use_openMP, typename Func>
inline void batch_loop(size_t start_, size_t size_, size_t batch_size_, Func f) {
  // inclusive
  size_t end_ = start_ + size_;

  if constexpr (use_openMP) {
#pragma omp parallel for
    for (size_t i = start_; i < end_; i += batch_size_) {
      size_t curr_batch_size = std::min(batch_size_, end_ - i);

      f(i, curr_batch_size);
    }
  } else {
    for (size_t i = start_; i < end_; i += batch_size_) {
      size_t curr_batch_size = std::min(batch_size_, end_ - i);

      f(i, curr_batch_size);
    }
  }
}

template <bool use_openMP, bool use_nthread, typename Func>
inline void shard_loop(size_t start_, size_t size_, Func f, size_t n_shard) {
  if constexpr (use_openMP) {
    if constexpr (use_nthread) {
#pragma omp parallel
      {
        const size_t num_threads = omp_get_num_threads(), thread_id = omp_get_thread_num();
        auto [shard_start, shard_size] = get_shard_range(start_, size_, num_threads, thread_id);

        if (shard_size != 0)
          f(shard_start, shard_size);
      }
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n_shard; i++) {
        auto [shard_start, shard_size] = get_shard_range(start_, size_, n_shard, i);
        if (shard_size != 0)
          f(shard_start, shard_size);
      }
    }
  } else {
    for (size_t i = 0; i < n_shard; i++) {
      auto [shard_start, shard_size] = get_shard_range(start_, size_, n_shard, i);
      if (shard_size != 0)
        f(shard_start, shard_size);
    }
  }
}

template <bool use_openMP, typename Func>
inline void for_loop(size_t start_, size_t size_, Func f) {
  if constexpr (use_openMP) {
#pragma omp parallel for
    for (size_t i = start_; i < start_ + size_; i++) {
      f(i);
    }
  } else {
    for (size_t i = start_; i < start_ + size_; i++) {
      f(i);
    }
  }
}

} // namespace seqr