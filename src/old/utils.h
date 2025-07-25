#pragma once

#include "cblas.h"
#include "omp.h"
#include "types.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ctime>
#include <tuple>
#include <vector>

namespace seqr {

std::tuple<size_t, size_t> get_shards_range(size_t total, size_t n_workers, size_t worker_id);

std::tuple<size_t, size_t> get_shard(size_t start_, size_t size_, size_t n_workers, size_t worker_id);

// split jobs into continous segment evenly (num_batches_limit == 0 will be ignored)
std::vector<std::tuple<size_t, size_t>> split_jobs(size_t start_, size_t size_, const SeqInfos &job_info, size_t batch_size, size_t num_batches_limit = 0);

void tranpose(cfloat *A, float *B, const size_t m, const size_t n);

template <typename Func>
void batch_loop(size_t start_, size_t size_, size_t batch_size_, Func f, bool use_openMP = false);

// batch_loop but for evenly distribute segments
// batch_size_ might be ignored if there is a segment larger than it
template <typename Func>
void constrained_batch_loop(size_t start_seq_, size_t n_seg_, const SeqInfos &info, size_t batch_size_, Func f, bool use_openMP = false);

// start_seq_ and n_seg_ are the start segment and number of segments to loop through
// seg_start_ids is the starting indices of segments
// (it will try to balance the total size of segments in a batch, won't work if it is too unbalanced though)
template <typename Func>
void constrained_shard_loop(size_t start_seq_, size_t n_seg_, const SeqInfos &info, Func f, bool use_openMP = false, size_t n_shard = 0);

/*n_shard = 0 means using n_thread if use_openMP = true*/
template <typename Func>
void shard_loop(size_t start_, size_t size_, Func f, bool use_openMP = false, size_t n_shard = 0);

// for benchmarking
struct StopWatch {
  struct timespec start;

  StopWatch();

  void reset();

  // return ms since last reset
  float split() const;
};

// row major matrix multiplication, no tranpose
void rmmm(cfloat *A, cfloat *B, float *C, size_t m, size_t n, size_t k);

} // namespace seqr

// template implementation
namespace seqr {
template <typename Func>
void batch_loop(size_t start_, size_t size_, size_t batch_size_, Func f, bool use_openMP) {
  // divide size_ into batches

  // inclusive
  size_t end_ = start_ + size_;

  if (use_openMP) {
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

template <typename Func>
void shard_loop(size_t start_, size_t size_, Func f, bool use_openMP, size_t n_shard) {
  if (use_openMP && n_shard >= 2) {
#pragma omp parallel for
    for (size_t i = 0; i < n_shard; i++) {
      auto [shard_start, shard_size] = get_shard(start_, size_, n_shard, i);
      f(shard_start, shard_size);
    }
    return;
  }

  if (use_openMP && n_shard == 0) {
#pragma omp parallel
    {
      const size_t num_threads = omp_get_num_threads(), thread_id = omp_get_thread_num();
      auto [shard_start, shard_size] = get_shard(start_, size_, num_threads, thread_id);
      f(shard_start, shard_size);
    }
    return;
  }

  // don't use threading here
  assert(n_shard >= 1);
  for (size_t i = 0; i < n_shard; i++) {
    auto [shard_start, shard_size] = get_shard(start_, size_, n_shard, i);
    f(shard_start, shard_size);
  }
}

template <typename Func>
void constrained_shard_loop(size_t start_seq_, size_t n_seg_, const SeqInfos &info, Func f, bool use_openMP, size_t n_shard) {
  // get the start and end of the segment
  size_t start_ = info.get_start(start_seq_);
  size_t size_ = info.get_end(start_seq_ + n_seg_ - 1) - start_ + 1;

  if (use_openMP && n_shard == 0) {

    std::vector<std::tuple<size_t, size_t>> job_per_thread;

#pragma omp parallel
    {

// divide the job
#pragma omp master
      {
        size_t num_threads = omp_get_num_threads();

        // estimate the batch_size
        size_t ideal_batch_size = (size_ + num_threads - 1) / num_threads;

        job_per_thread = split_jobs(start_seq_, n_seg_, info, ideal_batch_size, num_threads);
      }
#pragma omp barrier
      // then read the job
      const size_t num_threads = omp_get_num_threads(), thread_id = omp_get_thread_num();

      if (thread_id < job_per_thread.size()) {
        auto [shard_start_seq_, shard_n_seqs] = job_per_thread[thread_id];

        f(shard_start_seq_, shard_n_seqs);
      }
    }

    return;
  }
  size_t ideal_batch_size = (size_ + n_shard - 1) / n_shard;
  std::vector<std::tuple<size_t, size_t>> shards = split_jobs(start_seq_, n_seg_, info, ideal_batch_size, n_shard);

  assert(n_shard >= 1);
  if (use_openMP) {
#pragma omp parallel for
    for (size_t i = 0; i < shards.size(); ++i) {
      auto [shard_start_seq_, shard_n_seqs] = shards[i];
      
      f(shard_start_seq_, shard_n_seqs);
    }
  } else {
    for (size_t i = 0; i < shards.size(); ++i) {
      auto [shard_start_seq_, shard_n_seqs] = shards[i];
      
      f(shard_start_seq_, shard_n_seqs);
    }
  }
}

template <typename Func>
void constrained_batch_loop(size_t start_seq_, size_t n_seg_, const SeqInfos &info, size_t batch_size_, Func f, bool use_openMP) {
  std::vector<std::tuple<size_t, size_t>> shards = split_jobs(start_seq_, n_seg_, info, batch_size_);

  if (use_openMP) {
#pragma omp parallel for
    for (size_t i = 0; i < shards.size(); ++i) {

      auto [shard_start_seq_, shard_n_seqs] = shards[i];

      f(shard_start_seq_, shard_n_seqs);
    }
  } else {
    for (size_t i = 0; i < shards.size(); ++i) {
      auto [shard_start_seq_, shard_n_seqs] = shards[i];

      f(shard_start_seq_, shard_n_seqs);
    }
  }
}

} // namespace seqr