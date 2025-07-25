#include "utils.h"
#include "omp.h"
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <tuple>

namespace seqr {

std::tuple<size_t, size_t> get_shards_range(size_t total, size_t n_workers, size_t worker_id) {
  // first, evenly distribute the job
  size_t base_batch_size = total / n_workers;
  size_t remainder = total % n_workers;

  size_t start_idx = worker_id * base_batch_size + std::min(worker_id, remainder);

  size_t end_idx = start_idx + base_batch_size + (worker_id < remainder ? 1 : 0) - 1;

  return std::make_tuple(start_idx, end_idx);
}

std::tuple<size_t, size_t> get_shard_range(size_t start_, size_t size_, size_t n_workers, size_t worker_id) {
  // evenly distribute the job
  size_t base_batch_size = size_ / n_workers;
  size_t remainder = size_ % n_workers;

  size_t start_idx = worker_id * base_batch_size + std::min(worker_id, remainder) + start_;

  size_t shard_size_ = base_batch_size + (worker_id < remainder ? 1 : 0) - 1;

  return std::make_tuple(start_idx, shard_size_);
}

std::vector<std::tuple<size_t, size_t>> split_jobs(size_t start_, size_t size_, const SeqInfos &job_info, size_t batch_size, size_t num_batches_limit) {
  // start job and batch_size
  std::vector<std::tuple<size_t, size_t>> batches;

  // curr_batch_sum store the total number of item in selected jobs
  // curr_batch_size is number of jobs
  // curr_start is the first job
  size_t curr_start = start_, curr_batch_sum = 0, curr_batch_size = 0;

  for (size_t i = start_; i < start_ + size_; i++) {
    if (curr_batch_size != 0 && curr_batch_sum + job_info.num_items(i) > batch_size) {
      // create new batch if exceed batch_size
      batches.push_back(std::make_tuple(curr_start, curr_batch_size));

      curr_start = i;
      curr_batch_sum = 0;
      curr_batch_size = 0;
    }

    curr_batch_sum += job_info.num_items(i);
    curr_batch_size += 1;
  }

  if (curr_batch_size != 0) {
    batches.push_back(std::make_tuple(curr_start, curr_batch_size));
  }

  // then merge the last few batch if it exceed limit
  if (num_batches_limit >= 1) {
    while (batches.size() > num_batches_limit) {
      size_t last_bs = std::get<1>(batches.back());
      batches.pop_back();
      std::get<1>(batches.back()) += last_bs;
    }
  }

  return batches;
}

void tranpose(cfloat *A, float *B, const size_t m, const size_t n) {
  // A (m, n)
  // B (n, m)

  //   // simple version
  //   for (size_t i = 0; i < m; i++) {
  //     for (size_t j = 0; j < n; j++) {

  //       // B[j][i] = A[i][j]
  //       B[j * m + i] = A[i * n + j];
  //     }
  //   }

  // this will keep block of B in cache during writing
#define BLOCK_SIZE 32
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m; i += BLOCK_SIZE) {
    for (size_t j = 0; j < n; j += BLOCK_SIZE) {
      // A[i: min(i + BLOCK_SIZE, m), j: min(j + BLOCK_SIZE, n)]
      for (size_t bi = i; bi < std::min(i + BLOCK_SIZE, m); bi++) {
        for (size_t bj = j; bj < std::min(j + BLOCK_SIZE, n); bj++) {

          // B[bj][bi] = A[bi][bj]
          B[bj * m + bi] = A[bi * n + bj];
        }
      }
    }
  }
}

StopWatch::StopWatch() {
  reset();
}

void StopWatch::reset() {
  clock_gettime(CLOCK_MONOTONIC, &start);
}

float StopWatch::split() const {
  struct timespec current;
  clock_gettime(CLOCK_MONOTONIC, &current);

  return (current.tv_sec - start.tv_sec) * 1000.0 + (current.tv_nsec - start.tv_nsec) / 1e6;
}

void rmmm(cfloat *A, cfloat *B, float *C, size_t m, size_t n, size_t k) {
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1,
      A, n,
      B, k,
      0,
      C, k);
}

} // namespace seqr