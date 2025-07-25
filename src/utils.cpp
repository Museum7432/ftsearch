#include "utils.h"
#include "cblas.h"
#include "omp.h"
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>

namespace seqr {

std::pair<size_t, size_t> get_shard_range(size_t start_, size_t size_, size_t n_workers, size_t worker_id) {
  // evenly distribute the job
  size_t base_batch_size = size_ / n_workers;
  size_t remainder = size_ % n_workers;

  size_t offset = worker_id < remainder ? worker_id : remainder;
  size_t start_idx = worker_id * base_batch_size + offset + start_;

  size_t shard_size_ = base_batch_size + (worker_id < remainder ? 1 : 0);

  return std::make_pair(start_idx, shard_size_);
}

void tranpose(const float *A, float *B, const size_t m, const size_t n) {
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

void rmmm(const float *A, const float *B, float *C, size_t m, size_t n, size_t k) {
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1,
      A, k,
      B, n,
      0,
      C, n);
}

void similarity_dot_product(const float *Q, const float *Xs, float *S, const size_t nq, const size_t nv, const size_t d) {

  // assume that nq is ussually small and nv is always filled.
  if (nq < 15) {
    for_loop(0, nv, [&](size_t x_i) {
      for_loop(0, nq, [&](size_t q_i) {
        S[q_i * nv + x_i] = cblas_sdot(d, Xs + x_i * d, 1, Q + q_i * d, 1);
      });
    });
  } else if (nq < 35) {
    for_loop(0, nv, [&](size_t x_i) {
      cblas_sgemv(
          CblasRowMajor, CblasNoTrans,
          nq, d,
          1, Q, d,
          Xs + x_i * d, 1, 0,
          S + x_i, nv);
    });
  } else {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        nq, nv, d,
        1, Q, d,
        Xs, d,
        0, S, nv);
  }
}

void similarity_dot_product_trans_out(const float *Q, const float *Xs, float *S, const size_t nq, const size_t nv, const size_t d) {
  // Q: (nq, d)
  // Xs: (nv, d)
  // transpose C
  // C: (nv, nq)
  // assume that nq is ussually small and nv is always filled.

  if (nq < 15) {

    // for_loop(0, nv, [&](size_t x_i) {
    //   for_loop(0, nq, [&](size_t q_i) {
    //     S[q_i + x_i * nq] = cblas_sdot(d, Xs + x_i * d, 1, Q + q_i * d, 1);
    //   });
    // });

    const constexpr size_t tile = 4;
    for (size_t ii = 0; ii < nv; ii += tile) {
      for (size_t jj = 0; jj < nq; jj += tile) {

        for (int i = ii; i < std::min(ii + tile, nv); i++) {

          for (int j = jj; j < std::min(jj + tile, nq); j++) {

            float sum = 0;
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k < d; k++)
              sum += Xs[i * d + k] + Q[j * d + k];
            S[j + i * nq] = sum;

            // S[j + i * nq] = cblas_sdot(d, Xs + i * d, 1, Q + j * d, 1);
          }
        }
      }
    }

  } else if (nq < 35) {
    for_loop(0, nv, [&](size_t x_i) {
      cblas_sgemv(
          CblasRowMajor, CblasNoTrans,
          nq, d,
          1, Q, d,
          Xs + x_i * d, 1, 0,
          S + x_i * nq, 1);
    });
  } else {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        nv, nq, d,
        1,
        Xs, d,
        Q, d,
        0, S, nq);
  }
}

void similarity_dot_product_trans_out_fp16(const float *Q, const half_float::half *Xs, float *S, const size_t nq, const size_t nv, const size_t d, float *fp32_cache_buffer, size_t cache_size) {
  // Q: (nq, d)
  // Xs: (nv, d)
  // transpose C
  // C: (nv, nq)

  // for_loop(0, nv, [&](size_t x_i) {
  //   fp16_load(fp32_cache_buffer, Xs + x_i * d, d);

  //   cblas_sgemv(
  //       CblasRowMajor, CblasNoTrans,
  //       nq, d,
  //       1, Q, d,
  //       fp32_cache_buffer, 1, 0,
  //       S + x_i * nq, 1);
  // });

  if (nq < 15) {
    batch_loop(0, nv, cache_size, [&](size_t i_batch_start, size_t i_batch_size) {
      // load into cache
      fp16_load(fp32_cache_buffer, Xs + i_batch_start * d, i_batch_size * d);

      const constexpr size_t tile = 4;
      for (size_t ii = 0; ii < i_batch_size; ii += tile) {

        for (size_t jj = 0; jj < nq; jj += tile) {

          for (int i = ii; i < std::min(ii + tile, i_batch_size); i++) {

            for (int j = jj; j < std::min(jj + tile, nq); j++) {

              float sum = 0;
#pragma omp simd reduction(+ : sum)
              for (int k = 0; k < d; k++)
                sum += fp32_cache_buffer[i * d + k] + Q[j * d + k];

              S[j + (i + i_batch_start) * nq] = sum;

              // S[j + i * nq] = cblas_sdot(d, Xs + i * d, 1, Q + j * d, 1);
            }
          }
        }
      }
    });

  } else {

    batch_loop(0, nv, cache_size, [&](size_t i_batch_start, size_t i_batch_size) {
      fp16_load(fp32_cache_buffer, Xs + i_batch_start * d, i_batch_size * d);

      cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasTrans,
          i_batch_size, nq, d,
          1,
          fp32_cache_buffer, d,
          Q, d,
          0, S + i_batch_start * nq, nq);
    });
  }
}

void random_fill(float *A, const size_t n, size_t seed) {

  shard_loop<true>(0, n, [&](size_t start_, size_t size_) {
    std::mt19937 rng(seed + omp_get_thread_num());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for_loop(start_, size_, [&](size_t i) {
      A[i] = dist(rng);
    });
  });
}

void random_fill_fp16(half_float::half *A, const size_t n, size_t seed) {

  // this use openMP
  shard_loop<true>(0, n, [&](size_t start_, size_t size_) {
    std::mt19937 rng(seed + omp_get_thread_num());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for_loop(start_, size_, [&](size_t i) {
      A[i] = dist(rng);
    });
  });
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

float StopWatch::split_print(const std::string &message, bool reset_) {
  float t = split();
  std::cout << message << t << " ms" << std::endl;
  if (reset_)
    reset();

  return t;
}

} // namespace seqr