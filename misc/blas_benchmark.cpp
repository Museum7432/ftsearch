#include "ftsearch.h"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "cblas.h"
#include <cstdlib> // for rand() and RAND_MAX
#include <ctime>
#include <omp.h>

#include "utils.h"

void printMemoryUsage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Memory Usage: " << usage.ru_maxrss / 1024 << " MB" << std::endl;
}

void fillVector(std::vector<float> &vec) {
  static thread_local std::mt19937 generator(std::random_device{}());
  static thread_local std::uniform_real_distribution<float> distribution(0.0f,
                                                                         1.0f);

  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = distribution(generator);
  }
}

std::vector<float> create_rand_vector(size_t n) {
  std::vector<float> t(n ,0.1);

  //   fillVector(t);

  return t;
}

void openmp_for_loop_dot1(float *embs, float *Q, float *S, size_t n, size_t nq,
                          size_t vec_dim) {
  // embs: n * vec_dim
  // Q:    nq * vec_dim
  // S:    n * nq

#pragma omp parallel for
  for (int i = 0; i < n; i++) {

    for (int q = 0; q < nq; q++) {

      S[i * nq + q] = dot_product(embs + i * vec_dim, Q + q * vec_dim, vec_dim);
    }
  }
}

void openmp_for_loop_dot1_unroll(float *embs, float *Q, float *S, size_t n,
                                 size_t nq, size_t vec_dim) {
  // embs: n * vec_dim
  // Q:    nq * vec_dim
  // S:    n * nq

#pragma omp parallel for
  for (int i = 0; i < n; i++) {

    for (int q = 0; q < nq; q += 4) {

      S[i * nq + q] = dot_product(embs + i * vec_dim, Q + q * vec_dim, vec_dim);

      if (q + 1 < nq) S[i * nq + q + 1] = dot_product(embs + i * vec_dim, Q + (q + 1) * vec_dim, vec_dim);
      if (q + 2 < nq) S[i * nq + q + 2] = dot_product(embs + i * vec_dim, Q + (q + 2) * vec_dim, vec_dim);
      if (q + 3 < nq) S[i * nq + q + 3] = dot_product(embs + i * vec_dim, Q + (q + 3) * vec_dim, vec_dim);

    }
  }
}

void openmp_for_loop_dot2(float *embs, float *Q, float *S, size_t n, size_t nq,
                          size_t vec_dim) {
  // embs: n * vec_dim
  // Q:    nq * vec_dim
  // S:    n * nq

#pragma omp parallel
  {
    int num_threads = omp_get_num_threads();

    int batch_size = n / num_threads + 1;

    int thread_id = omp_get_thread_num();

    int start = thread_id * batch_size;

    int end = std::min<size_t>(n - 1, start + batch_size - 1);

    // int seq_size = end - start + 1;

    for (int i = start; i <= end; i++) {
      for (int q = 0; q < nq; q++) {

        S[i * nq + q] =
            dot_product(embs + i * vec_dim, Q + q * vec_dim, vec_dim);
      }
    }
  }
}

void openmp_for_loop_mv(float *embs, float *Q, float *S, size_t n, size_t nq,
                        size_t vec_dim) {
  // embs: n * vec_dim
  // Q:    nq * vec_dim
  // S:    n * nq

// for (int thread_id = 0; thread_id < num_threads; thread_id++)
#pragma omp parallel
  {

    int num_threads = omp_get_num_threads();

    int batch_size = n / num_threads + 1;

    int thread_id = omp_get_thread_num();

    int start = thread_id * batch_size;

    int end = std::min<size_t>(n - 1, start + batch_size - 1);

    int seq_size = end - start + 1;

    for (int i = start; i <= end; i++) {

      cblas_sgemv(CblasRowMajor, CblasNoTrans, nq, vec_dim, 1.0, Q, vec_dim,
                  embs + i * vec_dim, 1, 0.0, S + nq * i, 1);
    }
  }
}

void openmp_batch_mm(float *embs, float *Q, float *S, size_t n, size_t nq,
                     size_t vec_dim) {
  // embs: n * vec_dim
  // Q:    nq * vec_dim
  // S:    n * nq

// for (int thread_id = 0; thread_id < num_threads; thread_id++)
#pragma omp parallel
  {

    int num_threads = omp_get_num_threads();

    int batch_size = n / num_threads + 1;

    int thread_id = omp_get_thread_num();

    int start = thread_id * batch_size;

    int end = std::min<size_t>(n - 1, start + batch_size - 1);

    int seq_size = end - start + 1;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_size, nq,
                vec_dim, 1.0, embs + start * vec_dim, vec_dim, Q, nq, 0.0,
                S + start * nq, nq);
  }
}

int main() {
  //   openblas_set_num_threads(1);
  struct timespec start, end;
  double time_taken_ms;

  size_t vec_dim = 512, n = 5000000, nq = 5;

  clock_gettime(CLOCK_MONOTONIC, &start);
  auto embs = create_rand_vector(n * vec_dim);
  auto Q = create_rand_vector(nq * vec_dim);
  auto S = create_rand_vector(n * nq);

  //   std::vector<float> S(n * nq);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("init time: %f ms\n", time_taken_ms);

  // //////////////////////////////////////////////////////////////////
  // clock_gettime(CLOCK_MONOTONIC, &start);
  // // dot
  // #pragma omp parallel for
  //   for (int i = 0; i < n; i++) {

  //     for (int q = 0; q < nq; q++) {
  //       float sum = 0;
  //       for (int k = 0; k < vec_dim; k++)
  //         sum += embs[i * vec_dim + k] * Q[q * vec_dim + k];

  //       S[q * n + i] = sum;
  //     }
  //   }
  // clock_gettime(CLOCK_MONOTONIC, &end);
  // time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
  //                 (end.tv_nsec - start.tv_nsec) / 1e6;
  // printf("baseline: %f ms\n", time_taken_ms);
  // //////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_MONOTONIC, &start);

  openmp_for_loop_dot1(embs.data(), Q.data(), S.data(), n, nq, vec_dim);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("openmp_for_loop_dot1: %f ms\n", time_taken_ms);
  //////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_MONOTONIC, &start);

  openmp_for_loop_dot1_unroll(embs.data(), Q.data(), S.data(), n, nq, vec_dim);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("openmp_for_loop_dot1_unroll: %f ms\n", time_taken_ms);
  //////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_MONOTONIC, &start);

  openmp_for_loop_dot2(embs.data(), Q.data(), S.data(), n, nq, vec_dim);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("openmp_for_loop_dot2: %f ms\n", time_taken_ms);
  //////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_MONOTONIC, &start);

  openmp_for_loop_mv(embs.data(), Q.data(), S.data(), n, nq, vec_dim);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("openmp_for_loop_mv: %f ms\n", time_taken_ms);
  //////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////
  clock_gettime(CLOCK_MONOTONIC, &start);

  mmdot_product(embs.data(), Q.data(), S.data(), n, vec_dim, nq);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("mm: %f ms\n", time_taken_ms);
  //////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////

  clock_gettime(CLOCK_MONOTONIC, &start);

  openmp_batch_mm(embs.data(), Q.data(), S.data(), n, nq, vec_dim);

  clock_gettime(CLOCK_MONOTONIC, &end);
  time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                  (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("batch mm: %f ms\n", time_taken_ms);
  ////////////////////////////////////////////////////////////////

  //   //////////////////////////////////////////////////////////////////
  //   clock_gettime(CLOCK_MONOTONIC, &start);
  // #pragma omp parallel for
  //   for (int i = 0; i < n; i++) {

  //     // mmdot_product(Q.data(), embs.data() + i * vec_dim, S.data() + i *
  //     nq, nq,
  //     //               vec_dim, 1);

  //     // mdot_product(Q.data(), embs.data() + i * vec_dim, nq, size_t n)
  //     mdot_product(Q.data(), embs.data() + i * vec_dim, S.data() + i * nq,
  //     nq, vec_dim);
  //   }
  //   clock_gettime(CLOCK_MONOTONIC, &end);
  //   time_taken_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
  //                   (end.tv_nsec - start.tv_nsec) / 1e6;
  //   printf("loop m: %f ms\n", time_taken_ms);
  //   //////////////////////////////////////////////////////////////////
}