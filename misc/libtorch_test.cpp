#include "cblas.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
// #include <torch/torch.h>
// #include <torch/types.h>

void blas_test(int m = 100000, int n = 512, int k = 10, int lo = 10) {
  int size = 1000;

  // Allocate memory
  float *matrix1 = new float[m * k];
  float *matrix2 = new float[k * n];
  float *result = new float[m * n];

  // Fill your matrices with random numbers

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10; i++) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                matrix1, k, matrix2, n, 0.0, result, n);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << "CBLAS took " << diff.count() << " seconds.\n";
}

void openmp_for_loop_dot1(float *embs, float *Q, float *S, size_t n, size_t nq,
                          size_t vec_dim) {
  // embs: n * vec_dim
  // Q:    nq * vec_dim
  // S:    n * nq

#pragma omp parallel for
  for (int i = 0; i < n; i++) {

    for (int q = 0; q < nq; q++) {

      S[i * nq + q] =
          cblas_sdot(vec_dim, embs + i * vec_dim, 1, Q + q * vec_dim, 1);
    }
  }
// #pragma omp parallel for
//   for (int q = 0; q < nq; q++) {
//     for (int i = 0; i < n; i++) {

//       S[i + q * n] =
//           cblas_sdot(vec_dim, embs + i * vec_dim, 1, Q + q * vec_dim, 1);
//     }
//   }
}
void blas_dot_test(int m = 100000, int n = 512, int k = 10, int lo = 10) {
  int size = 1000;

  // Allocate memory
  float *matrix1 = new float[m * k];
  float *matrix2 = new float[n * k];
  float *result = new float[m * n];

  // Fill your matrices with random numbers

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10; i++) {

    openmp_for_loop_dot1(matrix1, matrix2, result, m, n, k);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << "CBLAS dot took " << diff.count() << " seconds.\n";
}

// void torch_test(int m = 100000, int n = 512, int k = 10, int lo = 10) {
//   auto options = torch::TensorOptions().dtype(torch::kFloat32);

//   // Random matrices
//   auto matrix1 = torch::randn({m, k}, options);
//   auto matrix2 = torch::randn({k, n}, options);

//   auto start = std::chrono::high_resolution_clock::now();
//   for (int i = 0; i < 10; i++) {
//     torch::mm(matrix1, matrix2);
//   }
//   auto end = std::chrono::high_resolution_clock::now();

//   std::chrono::duration<double> diff = end - start;
//   std::cout << "Torch took " << diff.count() << " seconds.\n";
// }

void exp_test(int n = 1e6) {

  std::vector<double> A(n, 2.0); // Initialize array with 1's

  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    A[i] = std::exp(A[i]);
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;

  std::cout << "bare exp Elapsed time: " << elapsed.count() << " s\n";
}

// void torch_exp_test() {
//   torch::Tensor A = torch::ones({long(1e6)});

//   auto start = std::chrono::high_resolution_clock::now();

//   A = A.exp();

//   auto finish = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed = finish - start;
//   std::cout << "torch expElapsed time: " << elapsed.count() << " s\n";
// }

int main() {
  // auto start_time = std::chrono::high_resolution_clock::now();

  // torch::Tensor tensor = torch::zeros({5000000});  // Create the tensor

  // auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  // auto duration =
  // std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();

  // std::cout << "Time taken: " << duration << " ms" << std::endl;
  // return 0;

  int m = 1000000, n = 512, k = 2, lo = 10;
  blas_test(m, n, k, lo);
  blas_dot_test(m, n, k, lo);
  // torch_test(m, n, k, lo);

  exp_test();
  // torch_exp_test();
}