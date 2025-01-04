#include <cstddef>
#include <cstdio>
#include <iostream>
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
  std::vector<float> t(n);

    fillVector(t);

  return t;
}

int main() {
    // Define matrix dimensions
    int M = 5; // Rows of A and C
    int N = 4; // Columns of B and C
    int K = 2; // Columns of A and Rows of B

    // Define matrices A, B, and C
    // float A[2][3] = {{1, 2}, {3, 4}};
    // float B[2][3] = {{5, 6}, {7, 8}};
    // float C[2][3] = {{0, 0}, {0, 0}}; // Initialize C to zero


    auto A=create_rand_vector(M*K),
    B = create_rand_vector(N * K),
    C = create_rand_vector(M * N);

    // Set parameters for cblas_dgemm
    float alpha = 1.0;
    float beta = 0.0;
    int lda = K; // Leading dimension of A (should be at least M)
    int ldb = N; // Leading dimension of B (should be at least K)
    int ldc = N; // Leading dimension of C (should be at least M)

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha,
                A.data(), lda,
                B.data(), ldb,
                beta, C.data(), ldc);

    // Print the result
    printf("Result matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i*N +  j]);
        }
        printf("\n");
    }

    return 0;
}
