#include "utils.h"

#include "cblas.h"
#include <cstddef>

float dot_product(const float *a, const float *b, const size_t d) {
  return cblas_sdot(d, a, 1, b, 1);
}

void mdot_product(const float *A, const float *b, float *c, size_t m,
                  size_t n) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, b, 1, 0.0, c, 1);
  // cblas_sgemv(CblasRowMajor,, m, n, 1.0, A, n, 0.0, c, 1);
}

void mmdot_product(const float *A, const float *B, float *C, const size_t m,
                   const size_t d, const size_t n) {
  if (m == 1 && n == 1) {
    *C = dot_product(A, B, d);
  } else {
    // A (m, d)
    // B (n, d)
    // C (m, n)

    // M=m
    // K=d
    // N=n
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, 1.0, A, d, B,
                d, 0.0, C, n);
  }
}

