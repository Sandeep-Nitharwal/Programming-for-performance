#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();

#define N (1024)

void matmul_seq(float** A, float** B, float** C) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_sse4(float** A, float** B, float** C) {
  __m128 tmp,a,b;
  for(int i = 0; i < N; ++i) {
    for(int k = 0; k < N; ++k) { 
        a = _mm_set1_ps(A[i][k]);
        for(int j = 0; j < N; j+=4) { 
            tmp = _mm_loadu_ps(&(C[i][j]));
            b = _mm_loadu_ps(&(B[k][j])); 
            tmp = _mm_add_ps(tmp, _mm_mul_ps(a, b));
            _mm_storeu_ps(&(C[i][j]), tmp);
        }
    }
  }
}

void matmul_avx2(float** A, float** B, float** C) {
    __m256 tmp, a, b;
    for(int i = 0; i < N; ++i) {
        for(int k = 0; k < N; ++k) {
            a = _mm256_set1_ps(A[i][k]);
            for(int j = 0; j < N; j += 8) {
                tmp = _mm256_loadu_ps(&(C[i][j]));
                b = _mm256_loadu_ps(&(B[k][j]));
                tmp = _mm256_add_ps(tmp, _mm256_mul_ps(a, b));
                _mm256_storeu_ps(&(C[i][j]), tmp);
            }
        }
    }
}

void matmul_sse4_align(float** A, float** B, float** C) {
    // Inform compiler that arrays are aligned
    __builtin_assume_aligned(A, 16);
    __builtin_assume_aligned(B, 16);
    __builtin_assume_aligned(C, 16);
    
    __m128 tmp, a, b;
    for(int i = 0; i < N; ++i) {
        for(int k = 0; k < N; ++k) {
            a = _mm_set1_ps(A[i][k]);
            for(int j = 0; j < N; j += 4) {
                tmp = _mm_load_ps(&(C[i][j]));
                b = _mm_load_ps(&(B[k][j]));
                tmp = _mm_add_ps(tmp, _mm_mul_ps(a, b));
                _mm_store_ps(&(C[i][j]), tmp);
            }
        }
    }
}

void matmul_avx2_align(float** A, float** B, float** C) {
    __builtin_assume_aligned(A, 32);
    __builtin_assume_aligned(B, 32);
    __builtin_assume_aligned(C, 32);
    
    __m256 tmp, a, b;
    for(int i = 0; i < N; ++i) {
        for(int k = 0; k < N; ++k) {
            a = _mm256_set1_ps(A[i][k]);
            for(int j = 0; j < N; j += 8) {
                tmp = _mm256_load_ps(&(C[i][j]));
                b = _mm256_load_ps(&(B[k][j]));
                tmp = _mm256_add_ps(tmp, _mm256_mul_ps(a, b));
                _mm256_store_ps(&(C[i][j]), tmp);
            }
        }
    }
}

void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > EPSILON) {
        numdiffs++;
        if (fabs(this_diff) > maxdiff)
          maxdiff = fabs(this_diff);
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  auto** A = new float*[N];
  auto** B = new float*[N];
  float** A_sse4_align = (float**)aligned_alloc(16, N * sizeof(float*));
  float** A_avx2_align = (float**)aligned_alloc(32, N * sizeof(float*));
  float** B_sse4_align = (float**)aligned_alloc(16, N * sizeof(float*));
  float** B_avx2_align = (float**)aligned_alloc(32, N * sizeof(float*));
  float** C_sse4_align = (float**)aligned_alloc(16, N * sizeof(float*));
  float** C_avx2_align = (float**)aligned_alloc(32, N * sizeof(float*));
  auto** C_seq = new float*[N];
  auto** C_sse4 = new float*[N];
  auto** C_avx2 = new float*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new float[N]();
    A_sse4_align[i] = (float*)aligned_alloc(16, N * sizeof(float));
    A_avx2_align[i] = (float*)aligned_alloc(32, N * sizeof(float));
    B[i] = new float[N]();
    B_sse4_align[i] = (float*)aligned_alloc(16, N * sizeof(float));
    B_avx2_align[i] = (float*)aligned_alloc(32, N * sizeof(float));
    C_seq[i] = new float[N]();
    C_sse4[i] = new float[N]();
    C_avx2[i] = new float[N]();
    C_sse4_align[i] = (float*)aligned_alloc(16, N * sizeof(float));
    C_avx2_align[i] = (float*)aligned_alloc(32, N * sizeof(float));
  }

  // initialize arrays
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 0.1F;
      B[i][j] = 0.2F;
      C_seq[i][j] = 0.0F;
      C_sse4[i][j] = 0.0F;
      C_avx2[i][j] = 0.0F;
      A_sse4_align[i][j] = 0.1F;
      A_avx2_align[i][j] = 0.1F;
      B_sse4_align[i][j] = 0.2F;
      B_avx2_align[i][j] = 0.2F;
      C_sse4_align[i][j] = 0.0F;
      C_avx2_align[i][j] = 0.0F;
    }
  }

  HRTimer start = HR::now();
  matmul_seq(A, B, C_seq);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_sse4(A, B, C_sse4);
  end = HR::now();
  check_result(C_seq, C_sse4);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2(A, B, C_avx2);
  end = HR::now();
  check_result(C_seq, C_avx2);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_sse4_align(A_sse4_align, B_sse4_align, C_sse4_align);
  end = HR::now();
  check_result(C_seq, C_sse4_align);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4_align time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2_align(A_avx2_align, B_avx2_align, C_avx2_align);
  end = HR::now();
  check_result(C_seq, C_avx2_align);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2_align time: " << duration << " ms" << endl;

  return EXIT_SUCCESS;
}
