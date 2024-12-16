#include <papi.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

#define N (2048)

void error (int val){
       cout << "PAPI error" << val << '\n';
        exit(1);
}

void matmul_ijk(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      uint32_t sum = 0.0;
      for (int k = 0; k < SIZE; k++) {
        sum += A[i * SIZE + k] * B[k * SIZE + j];
      }
      C[i * SIZE + j] += sum;
    }
  }
}

void matmul_ijk_blocking(const uint32_t *A, const uint32_t *B, uint32_t *C, const int SIZE,int BLK_A,int BLK_B,int BLK_C) {
 uint32_t s;
 for (int i = 0; i < SIZE; i+= BLK_A){
    for (int j = 0; j < SIZE; j+= BLK_B){
      for (int k = 0; k < SIZE; k+= BLK_C){
        for (int i1 = i; i1 < i+BLK_A && i1< SIZE; i1++){
          for (int j1 = j; j1 < j+BLK_B && j1< SIZE; j1++){
                  s=0.0;
            for (int k1 = k; k1 < k+BLK_C && k1< SIZE; k1++){
                s += A[i1*SIZE + k1]*B[k1*SIZE + j1];
            }
                C[i1*SIZE+j1] += s;
          }
        }
      }
    }
 }
}

void init(uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      mat[i * SIZE + j] = 1;
    }
  }
}

void print_matrix(const uint32_t *mat, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      cout << mat[i * SIZE + j] << "\t";
    }
    cout << "\n";
  }
}

void check_result(const uint32_t *ref, const uint32_t *opt, const int SIZE) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (ref[i * SIZE + j] != opt[i * SIZE + j]) {
        assert(false && "Diff found between sequential and blocked versions!\n");
      }
    }
  }
}

int main() {
  int val,EventSet = PAPI_NULL; 
  long_long values[4];
  ofstream MyFile("result.txt");
  ofstream MyFile1("result1.txt");
  uint32_t *A = new uint32_t[N * N];
  uint32_t *B = new uint32_t[N * N];
  uint32_t *C_seq = new uint32_t[N * N];

  init(A, N);
  init(B, N);
  val = PAPI_library_init(PAPI_VER_CURRENT);
  if (val != PAPI_VER_CURRENT)
      error(val);
  val = PAPI_create_eventset(&EventSet);
  if (val != PAPI_OK)
      error(val);
  val = PAPI_add_event(EventSet, PAPI_L1_DCM);
  val = PAPI_add_event(EventSet ,PAPI_L2_DCM);
  val = PAPI_add_event(EventSet,PAPI_L1_TCM);
  val = PAPI_add_event(EventSet ,PAPI_L2_TCM);
  if (val != PAPI_OK)
      error(val);
  int BLK_A,BLK_B,BLK_C;
  uint32_t duration=0.0;
  double fac=0.0;

  init(C_seq, N);
  HRTimer start = HR::now();
  val = PAPI_start(EventSet);
  if (val != PAPI_OK)
    error(val);
  matmul_ijk(A, B, C_seq, N);
  val = PAPI_stop(EventSet, values);
  if (val != PAPI_OK)
    error(val);
  HRTimer end = HR::now();
  duration += duration_cast<microseconds>(end - start).count();
  for(int it1=0;it1<4;it1++){
    MyFile1 << values[it1] << ' ';
  }
  MyFile1 << endl;
  MyFile1 << "Time without blocking (us): " << duration << "\n\n";
  fac = duration;

  for(BLK_A=4;BLK_A<=64;BLK_A=BLK_A*2){
      for(BLK_B=4;BLK_B<=64;BLK_B=BLK_B*2){
          for(BLK_C=4;BLK_C<=64;BLK_C=BLK_C*2){
              MyFile1 << BLK_A << ' ' << BLK_B << ' ' << BLK_C << "\n";
              duration = 0.0;

              uint32_t *C_blk = new uint32_t[N * N];
              init(C_blk,N);
              HRTimer start = HR::now();
              val = PAPI_start(EventSet);
              if (val != PAPI_OK)
                error(val);
              matmul_ijk_blocking(A, B, C_blk, N,BLK_A,BLK_B,BLK_C);
              val = PAPI_stop(EventSet, values);
              if (val != PAPI_OK)
                error(val);
              HRTimer end = HR::now();
              duration += duration_cast<microseconds>(end - start).count();
              check_result(C_seq,C_blk,N);
              for(int it1=0;it1<4;it1++){
                MyFile1 << values[it1] << ' ';
              }

              MyFile1 << endl;
              MyFile << BLK_A << '*' << BLK_C << " & " << BLK_C << '*' << BLK_B << " & " << BLK_A << '*' << BLK_B << " & " << (double)(fac/(double)duration) << "\\" << "\\" << endl;
          }
      }
  }
  MyFile.close();
  MyFile1.close();

  
  return EXIT_SUCCESS;
}