#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define THREADS_PER_BLOCK 16
const uint64_t N = (1 << 13);
const uint64_t SIZE_IN_BYTES_MATRIX = N * N * sizeof(float);
const int M = 5;
const int SIZE_IN_BYTES_FILTER = M * M * sizeof(float);


__global__ void kernel2D(const float *input, float *output, const float *kernel_mat, const uint64_t N, const uint64_t M) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		float sum = 0;		
		for(int i1 = -M / 2; i1 <= M / 2; i1++){
			for(int j1 = -M / 2; j1 <= M / 2; j1++){
					if(((i + i1) >= 0) && ((i + i1) < N) && ((j + j1) >= 0) && ((j + j1) < N)){
						sum = sum + input[(i + i1)  + (j + j1) * N ] * kernel_mat[(M / 2 + i1)  + (M / 2 + j1) * M];
					}
			}
		}
		output[i + j * N] = sum / (M * M); 
}

__host__ void check_result_2D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  uint64_t numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
        double this_diff =
            w_ref[i * N + j] - w_opt[i * N + j];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print2D(const float* A) {
  for (uint64_t i = 0; i < N; ++i) {
    for (uint64_t j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void init_matrix_2D(float *a, const uint64_t N){
	for(uint64_t i = 0; i < N; i++){
		for(uint64_t j = 0; j < N; j++){
			a[i * N + j] = rand() % 100;
		}
	}
}

void calculate_ref_2D(const float *a, float *b, const float *filter, const uint64_t N, const uint64_t M){
	for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				float sum = 0;		
				for(int i1 = -M / 2; i1 <= M / 2; i1++){
					for(int j1 = -M / 2; j1 <= M / 2; j1++){
						if(((i + i1) >= 0) && (i + i1 < N) && (j + j1 >= 0) && (j + j1 < N)){
							sum = sum + a[(i + i1)  + (j + j1) * N] * filter[(M / 2 + i1)  + (M / 2 + j1) * M];
						}
					}
				}
				b[i + j * N] = sum / (M * M); 
			}
	}
}

void convolution_2D(){

}

int main() {
  	srand(time(NULL));

	float *a, *b_ref, *b, *filter;
	a = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	b_ref = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	b = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	filter = (float*)malloc(SIZE_IN_BYTES_FILTER);

	init_matrix_2D(a, N);
	init_matrix_2D(filter, M);

	double clkbegin = rtclock();
	calculate_ref_2D(a, b_ref, filter, N, M);
	double clkend = rtclock();
	double cpu_time = clkend - clkbegin;
	cout << "Convolution 2D time on CPU: " << cpu_time * 1000 << " msec " << endl;

  	float *input = NULL, *output = NULL, *kernel_mat = NULL;
	cudaCheckError(cudaMalloc(&input, SIZE_IN_BYTES_MATRIX));
	cudaCheckError(cudaMalloc(&output, SIZE_IN_BYTES_MATRIX));
	cudaCheckError(cudaMalloc(&kernel_mat, SIZE_IN_BYTES_FILTER));

	cudaEvent_t start1,start2, end1,end2;
	float kernel_time1,kernel_time2;
	cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);

	dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	cudaCheckError(cudaEventRecord(start1, 0));
	cudaCheckError(cudaMemcpy(input, a, SIZE_IN_BYTES_MATRIX, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(kernel_mat, filter, SIZE_IN_BYTES_FILTER, cudaMemcpyHostToDevice));

	cudaCheckError(cudaEventRecord(start2, 0));
	kernel2D<<<numBlocks, threadsPerBlock>>>(input, output, kernel_mat, N, M);
	cudaCheckError(cudaEventRecord(end2, 0));
	cudaCheckError(cudaDeviceSynchronize());
	cudaCheckError(cudaMemcpy(b, output, SIZE_IN_BYTES_MATRIX, cudaMemcpyDeviceToHost));	
	cudaCheckError(cudaEventRecord(end1, 0));
	cudaCheckError(cudaDeviceSynchronize());

	cudaCheckError(cudaGetLastError());

	check_result_2D(b_ref, b);
	cudaEventElapsedTime(&kernel_time1, start1, end1);
    cudaEventElapsedTime(&kernel_time2, start2, end2);

	cout << "Convolution 2D time on GPU kernel_time1 for cpu compare: " << kernel_time1 << " msec " << endl;
	cout << "Convolution 2D time on GPU kernel_time2 for gpu compare: " << kernel_time2 << " msec " << endl;
	cudaEventDestroy(start1);
	cudaEventDestroy(start2);
	cudaEventDestroy(end1);
	cudaEventDestroy(end2);



	free(a);
	free(b);
	free(filter);

	cudaFree(input);
	cudaFree(output);
	cudaFree(kernel_mat);

  	return EXIT_SUCCESS;
}
