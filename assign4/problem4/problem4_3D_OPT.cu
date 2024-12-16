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

const uint64_t N = (1 << 7); // Matrix width
const uint64_t SIZE_IN_BYTES_MATRIX = N * N * N * sizeof(float);
#define M 5 // Convolution filter width
#define TILE_WIDTH 4 // Output tile width
#define BLOCK_WIDTH (TILE_WIDTH + M - 1) // Block width

__host__ __device__ bool is_valid_3D(const int i, const int j, const int k,  const uint64_t N){
	return 0 <= i && i < N && 0 <= j && j < N && 0 <= k && k < N;
}

__constant__ float kernel_mat[M][M][M];

__global__ void kernel3D(const float *input, float *output){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int dep_o = blockIdx.z * TILE_WIDTH + tz; // Output depth
	int row_o = blockIdx.y * TILE_WIDTH + ty; // Output row
	int col_o = blockIdx.x * TILE_WIDTH + tx; // Outut column
	
	int dep_i = dep_o - M / 2; // Input depth
	int row_i = row_o - M / 2; // Input row
	int col_i = col_o - M / 2; // Input column

	__shared__ float temp[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

	if(is_valid_3D(dep_i, row_i, col_i, N)){
		temp[tz][ty][tx] = input[dep_i * N * N + row_i * N + col_i];
	}
	else{
		temp[tz][ty][tx] = 0;
	}
	__syncthreads();

	if(is_valid_3D(tz, ty, tx, TILE_WIDTH) && is_valid_3D(dep_o, row_o, col_o, N)){
		float sum = 0;
		for(int i = 0; i < M; i++){
			for(int j = 0; j < M; j++){
				for(int k = 0; k < M; k++){
					sum += temp[tz + i][ty + j][tx + k] * kernel_mat[i][j][k];
				}
			}
		}
		output[dep_o * N * N + row_o * N + col_o] = sum / (M * M * M);
	}
}

__host__ void check_result_3D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
        double this_diff =
            w_ref[i * N * N + j * N + k] - w_opt[i * N * N + j * N + k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
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

void print3D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "\n";
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

void init_matrix_3D(float *a,  const uint64_t N){
		for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++){
						for(int k = 0; k < N; k++){
								a[i * N * N + j * N + k] = rand() % 100;
						}
				}
		}
}

void calculate_ref_3D(const float *a, float *b, const float *filter){
	float sum;
	for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				for(int k = 0; k < N; k++){
					sum = 0;		
					for(int di = -M / 2; di <= M / 2; di++){
						for(int dj = -M / 2; dj <= M / 2; dj++){
							for(int dk = -M / 2; dk <= M / 2; dk++){
								if(!is_valid_3D(i + di, j + dj, k + dk, N)){
										continue;
								}
								sum += a[(i + di) * N * N + (j + dj) * N + (k + dk)] * filter[(M / 2 + di) * M * M + (M / 2 + dj) * M + (M / 2 + dk)];
							}
						}
					}
					b[i * N * N + j * N + k] = sum / (M * M * M); 
				}
			}
	}
}

void convolution_3D(){
	int SIZE_IN_BYTES_FILTER = M * M * M * sizeof(float);

	float *a = NULL, *b_ref = NULL, *b = NULL, *filter = NULL;
	a = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	b_ref = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	b = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	filter = (float*)malloc(SIZE_IN_BYTES_FILTER);

	init_matrix_3D(a, N);
	init_matrix_3D(filter, M);

	double clkbegin = rtclock();
	calculate_ref_3D(a, b_ref, filter);
	double clkend = rtclock();
	double cpu_time = clkend - clkbegin;
	cout << "Convolution 3D time on CPU: " << cpu_time * 1000 << " msec " << endl;

  	float *input = NULL, *output = NULL;
	cudaCheckError(cudaMalloc(&input, SIZE_IN_BYTES_MATRIX));
	cudaCheckError(cudaMalloc(&output, SIZE_IN_BYTES_MATRIX));

	cudaCheckError(cudaMemcpy(input, a, SIZE_IN_BYTES_MATRIX, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpyToSymbol(kernel_mat, filter, SIZE_IN_BYTES_FILTER));

	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N/ TILE_WIDTH, N / TILE_WIDTH);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	kernel3D<<<numBlocks, threadsPerBlock>>>(input, output);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0);
	cudaCheckError(cudaMemcpy(b, output, SIZE_IN_BYTES_MATRIX, cudaMemcpyDeviceToHost));	
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	check_result_3D(b_ref, b);
	cout << "Convolution 3D time on GPU: " << kernel_time << " msec " << endl;
	
	free(a);
	free(b);
	free(filter);

	cudaFree(input);
	cudaFree(output);
}

int main() {
  	srand(time(NULL));

	int SIZE_IN_BYTES_FILTER = M * M * M * sizeof(float);

	float *a = NULL, *b_ref = NULL, *b = NULL, *filter = NULL;
	a = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	b_ref = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	b = (float*)malloc(SIZE_IN_BYTES_MATRIX);
	filter = (float*)malloc(SIZE_IN_BYTES_FILTER);

	init_matrix_3D(a, N);
	init_matrix_3D(filter, M);

	double clkbegin = rtclock();
	calculate_ref_3D(a, b_ref, filter);
	double clkend = rtclock();
	double cpu_time = clkend - clkbegin;
	cout << "Convolution 3D time on CPU: " << cpu_time * 1000 << " msec " << endl;

  	float *input = NULL, *output = NULL;
	cudaCheckError(cudaMalloc(&input, SIZE_IN_BYTES_MATRIX));
	cudaCheckError(cudaMalloc(&output, SIZE_IN_BYTES_MATRIX));

	cudaEvent_t start1,start2, end1,end2;
	float kernel_time1,kernel_time2;
	cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);

	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N/ TILE_WIDTH, N / TILE_WIDTH);

	cudaCheckError(cudaEventRecord(start1, 0));
	cudaCheckError(cudaMemcpy(input, a, SIZE_IN_BYTES_MATRIX, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpyToSymbol(kernel_mat, filter, SIZE_IN_BYTES_FILTER));

	cudaCheckError(cudaEventRecord(start2, 0));
	kernel3D<<<numBlocks, threadsPerBlock>>>(input, output);
	cudaCheckError(cudaEventRecord(end2, 0));
	cudaCheckError(cudaDeviceSynchronize());
	cudaCheckError(cudaMemcpy(b, output, SIZE_IN_BYTES_MATRIX, cudaMemcpyDeviceToHost));	
	cudaCheckError(cudaEventRecord(end1, 0));
	cudaCheckError(cudaDeviceSynchronize());

	cudaCheckError(cudaGetLastError());

	check_result_3D(b_ref, b);
	cudaEventElapsedTime(&kernel_time1, start1, end1);
    cudaEventElapsedTime(&kernel_time2, start2, end2);

	cout << "Convolution 3D time on GPU kernel_time1 for cpu compare: " << kernel_time1 << " msec " << endl;
	cout << "Convolution 3D time on GPU kernel_time2 for gpu compare: " << kernel_time2 << " msec " << endl;
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
