#include <cassert>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

using namespace std;

#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK (2 * THREADS_PER_BLOCK)
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(N) ((N) >> LOG_MEM_BANKS)

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

const uint64_t N = (1 << 10);

void type1(uint32_t *input, uint32_t *output, uint32_t N);
void type2(uint32_t *input, uint32_t *output, uint32_t N);
void type3(uint32_t *input, uint32_t *output, uint32_t N);

__global__ void ptype1(uint32_t *input, uint32_t *output, uint32_t N, uint32_t powerOfTwo);
__global__ void ptype2(uint32_t *input, uint32_t *output, uint32_t N, uint32_t *sums);
__global__ void add(uint32_t *input, uint32_t *output, uint32_t N);
__global__ void add(uint32_t *in_1, uint32_t *in_2, uint32_t *output, uint32_t N);


__host__ void thrust_sum(const uint32_t* input, uint32_t* output) {
    output[0] = 0;
    for(uint32_t i = 1; i < N; i++){
      output[i] = output[i - 1] + input[i - 1];
    }
}

__host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt,
                           const uint64_t size) {
  for (uint64_t i = 0; i < size; i++) {
    if (w_ref[i] != w_opt[i]) {
      cout << "Differences found between the two arrays.\n";
      assert(false);
    }
  }
  cout << "No differences found between base and test versions\n";
}

void cuda_sum(uint32_t *input, uint32_t *output, uint32_t N){
	uint32_t SIZE_IN_BYTES = N * sizeof(uint32_t);
	
	uint32_t *d_out, *d_in;

	cudaMalloc(&d_in, SIZE_IN_BYTES);
	cudaMalloc(&d_out, SIZE_IN_BYTES);

	cudaMemcpy(d_in, input, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	if(N > ELEMENTS_PER_BLOCK){
		type2(d_in, d_out, N);
	}
	else{
		type1(d_in, d_out, N);
	}
	
	cudaMemcpy(output, d_out, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	
	cudaFree(d_in);
	cudaFree(d_out);
}

int main() {
  srand(time(NULL));
  auto* h_input = new uint32_t[N];
  fill_n(h_input, N, 1);

  // Use Thrust code as reference
  auto* h_thrust_ref = new uint32_t[N];
  fill_n(h_thrust_ref, N, 0);
  // TODO: Time your code
  thrust_sum(h_input, h_thrust_ref);

	
	uint32_t SIZE_IN_BYTES = N * sizeof(uint32_t);
	uint32_t *input = (uint32_t*)malloc(SIZE_IN_BYTES);
	
	for(uint32_t i = 0; i < N; i++){
		// input[i] = rand() % 10;
		input[i] = 1;
	}
	
	printf("N = %lu\n", N);

	auto* output = new uint32_t[N];
  fill_n(output, N, 0);
	cuda_sum(input, output, N);

  check_result(h_thrust_ref, output, N);

  // TODO: Use a CUDA kernel, time your code
  delete[] h_thrust_ref;
  delete[] h_input;

  return EXIT_SUCCESS;
}

uint32_t nextPowerOfTwo(uint32_t N){
	uint32_t powerOfTwo = 1;
	while(powerOfTwo < N){
		powerOfTwo <<= 1;
	}
	return powerOfTwo;
}

void type1(uint32_t *input, uint32_t *output, uint32_t N){
	// printf("Entered type1\n");
	uint32_t powerOfTwo = nextPowerOfTwo(N);
	ptype1<<<1, (N + 1) / 2, 2 * powerOfTwo * sizeof(uint32_t)>>>(input, output, N, powerOfTwo);
	// printf("Exited type1\n");
}

void type2(uint32_t *input, uint32_t *output, uint32_t N){
	// printf("Entered type2\n");
	uint32_t remainder = N % ELEMENTS_PER_BLOCK;

	if(remainder == 0){
		type3(input, output, N);
	}
	else{
		uint32_t lengthMultiple = N - remainder;
		type3(input, output, lengthMultiple);
		type1(&(input[lengthMultiple]), &(output[lengthMultiple]), remainder);
		add<<<1, remainder>>>(&(input[lengthMultiple - 1]), &(output[lengthMultiple - 1]), &(output[lengthMultiple]), remainder);
	}
	// printf("Exited type2\n");
}

void type3(uint32_t *input, uint32_t *output, uint32_t N){
	// printf("Entered type3\n");
	uint32_t numBlocks = N / ELEMENTS_PER_BLOCK;
	uint32_t sharedMemorySize = ELEMENTS_PER_BLOCK * sizeof(uint32_t);

	uint32_t *sums, *incr;
	cudaMalloc(&sums, numBlocks * sizeof(uint32_t));
	cudaMalloc(&incr, numBlocks * sizeof(uint32_t));

	ptype2<<<numBlocks, THREADS_PER_BLOCK, 2 * sharedMemorySize>>>(input, output, ELEMENTS_PER_BLOCK, sums);

	uint32_t sumsArrThreadsNeeded = (numBlocks + 1) / 2;

	if(sumsArrThreadsNeeded > THREADS_PER_BLOCK){
		type2(sums, incr, numBlocks);
	}
	else{
		type1(sums, incr, numBlocks);
	}
	
	add<<<numBlocks, ELEMENTS_PER_BLOCK>>>(incr, output, ELEMENTS_PER_BLOCK);

	cudaFree(sums);
	cudaFree(incr);
	// printf("Exited type3\n");
}

__global__ void ptype1(uint32_t *input, uint32_t *output, uint32_t N, uint32_t powerOfTwo){
	extern __shared__ uint32_t temp[];
	uint32_t threadID = threadIdx.x;

	uint32_t ai = threadID;
	uint32_t bi = threadID + (N / 2);

	uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if(threadID < N){
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else{
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}

	uint32_t offset = 1;
	for(uint32_t d = powerOfTwo >> 1; d > 0; d >>= 1){
		__syncthreads();

		if(threadID < d){
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if(threadID == 0){
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;
	}

	for(uint32_t d = 1; d < powerOfTwo; d <<= 1){
		offset >>= 1;
		__syncthreads();

		if(threadID < d){
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			uint32_t t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	
	if(threadID < N){
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void ptype2(uint32_t *input, uint32_t *output, uint32_t N, uint32_t *sums){
	extern __shared__ uint32_t temp[];

	uint32_t blockID = blockIdx.x;
	uint32_t threadID = threadIdx.x;
	uint32_t blockOffset = blockID * N;

	uint32_t ai = threadID;
	uint32_t bi = threadID + (N / 2);

	uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	uint32_t offset = 1;
	for(uint32_t d = (N >> 1); d > 0; d >>= 1){
		__syncthreads();

		if(threadID < d){
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}
	__syncthreads();

	if(threadID == 0){
		sums[blockID] = temp[N - 1 + CONFLICT_FREE_OFFSET(N - 1)];
		temp[N - 1 + CONFLICT_FREE_OFFSET(N - 1)] = 0;
	}

	for(uint32_t d = 1; d < N; d <<= 1){
		offset >>= 1;
		__syncthreads();
		
		if(threadID < d){
			uint32_t ai = offset * (2 * threadID + 1) - 1;
			uint32_t bi = offset * (2 * threadID + 2) - 1;
			
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			uint32_t t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
			
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void add(uint32_t *input, uint32_t *output, uint32_t N){
	uint32_t blockID = blockIdx.x;
	uint32_t threadID = threadIdx.x;
	uint32_t blockOffset = blockID * N;
	output[blockOffset + threadID] += input[blockID];
}

__global__ void add(uint32_t *in_1, uint32_t *in_2, uint32_t *output, uint32_t N){
	uint32_t blockID = blockIdx.x;
	uint32_t threadID = threadIdx.x;
	uint32_t blockOffset = blockID * N;
	output[blockOffset + threadID] += in_1[blockID] + in_2[blockID];
}