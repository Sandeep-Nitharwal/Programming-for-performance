#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)               \
{                                       \
	gpuAssert((ans), __FILE__, __LINE__); \
}

inline void gpuAssert(cudaError_t code, const char *file, int line,
											bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
						line);
		if (abort)
			exit(code);
	}
}

const uint64_t N = (256);

__global__ void kernel1(const double* input, double* output) 
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>0 && j>0 && k>0 && i<N-1 && j< N-1 && k < N-1)
    {
        output[i + j * N + k * N * N] =  0.8 * (input[(i - 1) + j * N + k * N * N] + input[(i + 1) + j * N + k* N * N] + input[i + (j - 1) * N + k * N * N] + input[i + (j + 1) * N + k* N * N] + input[i + j * N + (k - 1)* N * N] + input[i + j * N + (k + 1)* N * N]);
    }
}

__host__ void stencil(double* input, double* output) {
    for (uint64_t i=1; i<N-1; i++) 
    {
        for (uint64_t j=1; j<N-1; j++) 
        {
            for (uint64_t k=1; k<N-1; k++) 
            {
                output[i * N * N + j * N + k] = 0.8 * (input[(i - 1) * N * N + j * N + k] + input[(i + 1) * N * N + j * N + k] + input[i * N * N + (j - 1) * N + k] + input[i * N * N + (j + 1) * N + k] + input[i * N * N + j * N + k - 1] + input[i * N * N + j * N + k + 1]);
            }
        }
    }
}

__host__ void check_result(const double *w_ref, const double *w_opt,
													 const uint64_t size)
{
	double maxdiff = 0.0;
	int numdiffs = 0;

	for (uint64_t i = 0; i < size; i++)
	{
		for (uint64_t j = 0; j < size; j++)
		{
			for (uint64_t k = 0; k < size; k++)
			{
				double this_diff =
						w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
				if (std::fabs(this_diff) > THRESHOLD)
				{
					numdiffs++;
					if (this_diff > maxdiff)
					{
						maxdiff = this_diff;
					}
				}
			}
		}
	}

	if (numdiffs > 0)
	{
		cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
				 << "; Max Diff = " << maxdiff << endl;
	}
	else
	{
		cout << "No differences found between base and test versions\n";
	}
}

void print_mat(const double *A)
{
	for (uint64_t i = 0; i < N; ++i)
	{
		for (uint64_t j = 0; j < N; ++j)
		{
			for (uint64_t k = 0; k < N; ++k)
			{
				printf("%lf,", A[i * N * N + j * N + k]);
			}
			printf("      ");
		}
		printf("\n");
	}
}

double rtclock()
{ // Seconds
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0)
	{
		cout << "Error return from gettimeofday: " << stat << "\n";
	}
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main()
{
	uint64_t SIZE = N * N * N;
	uint64_t SIZE_BYTES = SIZE * sizeof(double);

    double *h_in, *h_out, *h_out_1;
    double *k1_in, *k1_out;

    h_in = new double[SIZE]; 
    h_out = new double[SIZE](); 
    h_out_1 = new double[SIZE]; 

    for(uint64_t i=0;i<N;i++)
    {
        for(uint64_t j=0;j<N;j++)
        {
            for(uint64_t k=0;k<N;k++)
            {
                h_in[i * N * N + j * N + k] = (i - j + k) * 0.1;
            }
        }
    }

    cudaCheckError(cudaMalloc(&k1_in, SIZE_BYTES));
    cudaCheckError(cudaMalloc(&k1_out, SIZE_BYTES));


	double clkbegin = rtclock();
	stencil(h_in, h_out);
	double clkend = rtclock();

	double cpu_time = clkend - clkbegin;
	cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

	float t1, t2;
    cudaEvent_t start1,start2, end1, end2 ;
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);

    dim3 k1_TBP(8,8,8);
    dim3 k1_grid (N/8, N/8, N/8);

    cudaCheckError(cudaEventRecord(start1, 0));
    cudaCheckError(cudaMemcpy(k1_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));
    cudaCheckError(cudaEventRecord(start2, 0));
    kernel1<<<k1_grid, k1_TBP>>>(k1_in, k1_out);
    cudaCheckError(cudaEventRecord(end2, 0));
    cudaCheckError(cudaEventSynchronize(end2));
    cudaCheckError(cudaMemcpy(h_out_1, k1_out, SIZE_BYTES, cudaMemcpyDeviceToHost)); 
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(end1, 0));
    cudaCheckError(cudaEventSynchronize(end1));

    check_result(h_out, h_out_1, N);

	cudaEventElapsedTime(&t2, start2, end2);
	cudaEventElapsedTime(&t1, start1, end1);

	std::cout << "Kernel 1 time (ms) (without 2-way memcpy): " << t2 << "\n";
	std::cout << "Kernel 1 time (ms) (with 2-way memcpy): " << t1 << "\n";

    cudaFree(k1_in);
    cudaFree(k1_out);

    free(h_in);
    free(h_out);
    free(h_out_1);

	return EXIT_SUCCESS;
}
