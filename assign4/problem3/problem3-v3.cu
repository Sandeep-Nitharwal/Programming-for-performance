#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define NSEC_SEC_MUL (1.0e9)
typedef unsigned long long int ull;

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


struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

__constant__ int s[10];
__constant__ double e[10];

__global__ void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6,
    double dd7, double dd8, double dd9, double dd10, double dd11, double dd12,
    double dd13, double dd14, double dd15, double dd16, double dd17,
    double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27,
    double dd28, double dd29, double dd30, double c11, double c12, double c13,
    double c14, double c15, double c16, double c17, double c18, double c19,
    double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29,
    double c210, double d2, double ey2, double c31, double c32, double c33,
    double c34, double c35, double c36, double c37, double c38, double c39,
    double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49,
    double c410, double d4, double ey4, double c51, double c52, double c53,
    double c54, double c55, double c56, double c57, double c58, double c59,
    double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69,
    double c610, double d6, double ey6, double c71, double c72, double c73,
    double c74, double c75, double c76, double c77, double c78, double c79,
    double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89,
    double c810, double d8, double ey8, double c91, double c92, double c93,
    double c94, double c95, double c96, double c97, double c98, double c99,
    double c910, double d9, double ey9, double c101, double c102, double c103,
    double c104, double c105, double c106, double c107, double c108,
    double c109, double c1010, double d10, double ey10, double kk, ull* pnts, double* temp)
{

    int my_pnts = 0;

    int r1 = blockIdx.x;
    int r2 = blockIdx.y;
    int r3 = blockIdx.z;

    int r4 = threadIdx.x;
    int r5 = threadIdx.y;
    int index = r1 * s[1] * s[2] * s[3] * s[4]  + r2 * s[2] * s[3] * s[4]  + r3 * s[3] * s[4]  + r4 * s[4]  + r5;

    int data_index = pnts[index];

    double x1 = dd1 + r1 * dd3;
    double x2 = dd4 + r2 * dd6;
    double x3 = dd7 + r3 * dd9;
    double x4 = dd10 + r4 * dd12;
    double x5 = dd13 + r5 * dd15;
    
    for (int r6 = 0; r6 < s[5]; ++r6) {
        double x6 = dd16 + r6 * dd18;

        for (int r7 = 0; r7 < s[6]; ++r7) {
            double x7 = dd19 + r7 * dd21;

            for (int r8 = 0; r8 < s[7]; ++r8) {
                double x8 = dd22 + r8 * dd24;

                for (int r9 = 0; r9 < s[8]; ++r9) {
                    double x9 = dd25 + r9 * dd27;

                    for (int r10 = 0; r10 < s[9]; ++r10) 
                    {
                        
                        double x10 = dd28 + r10 * dd30;
                    
                        double q1 = fabs(c11 * x1 + c12 * x2 + c13 * x3 + c14 * x4 +
                                                c15 * x5 + c16 * x6 + c17 * x7 + c18 * x8 +
                                                c19 * x9 + c110 * x10 - d1);

                        double q2 = fabs(c21 * x1 + c22 * x2 + c23 * x3 + c24 * x4 +
                                c25 * x5 + c26 * x6 + c27 * x7 + c28 * x8 +
                                c29 * x9 + c210 * x10 - d2);

                        double q3 = fabs(c31 * x1 + c32 * x2 + c33 * x3 + c34 * x4 +
                                c35 * x5 + c36 * x6 + c37 * x7 + c38 * x8 +
                                c39 * x9 + c310 * x10 - d3);

                        double q4 = fabs(c41 * x1 + c42 * x2 + c43 * x3 + c44 * x4 +
                                c45 * x5 + c46 * x6 + c47 * x7 + c48 * x8 +
                                c49 * x9 + c410 * x10 - d4);

                        double q5 = fabs(c51 * x1 + c52 * x2 + c53 * x3 + c54 * x4 +
                                c55 * x5 + c56 * x6 + c57 * x7 + c58 * x8 +
                                c59 * x9 + c510 * x10 - d5);

                        double q6 = fabs(c61 * x1 + c62 * x2 + c63 * x3 + c64 * x4 +
                                c65 * x5 + c66 * x6 + c67 * x7 + c68 * x8 +
                                c69 * x9 + c610 * x10 - d6);

                        double q7 = fabs(c71 * x1 + c72 * x2 + c73 * x3 + c74 * x4 +
                                c75 * x5 + c76 * x6 + c77 * x7 + c78 * x8 +
                                c79 * x9 + c710 * x10 - d7);

                        double q8 = fabs(c81 * x1 + c82 * x2 + c83 * x3 + c84 * x4 +
                                c85 * x5 + c86 * x6 + c87 * x7 + c88 * x8 +
                                c89 * x9 + c810 * x10 - d8);

                        double q9 = fabs(c91 * x1 + c92 * x2 + c93 * x3 + c94 * x4 +
                                c95 * x5 + c96 * x6 + c97 * x7 + c98 * x8 +
                                c99 * x9 + c910 * x10 - d9);

                        double q10 = fabs(c101 * x1 + c102 * x2 + c103 * x3 + c104 * x4 +
                                    c105 * x5 + c106 * x6 + c107 * x7 + c108 * x8 +
                                    c109 * x9 + c1010 * x10 - d10);

                        if ((q1 <= e[0]) && (q2 <= e[1]) && (q3 <= e[2]) &&
                            (q4 <= e[3]) && (q5 <= e[4]) && (q6 <= e[5]) &&
                            (q7 <= e[6]) && (q8 <= e[7]) && (q9 <= e[8]) &&
                            (q10 <= e[9])) 
                            {
                                my_pnts++;

                                if (!temp) continue;

                                int tmp = data_index * 10;
                                temp[tmp] = x1;
                                temp[tmp + 1] = x2;
                                temp[tmp + 2] = x3;
                                temp[tmp + 3] = x4;
                                temp[tmp + 4] = x5;
                                temp[tmp + 5] = x6;
                                temp[tmp + 6] = x7;
                                temp[tmp + 7] = x8;
                                temp[tmp + 8] = x9;
                                temp[tmp + 9] = x10;

                                data_index++;
                            }
                    }
    
                }
            }
        }
    }
    
    if (!temp) 
    {
        pnts[index] = my_pnts;
    }
}

int main() 
{
    int i, j;

    i = 0;
    FILE* fp = fopen("./disp.txt", "r");
    if (fp == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }

    while (!feof(fp)) {
        if (!fscanf(fp, "%lf", &a[i])) {
        printf("Error: fscanf failed while reading disp.txt\n");
        exit(EXIT_FAILURE);
        }
        i++;
    }
    fclose(fp);

    // read grid file
    j = 0;
    FILE* fpq = fopen("./grid.txt", "r");
    if (fpq == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }

    while (!feof(fpq)) {
        if (!fscanf(fpq, "%lf", &b[j])) {
        printf("Error: fscanf failed while reading grid.txt\n");
        exit(EXIT_FAILURE);
        }
        j++;
    }
    fclose(fpq);

    FILE* fptr = fopen("./results-v3.txt", "w");
    if (fptr == NULL) {
        printf("Error in creating file !");
        exit(1);
    }

    double kk = 0.3;

    float kernel_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int host_s[10];
    double host_e[10];

    for(int i=0;i<10;i++)
    {
        host_s[i] = floor((b[3*i + 1] - b[3*i]) / b[3*i + 2]);
        host_e[i] = kk * a[12 * i + 11];
    }

    dim3 grid_dim(host_s[0], host_s[1], host_s[2]);
    dim3 block_dim(host_s[3], host_s[4], 1);

    int NUM_THREADS = host_s[0] * host_s[1] * host_s[2] * host_s[3] * host_s[4]; 
    ull *dev_points;
    cudaCheckError(cudaMallocManaged(&dev_points, NUM_THREADS * sizeof(ull)));

    double *d_data = NULL;

    cudaCheckError(cudaEventRecord(start, 0));
    cudaCheckError(cudaMemcpyToSymbol(s, host_s, 10 * sizeof(int)));
    cudaCheckError(cudaMemcpyToSymbol(e, host_e, 10 * sizeof(double)));

    gridloopsearch<<<grid_dim, block_dim>>>(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                                            b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21],
                                            b[22], b[23], b[24], b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2],
                                            a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                                            a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
                                            a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33],
                                            a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                                            a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53],
                                            a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63],
                                            a[64], a[65], a[66], a[67], a[68], a[69], a[70], a[71], a[72], a[73],
                                            a[74], a[75], a[76], a[77], a[78], a[79], a[80], a[81], a[82], a[83],
                                            a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91], a[92], a[93],
                                            a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102], a[103],
                                            a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                                            a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk, dev_points, d_data);

    // calculate prefix sum

    cudaCheckError(cudaDeviceSynchronize());

    ull last_thread_points = dev_points[NUM_THREADS - 1];

    thrust::device_ptr<ull> thrust_dev_points(dev_points);
    thrust::exclusive_scan(thrust_dev_points, thrust_dev_points + NUM_THREADS, thrust_dev_points);

    ull curr_points =  dev_points[NUM_THREADS - 1];

    curr_points += last_thread_points;
    curr_points *= 10;

    cudaCheckError(cudaMallocManaged(&d_data, curr_points * sizeof(double)));

    gridloopsearch<<<grid_dim, block_dim>>>(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                                            b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21],
                                            b[22], b[23], b[24], b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2],
                                            a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                                            a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23],
                                            a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33],
                                            a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                                            a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53],
                                            a[54], a[55], a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63],
                                            a[64], a[65], a[66], a[67], a[68], a[69], a[70], a[71], a[72], a[73],
                                            a[74], a[75], a[76], a[77], a[78], a[79], a[80], a[81], a[82], a[83],
                                            a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91], a[92], a[93],
                                            a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102], a[103],
                                            a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                                            a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk, dev_points, d_data);

    cudaCheckError(cudaDeviceSynchronize());
    curr_points /= 10;
    for(int i=0;i<curr_points;i++)
    {
        for(int j=0;j<9;j++)
        {
            fprintf(fptr, "%lf\t", d_data[i*10 + j]);
        }
        fprintf(fptr, "%lf\n", d_data[i*10 + 9]);
    }

    cudaFree(d_data);
    d_data = NULL;

    cudaCheckError(cudaEventRecord(end, 0));
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaGetLastError());

    cudaEventElapsedTime(&kernel_time, start, end);
    fclose(fptr);
    std::cout << "Time for execution: " << kernel_time << std::endl;
    std::cout << "Points: " << curr_points << std::endl;

    cudaFree(d_data);
    cudaFree(dev_points);

    return EXIT_SUCCESS;
}