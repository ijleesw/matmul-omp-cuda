#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#include "utils.h"


#define BLOCK_SIZE 16
#define N_TEST 10
#define DEVICE_ID 0


template <typename ring>
void fillMatrix(ring* arr, const int N)
{
	for (int i = 0; i < N; ++i)
	{
		arr[i] = (ring) (rand() & 0xFF) / 10;
	}
}


template <typename ring>
void checkResult(ring* hostRef, ring* gpuRef, const int dim, const char* name)
{
	double max_diff = 0;
	int max_idx = 0;

	for (int i = 0; i < dim*dim; ++i)
	{
		double curr_diff = abs(hostRef[i] - gpuRef[i]);
		if (curr_diff > max_diff)
		{
			max_diff = curr_diff;
			max_idx = i;
		}
	}

	printf("[%s] Max difference is %.8lf at index %d.\n", name, max_diff, max_idx);
}


template <typename ring>
void naive_mm_host(ring* A, ring* B, ring* C, const int dim)
{
	for (int i = 0; i < dim; ++i)
	{
		for (int j = 0; j < dim; ++j)
		{
			ring sum = 0;
			for (int k = 0; k < dim; ++k)
			{
				sum += A[i*dim+k] * B[k*dim+j];
			}
			C[i*dim+j] = sum;
		}
	}
}


template <typename ring>
__global__
void naive_mm_GPU(ring* A, ring* B, ring* C, const int dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < dim && col < dim)
	{
		ring sum = 0;
		for (int k = 0; k < dim; ++k)
		{
			sum += A[row*dim + k] * B[k*dim + col];
		}
		C[row*dim + col] = sum;
	}
}

/*
 * Reference: http://cseweb.ucsd.edu/classes/wi12/cse260-a/Lectures/Lec08.pdf
 */
template <typename ring>
__global__
void shared_mm_GPU(ring* A, ring* B, ring* C, const int dim)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int gd = gridDim.x;

	__shared__ ring _A[BLOCK_SIZE][BLOCK_SIZE], _B[BLOCK_SIZE][BLOCK_SIZE];

	if (row < dim && col < dim)
	{
		ring sum = 0;
		for (int k = 0; k < gd; ++k)
		{
			_A[threadIdx.y][threadIdx.x] = A[row*dim + k*BLOCK_SIZE + threadIdx.x];
			_B[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE+threadIdx.y) * dim + col];
			__syncthreads();

			for (int l = 0; l < BLOCK_SIZE; ++l)
			{
				sum += _A[threadIdx.y][l] * _B[l][threadIdx.x];
			}
			__syncthreads();
		}

		C[row*dim + col] = sum;
	}
}

/*
 * Reference: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
 */
void cublas_matmul(cublasHandle_t& handle, const float* A, const float* B, float* C, const int dim)
{
	int lda = dim, ldb = dim, ldc = dim;
	const int m = dim, n = dim, k = dim;
	const float a = 1;
	const float b = 0;
	const float *alpha = &a;
	const float *beta = &b;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
}


int main(int argc, char** argv)
{
	if (argc < 2 || argc > 3)
	{
		printf("Usage: %s <dim> <check (optional)>\n", argv[0]);
		exit(0);
	}


	/* Initialize */

	int nDim = atoi(argv[1]), nThreads = BLOCK_SIZE, nBlocks = (nDim+BLOCK_SIZE-1)/BLOCK_SIZE;
	int check = (argc == 3 ? atoi(argv[2]) : 0);

	setDeviceAndGetInfo(DEVICE_ID);

	size_t nBytes = nDim * nDim * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float*) malloc(nBytes);
	h_B = (float*) malloc(nBytes);
	hostRef = (float*) malloc(nBytes);
	gpuRef = (float*) malloc(nBytes);

	srand(0);
	fillMatrix<float>(h_A, nDim*nDim);
	fillMatrix<float>(h_B, nDim*nDim);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	float *d_A, *d_B, *d_C;
	cudaMalloc((float**) &d_A, nBytes);
	cudaMalloc((float**) &d_B, nBytes);
	cudaMalloc((float**) &d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 block(nThreads, nThreads);
	dim3 grid(nBlocks, nBlocks);


	/* Run naive_mm_host */

	if (check)
	{
		naive_mm_host<float>(h_A, h_B, hostRef, nDim);
	}


	/* Run GPU Naive-MM */

	for (int i = 0; i < N_TEST; ++i)
	{
		naive_mm_GPU<float><<< grid, block >>>(d_A, d_B, d_C, nDim);
		cudaDeviceSynchronize();
	}
	if (check)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		checkResult<float>(hostRef, gpuRef, nDim, "naive_mm_GPU");
	}


	/* Run GPU Shared-MM */

	for (int i = 0; i < N_TEST; ++i)
	{
		shared_mm_GPU<float><<< grid, block >>>(d_A, d_B, d_C, nDim);
		cudaDeviceSynchronize();
	}
	if (check)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		checkResult<float>(hostRef, gpuRef, nDim, "share_mm_GPU");
	}


	/* Run cublasSgemm */

	cublasHandle_t handle;
	cublasCreate(&handle);
	for (int i = 0; i < N_TEST; ++i)
	{
		cublas_matmul(handle, d_A, d_B, d_C, nDim);
	}
	cublasDestroy(handle);
	if (check)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		checkResult<float>(hostRef, gpuRef, nDim, "cublasSgemm");
	}


	/* Free memory */

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceReset();

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	printf("Done.\n");

	return 0;
}
