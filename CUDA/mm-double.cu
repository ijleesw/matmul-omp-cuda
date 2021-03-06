#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"
#include "cudaTimer.h"


#define BLOCK_SIZE 16
#define N_TEST 50
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
void checkResult(ring* cublasRef, ring* gpuRef, const int dim, const char* name)
{
	double max_diff = 0;
	int max_idx = 0;

	for (int i = 0; i < dim*dim; ++i)
	{
		double curr_diff = abs(cublasRef[i] - gpuRef[i]);
		if (curr_diff > max_diff)
		{
			max_diff = curr_diff;
			max_idx = i;
		}
	}

	printf("[%s] Max difference is %.8lf at index %d.\n", name, max_diff, max_idx);
}


template <typename ring>
__global__
void naiveMatmulGPU(ring* A, ring* B, ring* C, const int dim)
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


template <typename ring>
__global__
void sharedMatmulGPU(ring* A, ring* B, ring* C, const int dim)
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


int main(int argc, char** argv)
{
	if (argc != 3)
	{
		printf("Usage: %s <dim> <check>\n", argv[0]);
		exit(0);
	}


	/* Initialize */

	int nDim = atoi(argv[1]);
	int check = atoi(argv[2]);

	assert(nDim >= BLOCK_SIZE);

	setDeviceAndGetInfo(DEVICE_ID);

	size_t nBytes = nDim * nDim * sizeof(double);

	double *h_A, *h_B, *cublasRef, *gpuRef;
	h_A = (double*) malloc(nBytes);
	h_B = (double*) malloc(nBytes);
	cublasRef = (double*) malloc(nBytes);
	gpuRef = (double*) malloc(nBytes);

	srand(0);
	fillMatrix<double>(h_A, nDim*nDim);
	fillMatrix<double>(h_B, nDim*nDim);

	memset(cublasRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	double *d_A, *d_B, *d_C;
	cudaMalloc((double**) &d_A, nBytes);
	cudaMalloc((double**) &d_B, nBytes);
	cudaMalloc((double**) &d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nDim+BLOCK_SIZE-1)/BLOCK_SIZE, (nDim+BLOCK_SIZE-1)/BLOCK_SIZE);

	CudaTimer ct;


	/* Run cublasDgemm */

	int lda = nDim, ldb = nDim, ldc = nDim;
	const int m = nDim, n = nDim, k = nDim;
	const double a = 1;
	const double b = 0;
	const double *alpha = &a;
	const double *beta = &b;

	cublasHandle_t handle;
	cublasCreate(&handle);

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_B, ldb, d_A, lda, beta, d_C, ldc);
	}
	ct.stop();
	printf("[cublasDgemm] %.5fms\n", ct.value()/N_TEST);

	cublasDestroy(handle);

	if (check)
	{
		cudaMemcpy(cublasRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	}


	/* Run naiveMatmulGPU */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		naiveMatmulGPU<double><<< grid, block >>>(d_A, d_B, d_C, nDim);
		cudaDeviceSynchronize();
	}
	ct.stop();
	printf("[naiveMatmulGPU] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		checkResult<double>(cublasRef, gpuRef, nDim, "naiveMatmulGPU");
	}


	/* Run sharedMatmulGPU */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		sharedMatmulGPU<double><<< grid, block >>>(d_A, d_B, d_C, nDim);
		cudaDeviceSynchronize();
	}
	ct.stop();
	printf("[sharedMatmulGPU] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		checkResult<double>(cublasRef, gpuRef, nDim, "sharedMatmulGPU");
	}


	/* Free memory */

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceReset();

	free(h_A);
	free(h_B);
	free(cublasRef);
	free(gpuRef);

	printf("Done.\n");

	return 0;
}
