#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#include "cuPolynomial.h"
#include "utils.h"
#include "cudaTimer.h"


#define BLOCK_SIZE 16
#define N_TEST 50
#define DEVICE_ID 0
#define MAX_DEPTH 20


cuFloatPoly4 *h_A, *h_B, *strassenRef, *classicalRef;
cuFloatPoly4 *d_A[MAX_DEPTH], *d_B[MAX_DEPTH], *d_C[MAX_DEPTH];
cuFloatPoly4 *d_M1[MAX_DEPTH], *d_M2[MAX_DEPTH], *d_M3[MAX_DEPTH], *d_M4[MAX_DEPTH], *d_M5[MAX_DEPTH], *d_M6[MAX_DEPTH], *d_M7[MAX_DEPTH];


template <typename ring>
void fillMatrix(ring* arr, const int N)
{
	for (int i = 0; i < N; ++i)
	{
		arr[i].x = (rand() & 0xFF) / 10;
		arr[i].y = (rand() & 0xFF) / 10;
		arr[i].z = (rand() & 0xFF) / 10;
		arr[i].w = (rand() & 0xFF) / 10;
	}
}


template <typename ring>
void checkResult(ring* lhs, ring* rhs, const int dim, const char* name)
{
	double max_diff = 0;
	double avg_diff = 0;
	int max_idx = 0;

	for (int i = 0; i < dim*dim; ++i)
	{
		double curr_diff = LinfDistPoly4(lhs[i] - rhs[i]);
		avg_diff += curr_diff;
		if (curr_diff > max_diff)
		{
			max_diff = curr_diff;
			max_idx = i;
		}
	}
	avg_diff /= (dim*dim);

	printf("[%s] Avg diff is %.8lf. Max diff is %.8lf at index %d.\n", name, avg_diff, max_diff, max_idx);
}


template <typename ring>
__global__
void matadd(const int m, const int n, ring* A, const int lda, ring* B, const int ldb, ring* C, const int ldc)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
	{
		C[row*ldc + col] = A[row*lda + col] + B[row*ldb + col];
	}
}


template <typename ring>
__global__
void matsub(const int m, const int n, ring* A, const int lda, ring* B, const int ldb, ring* C, const int ldc)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
	{
		C[row*ldc + col] = A[row*lda + col] - B[row*ldb + col];
	}
}


template <typename ring>
__global__
void matcpy(const int m, const int n, ring* A, const int lda, ring* C, const int ldc)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
	{
		C[row*ldc + col] = A[row*lda + col];
	}
}


template <typename ring>
__global__
void classicalMatmul(ring* A, ring* B, ring* C, const int dim)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int gd = gridDim.x;

	__shared__ ring _A[BLOCK_SIZE][BLOCK_SIZE], _B[BLOCK_SIZE][BLOCK_SIZE];

	if (row < dim && col < dim)
	{
		ring sum = make_cuFloatPoly4(0, 0, 0, 0);
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


template <typename ring>
void strassenMatmul(cublasHandle_t& handle, ring* A, ring* B, ring* C, const int dim, const int d, const int threshold)
{
	const int dim_2 = dim/2;

	int lda = dim, ldb = dim, ldc = dim_2;
	int m = dim_2, n = dim_2;

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((dim+BLOCK_SIZE-1)/BLOCK_SIZE, (dim+BLOCK_SIZE-1)/BLOCK_SIZE);

	if (dim <= threshold)
	{
		classicalMatmul<ring><<< grid, block >>>(A, B, C, dim);
		return;
	}


	/* M1 */
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, A, lda, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, B, lda, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M1[d+1], dim_2, d+1, threshold);

	/* M2 */
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, A+dim_2*dim, lda, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, B, lda, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M2[d+1], dim_2, d+1, threshold);

	/* M3 */
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, A, lda, d_A[d+1], ldc);
	matsub<cuFloatPoly4><<< grid, block >>>(m, n, B+dim_2, lda, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M3[d+1], dim_2, d+1, threshold);

	/* M4 */
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, A+dim_2*dim+dim_2, lda, d_A[d+1], ldc);
	matsub<cuFloatPoly4><<< grid, block >>>(m, n, B+dim_2*dim, lda, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M4[d+1], dim_2, d+1, threshold);

	/* M5 */
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, A, lda, A+dim_2, ldb, d_A[d+1], ldc);
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, B+dim_2*dim+dim_2, lda, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M5[d+1], dim_2, d+1, threshold);

	/* M6 */
	matsub<cuFloatPoly4><<< grid, block >>>(m, n, A+dim_2*dim, lda, A, ldb, d_A[d+1], ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, B, lda, B+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M6[d+1], dim_2, d+1, threshold);

	/* M7 */
	matsub<cuFloatPoly4><<< grid, block >>>(m, n, A+dim_2, lda, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, B+dim_2*dim, lda, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M7[d+1], dim_2, d+1, threshold);


	/* C1 */
	lda = dim, ldb = dim/2, ldc = dim;  // C = C + B
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, d_M1[d+1], ldb, C, ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, C, lda, d_M4[d+1], ldb, C, ldc);
	matsub<cuFloatPoly4><<< grid, block >>>(m, n, C, lda, d_M5[d+1], ldb, C, ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, C, lda, d_M7[d+1], ldb, C, ldc);

	/* C2 */
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, d_M3[d+1], ldb, C+dim_2, ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, C+dim_2, lda, d_M5[d+1], ldb, C+dim_2, ldc);

	/* C3 */
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, d_M2[d+1], ldb, C+dim_2*dim, ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, C+dim_2*dim, lda, d_M4[d+1], ldb, C+dim_2*dim, ldc);

	/* C4 */
	matcpy<cuFloatPoly4><<< grid, block >>>(m, n, d_M1[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	matsub<cuFloatPoly4><<< grid, block >>>(m, n, C+dim_2*dim+dim_2, lda, d_M2[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, C+dim_2*dim+dim_2, lda, d_M3[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	matadd<cuFloatPoly4><<< grid, block >>>(m, n, C+dim_2*dim+dim_2, lda, d_M6[d+1], ldb, C+dim_2*dim+dim_2, ldc);
}


int main(int argc, char** argv)
{
	if (argc != 4)
	{
		printf("Usage: %s <dim> <threshold> <check>\n", argv[0]);
		exit(0);
	}


	/* Initialize */

	int nDim = atoi(argv[1]);
	int threshold = atoi(argv[2]);
	int check = atoi(argv[3]);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nDim+BLOCK_SIZE-1)/BLOCK_SIZE, (nDim+BLOCK_SIZE-1)/BLOCK_SIZE);

	assert(nDim >= threshold && threshold >= BLOCK_SIZE);

	setDeviceAndGetInfo(DEVICE_ID);

	size_t nBytes = nDim * nDim * sizeof(cuFloatPoly4);

	h_A = (cuFloatPoly4*) malloc(nBytes);
	h_B = (cuFloatPoly4*) malloc(nBytes);
	strassenRef = (cuFloatPoly4*) malloc(nBytes);
	classicalRef = (cuFloatPoly4*) malloc(nBytes);

	srand(0);
	fillMatrix<cuFloatPoly4>(h_A, nDim*nDim);
	fillMatrix<cuFloatPoly4>(h_B, nDim*nDim);

	int depth, _dim = nDim;
	for (depth = 0; depth < MAX_DEPTH && _dim > 0; ++depth)
	{
		cudaMalloc((cuFloatPoly4**) &d_A[depth], _dim*_dim*sizeof(cuFloatPoly4));
		cudaMalloc((cuFloatPoly4**) &d_B[depth], _dim*_dim*sizeof(cuFloatPoly4));

		if (depth == 0)
		{
			cudaMalloc((cuFloatPoly4**) &d_C[depth], _dim*_dim*sizeof(cuFloatPoly4));
		}
		else
		{
			cudaMalloc((cuFloatPoly4**) &d_M1[depth], _dim*_dim*sizeof(cuFloatPoly4));
			cudaMalloc((cuFloatPoly4**) &d_M2[depth], _dim*_dim*sizeof(cuFloatPoly4));
			cudaMalloc((cuFloatPoly4**) &d_M3[depth], _dim*_dim*sizeof(cuFloatPoly4));
			cudaMalloc((cuFloatPoly4**) &d_M4[depth], _dim*_dim*sizeof(cuFloatPoly4));
			cudaMalloc((cuFloatPoly4**) &d_M5[depth], _dim*_dim*sizeof(cuFloatPoly4));
			cudaMalloc((cuFloatPoly4**) &d_M6[depth], _dim*_dim*sizeof(cuFloatPoly4));
			cudaMalloc((cuFloatPoly4**) &d_M7[depth], _dim*_dim*sizeof(cuFloatPoly4));
		}
		_dim /= 2;
	}

	cudaMemcpy(d_A[0], h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B[0], h_B, nBytes, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	CudaTimer ct;


	/* Run classicalMatmul */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		classicalMatmul<cuFloatPoly4><<< grid, block >>>(d_A[0], d_B[0], d_C[0], nDim);
		cudaDeviceSynchronize();
	}
	ct.stop();
	printf("[classicalMatmul] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(classicalRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
	}


	/* Run strassenMatmul */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		strassenMatmul<cuFloatPoly4>(handle, d_A[0], d_B[0], d_C[0], nDim, 0, threshold);
	}
	ct.stop();
	printf("[strassenMatmul] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(strassenRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
		checkResult<cuFloatPoly4>(classicalRef, strassenRef, nDim, "strassenMatmul");
	}


	/* Free memory */

	cublasDestroy(handle);

	for (int i = 0; i < depth; ++i)
	{
		cudaFree(d_A[i]);
		cudaFree(d_B[i]);

		if (i == 0)
		{
			cudaFree(d_C[i]);
		}
		else
		{
			cudaFree(d_M1[i]);
			cudaFree(d_M2[i]);
			cudaFree(d_M3[i]);
			cudaFree(d_M4[i]);
			cudaFree(d_M5[i]);
			cudaFree(d_M6[i]);
			cudaFree(d_M7[i]);
		}
	}

	cudaDeviceReset();

	free(h_A);
	free(h_B);
	free(strassenRef);
	free(classicalRef);

	printf("Done.\n");

	return 0;
}
