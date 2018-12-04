#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"
#include "cudaTimer.h"


#define BLOCK_SIZE 16
#define N_TEST 50
#define DEVICE_ID 0
#define MAX_DEPTH 20


cuDoubleComplex *h_A, *h_B, *matmulRef, *cublasRef;
cuDoubleComplex *d_A[MAX_DEPTH], *d_B[MAX_DEPTH], *d_C[MAX_DEPTH];
cuDoubleComplex *d_M1[MAX_DEPTH], *d_M2[MAX_DEPTH], *d_M3[MAX_DEPTH], *d_M4[MAX_DEPTH], *d_M5[MAX_DEPTH], *d_M6[MAX_DEPTH], *d_M7[MAX_DEPTH];


template <typename ring>
void fillMatrix(ring* arr, const int N)
{
	for (int i = 0; i < N; ++i)
	{
		arr[i].x = (rand() & 0xFF) / 10;
		arr[i].y = (rand() & 0xFF) / 10;
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
		double curr_diff = cuCabs(cuCsub(lhs[i], rhs[i]));
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
void classicalMatmul(ring* A, ring* B, ring* C, const int dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < dim && col < dim)
	{
		ring sum = make_cuDoubleComplex(0, 0);
		for (int k = 0; k < dim; ++k)
		{
			sum = cuCadd(sum, cuCmul(A[row*dim + k], B[k*dim + col]));
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
	ring one = make_cuDoubleComplex(1, 0);
	ring zero = make_cuDoubleComplex(0, 0);
	ring m_one = make_cuDoubleComplex(-1, 0);

	if (dim <= threshold)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((dim+BLOCK_SIZE-1)/BLOCK_SIZE, (dim+BLOCK_SIZE-1)/BLOCK_SIZE);
		classicalMatmul<ring><<< grid, block >>>(A, B, C, dim);
		// cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &one, B, dim, A, dim, &zero, C, dim);
		return;
	}


	/* M1 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &one, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &one, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M1[d+1], dim_2, d+1, threshold);

	/* M2 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2*dim, lda, &one, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &zero, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M2[d+1], dim_2, d+1, threshold);

	/* M3 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &zero, A, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2, lda, &m_one, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M3[d+1], dim_2, d+1, threshold);

	/* M4 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2*dim+dim_2, lda, &zero, A, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2*dim, lda, &m_one, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M4[d+1], dim_2, d+1, threshold);

	/* M5 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &one, A+dim_2, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2*dim+dim_2, lda, &zero, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M5[d+1], dim_2, d+1, threshold);

	/* M6 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2*dim, lda, &m_one, A, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &one, B+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M6[d+1], dim_2, d+1, threshold);

	/* M7 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2, lda, &m_one, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2*dim, lda, &one, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M7[d+1], dim_2, d+1, threshold);


	/* C1 */
	lda = dim, ldb = dim/2, ldc = dim;  // C = C + B
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C, lda, &one, d_M1[d+1], ldb, C, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &one, d_M4[d+1], ldb, C, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &m_one, d_M5[d+1], ldb, C, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &one, d_M7[d+1], ldb, C, ldc);

	/* C2 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C+dim_2, lda, &one, d_M3[d+1], ldb, C+dim_2, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2, lda, &one, d_M5[d+1], ldb, C+dim_2, ldc);

	/* C3 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C+dim_2*dim, lda, &one, d_M2[d+1], ldb, C+dim_2*dim, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim, lda, &one, d_M4[d+1], ldb, C+dim_2*dim, ldc);

	/* C4 */
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C+dim_2*dim+dim_2, lda, &one, d_M1[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim+dim_2, lda, &m_one, d_M2[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim+dim_2, lda, &one, d_M3[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim+dim_2, lda, &one, d_M6[d+1], ldb, C+dim_2*dim+dim_2, ldc);
}


void cublasMatmul(cublasHandle_t& handle, const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C, const int dim)
{
	int lda = dim, ldb = dim, ldc = dim;
	const int m = dim, n = dim, k = dim;
	const cuDoubleComplex a = make_cuDoubleComplex(1, 0);
	const cuDoubleComplex b = make_cuDoubleComplex(0, 0);
	const cuDoubleComplex *alpha = &a;
	const cuDoubleComplex *beta = &b;

	cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
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

	size_t nBytes = nDim * nDim * sizeof(cuDoubleComplex);

	h_A = (cuDoubleComplex*) malloc(nBytes);
	h_B = (cuDoubleComplex*) malloc(nBytes);
	matmulRef = (cuDoubleComplex*) malloc(nBytes);
	cublasRef = (cuDoubleComplex*) malloc(nBytes);

	srand(0);
	fillMatrix<cuDoubleComplex>(h_A, nDim*nDim);
	fillMatrix<cuDoubleComplex>(h_B, nDim*nDim);

	int depth, _dim = nDim;
	for (depth = 0; depth < MAX_DEPTH && _dim > 0; ++depth)
	{
		cudaMalloc((cuDoubleComplex**) &d_A[depth], _dim*_dim*sizeof(cuDoubleComplex));
		cudaMalloc((cuDoubleComplex**) &d_B[depth], _dim*_dim*sizeof(cuDoubleComplex));

		if (depth == 0)
		{
			cudaMalloc((cuDoubleComplex**) &d_C[depth], _dim*_dim*sizeof(cuDoubleComplex));
		}
		else
		{
			cudaMalloc((cuDoubleComplex**) &d_M1[depth], _dim*_dim*sizeof(cuDoubleComplex));
			cudaMalloc((cuDoubleComplex**) &d_M2[depth], _dim*_dim*sizeof(cuDoubleComplex));
			cudaMalloc((cuDoubleComplex**) &d_M3[depth], _dim*_dim*sizeof(cuDoubleComplex));
			cudaMalloc((cuDoubleComplex**) &d_M4[depth], _dim*_dim*sizeof(cuDoubleComplex));
			cudaMalloc((cuDoubleComplex**) &d_M5[depth], _dim*_dim*sizeof(cuDoubleComplex));
			cudaMalloc((cuDoubleComplex**) &d_M6[depth], _dim*_dim*sizeof(cuDoubleComplex));
			cudaMalloc((cuDoubleComplex**) &d_M7[depth], _dim*_dim*sizeof(cuDoubleComplex));
		}
		_dim /= 2;
	}

	cudaMemcpy(d_A[0], h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B[0], h_B, nBytes, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	CudaTimer ct;


	/* Prepare result */

	if (check)
	{
		cublasMatmul(handle, d_A[0], d_B[0], d_C[0], nDim);
		cudaMemcpy(cublasRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
	}


	/* Run classicalMatmul */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		classicalMatmul<cuDoubleComplex><<< grid, block >>>(d_A[0], d_B[0], d_C[0], nDim);
		cudaDeviceSynchronize();
	}
	ct.stop();
	printf("[classicalMatmul] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(matmulRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
		checkResult<cuDoubleComplex>(cublasRef, matmulRef, nDim, "classicalMatmul");
	}


	/* Run strassenMatmul */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		strassenMatmul<cuDoubleComplex>(handle, d_A[0], d_B[0], d_C[0], nDim, 0, threshold);
	}
	ct.stop();

	cudaMemcpy(matmulRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
	printf("[strassenMatmul] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(matmulRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
		checkResult<cuDoubleComplex>(cublasRef, matmulRef, nDim, "strassenMatmul");
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
	free(matmulRef);
	free(cublasRef);

	printf("Done.\n");

	return 0;
}
