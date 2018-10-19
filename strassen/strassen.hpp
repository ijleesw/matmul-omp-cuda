#pragma once

#ifndef THRESHOLD
#define THRESHOLD 128
#endif

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include "classical.hpp"


extern ring*** A;
extern ring*** B;
extern ring*** C;

extern ring*** M1;
extern ring*** M2;
extern ring*** M3;
extern ring*** M4;
extern ring*** M5;
extern ring*** M6;
extern ring*** M7;


/*
 * Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
 */
#ifndef OMP
void strassen_mm(ring*** C, ring*** A, ring*** B, const int& dim, const int& lv)
{
	int dim_2 = dim/2;
	if (dim <= THRESHOLD)
	{
		classical_mm(C, A, B, dim, lv);
		return;
	}


	/**** M1 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i][j] + A[lv][i+dim_2][j+dim_2];

	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i][j] + B[lv][i+dim_2][j+dim_2];

	strassen_mm(M1, A, B, dim_2, lv+1);


	/**** M2 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i+dim_2][j] + A[lv][i+dim_2][j+dim_2];

	for (int i = 0; i < dim_2; ++i)
		copy(&B[lv][i][0], &B[lv][i][0]+dim_2, &B[lv+1][i][0]);

	strassen_mm(M2, A, B, dim_2, lv+1);


	/**** M3 ****/
	for (int i = 0; i < dim_2; ++i)
		copy(&A[lv][i][0], &A[lv][i][0]+dim_2, &A[lv+1][i][0]);

	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i][j+dim_2] - B[lv][i+dim_2][j+dim_2];

	strassen_mm(M3, A, B, dim_2, lv+1);


	/**** M4 ****/
	for (int i = 0; i < dim_2; ++i)
		copy(&A[lv][i+dim_2][dim_2], &A[lv][i+dim_2][dim_2]+dim_2, &A[lv+1][i][0]);

	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i+dim_2][j] - B[lv][i][j];

	strassen_mm(M4, A, B, dim_2, lv+1);


	/**** M5 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i][j] + A[lv][i][j+dim_2];

	for (int i = 0; i < dim_2; ++i)
		copy(&B[lv][i+dim_2][dim_2], &B[lv][i+dim_2][dim_2]+dim_2, &B[lv+1][i][0]);

	strassen_mm(M5, A, B, dim_2, lv+1);


	/**** M6 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i+dim_2][j] - A[lv][i][j];

	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i][j] + B[lv][i][j+dim_2];

	strassen_mm(M6, A, B, dim_2, lv+1);


	/**** M7 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i][j+dim_2] - A[lv][i+dim_2][j+dim_2];

	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i+dim_2][j] + B[lv][i+dim_2][j+dim_2];

	strassen_mm(M7, A, B, dim_2, lv+1);


	/**** C1 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i][j] = M1[lv+1][i][j] + M4[lv+1][i][j] - M5[lv+1][i][j] + M7[lv+1][i][j];


	/**** C2 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i][j+dim_2] = M3[lv+1][i][j] + M5[lv+1][i][j];


	/**** C3 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i+dim_2][j] = M2[lv+1][i][j] + M4[lv+1][i][j];


	/**** C4 ****/
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i+dim_2][j+dim_2] = M1[lv+1][i][j] - M2[lv+1][i][j] + M3[lv+1][i][j] + M6[lv+1][i][j];
}


#else
void strassen_mm(ring*** C, ring*** A, ring*** B, const int& dim, const int& lv)
{
	int dim_2 = dim/2;
	if (dim <= THRESHOLD)
	{
		classical_mm(C, A, B, dim, lv);
		return;
	}


	/**** M1 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i][j] + A[lv][i+dim_2][j+dim_2];

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i][j] + B[lv][i+dim_2][j+dim_2];

	strassen_mm(M1, A, B, dim_2, lv+1);


	/**** M2 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i+dim_2][j] + A[lv][i+dim_2][j+dim_2];

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i)
		copy(&B[lv][i][0], &B[lv][i][0]+dim_2, &B[lv+1][i][0]);

	strassen_mm(M2, A, B, dim_2, lv+1);


	/**** M3 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i)
		copy(&A[lv][i][0], &A[lv][i][0]+dim_2, &A[lv+1][i][0]);

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i][j+dim_2] - B[lv][i+dim_2][j+dim_2];

	strassen_mm(M3, A, B, dim_2, lv+1);


	/**** M4 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i)
		copy(&A[lv][i+dim_2][dim_2], &A[lv][i+dim_2][dim_2]+dim_2, &A[lv+1][i][0]);

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i+dim_2][j] - B[lv][i][j];

	strassen_mm(M4, A, B, dim_2, lv+1);


	/**** M5 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i][j] + A[lv][i][j+dim_2];

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i)
		copy(&B[lv][i+dim_2][dim_2], &B[lv][i+dim_2][dim_2]+dim_2, &B[lv+1][i][0]);

	strassen_mm(M5, A, B, dim_2, lv+1);


	/**** M6 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i+dim_2][j] - A[lv][i][j];

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i][j] + B[lv][i][j+dim_2];

	strassen_mm(M6, A, B, dim_2, lv+1);


	/**** M7 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		A[lv+1][i][j] = A[lv][i][j+dim_2] - A[lv][i+dim_2][j+dim_2];

#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		B[lv+1][i][j] = B[lv][i+dim_2][j] + B[lv][i+dim_2][j+dim_2];

	strassen_mm(M7, A, B, dim_2, lv+1);


	/**** C1 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i][j] = M1[lv+1][i][j] + M4[lv+1][i][j] - M5[lv+1][i][j] + M7[lv+1][i][j];


	/**** C2 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i][j+dim_2] = M3[lv+1][i][j] + M5[lv+1][i][j];


	/**** C3 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i+dim_2][j] = M2[lv+1][i][j] + M4[lv+1][i][j];


	/**** C4 ****/
#pragma omp parallel for
	for (int i = 0; i < dim_2; ++i) for (int j = 0; j < dim_2; ++j)
		C[lv][i+dim_2][j+dim_2] = M1[lv+1][i][j] - M2[lv+1][i][j] + M3[lv+1][i][j] + M6[lv+1][i][j];
}


#endif
