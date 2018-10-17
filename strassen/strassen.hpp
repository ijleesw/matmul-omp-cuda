#pragma once

// #define THRESHOLD 32

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include "classical.hpp"

typedef pair<int, int> pii;

extern int*** A;
extern int*** B;
extern int*** C;

extern int*** M1;
extern int*** M2;
extern int*** M3;
extern int*** M4;
extern int*** M5;
extern int*** M6;
extern int*** M7;


/*
 * Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
 */
void strassen_mm(int*** C, int*** A, int*** B, const int& dim, const int& lv)
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
