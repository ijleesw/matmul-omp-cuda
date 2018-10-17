#pragma once

#define THRESHOLD 64

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include "classical.hpp"

typedef pair<int, int> pii;

extern int A[MAX_DEPTH][DIM][DIM];
extern int B[MAX_DEPTH][DIM][DIM];
extern int C[MAX_DEPTH][DIM][DIM];

extern int M1[MAX_DEPTH][DIM][DIM];
extern int M2[MAX_DEPTH][DIM][DIM];
extern int M3[MAX_DEPTH][DIM][DIM];
extern int M4[MAX_DEPTH][DIM][DIM];
extern int M5[MAX_DEPTH][DIM][DIM];
extern int M6[MAX_DEPTH][DIM][DIM];
extern int M7[MAX_DEPTH][DIM][DIM];


/*
 * Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
 */
void strassen_mm(int (*C)[DIM][DIM], int (*A)[DIM][DIM], int (*B)[DIM][DIM], const pii& A_s, const pii& B_s, const pii& C_s, const int& dim, const int& lv)
{
	int dim_2 = dim/2;
	if (dim <= THRESHOLD) {
		classical_mm(C, A, B, A_s, B_s, C_s, dim, lv);
		return;
	}


	/**** M1 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		for (int j = A_s.second; j < A_s.second+dim_2; ++j)
			A[lv+1][i][j] = A[lv][i][j] + A[lv][i+dim_2][j+dim_2];

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		for (int j = B_s.second; j < B_s.second+dim_2; ++j)
			B[lv+1][i][j] = B[lv][i][j] + B[lv][i+dim_2][j+dim_2];

	strassen_mm(M1, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** M2 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		for (int j = A_s.second; j < A_s.second+dim_2; ++j)
			A[lv+1][i][j] = A[lv][i+dim_2][j] + A[lv][i+dim_2][j+dim_2];

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		copy(&B[lv][i][B_s.second], &B[lv][i][B_s.second]+dim_2, &B[lv+1][i][B_s.second]);

	strassen_mm(M2, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** M3 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		copy(&A[lv][i][A_s.second], &A[lv][i][A_s.second]+dim_2, &A[lv+1][i][A_s.second]);

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		for (int j = B_s.second; j < B_s.second+dim_2; ++j)
			B[lv+1][i][j] = B[lv][i][j+dim_2] - B[lv][i+dim_2][j+dim_2];

	strassen_mm(M3, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** M4 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		copy(&A[lv][i+dim_2][A_s.second+dim_2], &A[lv][i+dim_2][A_s.second+dim_2]+dim_2, &A[lv+1][i][A_s.second]);

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		for (int j = B_s.second; j < B_s.second+dim_2; ++j)
			B[lv+1][i][j] = B[lv][i+dim_2][j] - B[lv][i][j];

	strassen_mm(M4, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** M5 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		for (int j = A_s.second; j < A_s.second+dim_2; ++j)
			A[lv+1][i][j] = A[lv][i][j] + A[lv][i][j+dim_2];

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		copy(&B[lv][i+dim_2][B_s.second+dim_2], &B[lv][i+dim_2][B_s.second+dim_2]+dim_2, &B[lv+1][i][B_s.second]);

	strassen_mm(M5, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** M6 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		for (int j = A_s.second; j < A_s.second+dim_2; ++j)
			A[lv+1][i][j] = A[lv][i+dim_2][j] - A[lv][i][j];

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		for (int j = B_s.second; j < B_s.second+dim_2; ++j)
			B[lv+1][i][j] = B[lv][i][j] + B[lv][i][j+dim_2];

	strassen_mm(M6, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** M7 ****/
	for (int i = A_s.first; i < A_s.first+dim_2; ++i)
		for (int j = A_s.second; j < A_s.second+dim_2; ++j)
			A[lv+1][i][j] = A[lv][i][j+dim_2] - A[lv][i+dim_2][j+dim_2];

	for (int i = B_s.first; i < B_s.first+dim_2; ++i)
		for (int j = B_s.second; j < B_s.second+dim_2; ++j)
			B[lv+1][i][j] = B[lv][i+dim_2][j] + B[lv][i+dim_2][j+dim_2];

	strassen_mm(M7, A, B, A_s, B_s, pii(0,0), dim_2, lv+1);


	/**** C1 ****/
	for (int i = 0; i < dim_2; ++i)
		for (int j = 0; j < dim_2; ++j)
			C[lv][i+C_s.first][j+C_s.second] = M1[lv+1][i][j] + M4[lv+1][i][j] - M5[lv+1][i][j] + M7[lv+1][i][j];


	/**** C2 ****/
	for (int i = 0; i < dim_2; ++i)
		for (int j = 0; j < dim_2; ++j)
			C[lv][i+C_s.first][j+C_s.second+dim_2] = M3[lv+1][i][j] + M5[lv+1][i][j];


	/**** C3 ****/
	for (int i = 0; i < dim_2; ++i)
		for (int j = 0; j < dim_2; ++j)
			C[lv][i+C_s.first+dim_2][j+C_s.second] = M2[lv+1][i][j] + M4[lv+1][i][j];


	/**** C4 ****/
	for (int i = 0; i < dim_2; ++i)
		for (int j = 0; j < dim_2; ++j)
			C[lv][i+C_s.first+dim_2][j+C_s.second+dim_2] = M1[lv+1][i][j] - M2[lv+1][i][j] + M3[lv+1][i][j] + M6[lv+1][i][j];
}
