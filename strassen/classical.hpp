#pragma once

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif

typedef pair<int, int> pii;

extern int A[MAX_DEPTH][DIM][DIM];
extern int B[MAX_DEPTH][DIM][DIM];
extern int C[MAX_DEPTH][DIM][DIM];

void classical_mm(int (*C)[DIM][DIM], int (*A)[DIM][DIM], int (*B)[DIM][DIM], const pii& A_s, const pii& B_s, const pii& C_s, const int& dim, const int& lv)
{
	fill(&C[lv][0][0], &C[lv][0][0]+DIM*DIM, 0);
	for (int i = A_s.first; i < A_s.first+dim; ++i)
		for (int k = A_s.second; k < A_s.second+dim; ++k)
			for (int j = B_s.second; j < B_s.second+dim; ++j)
				C[lv][i][j] += A[lv][i][k] * B[lv][k][j];
}