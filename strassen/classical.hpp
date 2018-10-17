#pragma once

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif

typedef pair<int, int> pii;

extern int*** A;
extern int*** B;
extern int*** C;

void classical_mm(int*** C, int*** A, int*** B, const pii& A_s, const pii& B_s, const pii& C_s, const int& dim, const int& lv)
{
	for (int i = 0; i < dim; ++i) fill(&C[lv][i][0], &C[lv][i][0]+dim, 0);
	for (int i = A_s.first; i < A_s.first+dim; ++i)
		for (int k = A_s.second; k < A_s.second+dim; ++k)
			for (int j = B_s.second; j < B_s.second+dim; ++j)
				C[lv][i][j] += A[lv][i][k] * B[lv][k][j];
}