#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif

extern int A[DIM][DIM];
extern int B[DIM][DIM];
extern int C[DIM][DIM];

typedef pair<int, int> pii;

void classical_mm_cache(const pii& A_s, const pii& B_s, const int& dim)
{
	/*
	 * A bit different from ordinary matmul due to cache efficiency
	 */
	for (int i = A_s.first; i < A_s.first+dim; ++i) {
		for (int k = A_s.second; k < A_s.second+dim; ++k) {
			for (int j = B_s.second; j < B_s.second+dim; ++j) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}


void classical_mm_naive(const pii& A_s, const pii& B_s, const int& dim)
{
	/*
	 * Naive matmul
	 */
	for (int i = A_s.first; i < A_s.first+dim; ++i) {
		for (int j = B_s.second; j < B_s.second+dim; ++j) {
			int tmp = 0;
			for (int k = A_s.second; k < A_s.second+dim; ++k) {
				tmp += A[i][k]*B[k][j];
			}
			C[i][j] = tmp;
		}
	}
}