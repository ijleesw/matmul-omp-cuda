#pragma once


void classical_mm_1(ring*** C, ring*** A, ring*** B, const int& dim, const int& lv)
{
#pragma omp parallel for
	for (int i = 0; i < dim; ++i) {
		for (int j = 0; j < dim; ++j) {
			ring sum = 0;
			for (int k = 0; k < dim; ++k) {
				sum += A[lv][i][k] * B[lv][k][k];
			}
			C[lv][i][j] = sum;
		}
	}
}


void classical_mm_2(ring*** C, ring*** A, ring*** B, const int& dim, const int& lv)
{
#pragma omp parallel for
	for (int i = 0; i < dim; ++i) {
		fill(&C[lv][i][0], &C[lv][i][0]+dim, 0);

		for (int k = 0; k < dim; ++k)
			for (int j = 0; j < dim; ++j)
				C[lv][i][j] += A[lv][i][k] * B[lv][k][j];
	}
}
