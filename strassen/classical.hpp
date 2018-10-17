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

void classical_mm(int*** C, int*** A, int*** B, const int& dim, const int& lv)
{
	for (int i = 0; i < dim; ++i) {
		fill(&C[lv][i][0], &C[lv][i][0]+dim, 0);

		for (int k = 0; k < dim; ++k)
			for (int j = 0; j < dim; ++j)
				C[lv][i][j] += A[lv][i][k] * B[lv][k][j];
	}
}
