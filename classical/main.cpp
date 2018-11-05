// #define DIM 256
#define MOD 1024

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif
using namespace std;

typedef long double ring;
typedef pair<int, int> pii;

#include "classical.hpp"


ring A[DIM][DIM];
ring B[DIM][DIM];
ring C[DIM][DIM];

int N_TEST = 20;


void init()
{
	for (int i = 0; i < DIM; ++i)
		for (int j = 0; j < DIM; ++j)
			A[i][j] = rand() % MOD;

	for (int i = 0; i < DIM; ++i)
		for (int j = 0; j < DIM; ++j)
			B[i][j] = rand() % MOD;

	fill(&C[0][0], &C[0][0]+DIM*DIM, 0);
}



int main(int argc, char** argv)
{
	srand(time(0));

	/* Warm-up */

	for (int cnt = 0; cnt < 10; ++cnt) {
		init();
		classical_mm_cache(pii(0,0), pii(0,0), DIM);
	}
	cout << "Warm-up done.\n" << endl;


	/* Test classical_mm_cache */

	clock_t tic, toc;
	double time_elapsed[300], avg_time = 0;

	for (int cnt = 0; cnt < N_TEST; ++cnt) {
		init();

		tic = clock();
		classical_mm_cache(pii(0,0), pii(0,0), DIM);
		toc = clock();

		time_elapsed[cnt] = (double)(toc-tic)/CLOCKS_PER_SEC;
		avg_time += time_elapsed[cnt];

		if ((cnt+1) % 100 == 0) cout << "Test " << cnt+1 << " done.\n";
	}

	cout << "classical_mm_cache test done.\n";
	cout << "Average time : " << avg_time/N_TEST << endl << endl;


	/* Test classical_mm_naive */

	avg_time = 0;
	for (int cnt = 0; cnt < N_TEST; ++cnt) {
		init();

		tic = clock();
		classical_mm_naive(pii(0,0), pii(0,0), DIM);
		toc = clock();

		time_elapsed[cnt] = (double)(toc-tic)/CLOCKS_PER_SEC;
		avg_time += time_elapsed[cnt];

		if ((cnt+1) % 100 == 0) cout << "Test " << cnt+1 << " done.\n";
	}

	cout << "classical_mm_naive test done.\n";
	cout << "Average time : " << avg_time/N_TEST << endl << endl;


	return 0;
}