#define DIM 512
#define MOD 10
#define MAX_DEPTH 14

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif
using namespace std;

#include "strassen.hpp"
#include "classical.hpp"

typedef pair<int, int> pii;

int A[MAX_DEPTH][DIM][DIM];
int B[MAX_DEPTH][DIM][DIM];
int C[MAX_DEPTH][DIM][DIM];

int M1[MAX_DEPTH][DIM][DIM];
int M2[MAX_DEPTH][DIM][DIM];
int M3[MAX_DEPTH][DIM][DIM];
int M4[MAX_DEPTH][DIM][DIM];
int M5[MAX_DEPTH][DIM][DIM];
int M6[MAX_DEPTH][DIM][DIM];
int M7[MAX_DEPTH][DIM][DIM];

int N_WARMUP = 5;
int N_TEST = 10;

void init()
{
	for (int i = 0; i < DIM; ++i)
		for (int j = 0; j < DIM; ++j)
			A[0][i][j] = rand() % MOD;

	for (int i = 0; i < DIM; ++i)
		for (int j = 0; j < DIM; ++j)
			B[0][i][j] = rand() % MOD;

	fill(&C[0][0][0], &C[0][0][0]+MAX_DEPTH*DIM*DIM, 0);

	fill(&M1[0][0][0], &M1[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
	fill(&M2[0][0][0], &M2[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
	fill(&M3[0][0][0], &M3[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
	fill(&M4[0][0][0], &M4[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
	fill(&M5[0][0][0], &M5[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
	fill(&M6[0][0][0], &M6[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
	fill(&M7[0][0][0], &M7[0][0][0]+MAX_DEPTH*DIM*DIM, 0);
}


int main(int argc, char** argv)
{
	srand(time(0));

	/* Warm-up */

	for (int cnt = 0; cnt < N_WARMUP; ++cnt) {
		init();
		strassen_mm(C, A, B, pii(0,0), pii(0,0), pii(0,0), DIM, 0);
	}
	cout << "Warm-up done.\n" << endl;


	/* Test strassen_mm */

	clock_t tic, toc;
	double time_elapsed[300], avg_time = 0;

	for (int cnt = 0; cnt < N_TEST; ++cnt) {
		init();

		tic = clock();
		strassen_mm(C, A, B, pii(0,0), pii(0,0), pii(0,0), DIM, 0);
		toc = clock();

		time_elapsed[cnt] = (double)(toc-tic)/CLOCKS_PER_SEC;
		avg_time += time_elapsed[cnt];

		if ((cnt+1) % 100 == 0) cout << "Test " << cnt+1 << " done.\n";
	}

	cout << "strassen_mm test done.\n";
	cout << "Average time : " << avg_time/N_TEST << endl << endl;


	/* Test classical_mm */

	avg_time = 0;
	for (int cnt = 0; cnt < N_TEST; ++cnt) {
		init();

		tic = clock();
		classical_mm(C, A, B, pii(0,0), pii(0,0), pii(0,0), DIM, 0);
		toc = clock();

		time_elapsed[cnt] = (double)(toc-tic)/CLOCKS_PER_SEC;
		avg_time += time_elapsed[cnt];

		if ((cnt+1) % 100 == 0) cout << "Test " << cnt+1 << " done.\n";
	}

	cout << "classical_mm test done.\n";
	cout << "Average time : " << avg_time/N_TEST << endl << endl;


	return 0;
}