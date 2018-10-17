/*
 *  Compile with :
 *    $ clang++ -O2 -DDIM=1024 -DTHRESHOLD=32 main.cpp
 */
// #define DIM 1024
#define MAX_DEPTH 20  // Requirement: 2^MAX_DEPTH > DIM

#define MOD 10

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif
using namespace std;

#include "strassen.hpp"
#include "classical.hpp"

typedef pair<int, int> pii;


int*** A;
int*** B;
int*** C;

int*** M1;
int*** M2;
int*** M3;
int*** M4;
int*** M5;
int*** M6;
int*** M7;


int N_WARMUP = 2;
int N_TEST = 5;


void init()
{
	int dim = DIM;

	A = new int**[MAX_DEPTH];
	B = new int**[MAX_DEPTH];
	C = new int**[MAX_DEPTH];

	M1 = new int**[MAX_DEPTH];
	M2 = new int**[MAX_DEPTH];
	M3 = new int**[MAX_DEPTH];
	M4 = new int**[MAX_DEPTH];
	M5 = new int**[MAX_DEPTH];
	M6 = new int**[MAX_DEPTH];
	M7 = new int**[MAX_DEPTH];

	for (int i = 0; i < MAX_DEPTH && dim > 0; ++i) {
		A[i] = new int*[dim], B[i] = new int*[dim], C[i] = new int*[dim];
		M1[i] = new int*[dim], M2[i] = new int*[dim], M3[i] = new int*[dim];
		M4[i] = new int*[dim], M5[i] = new int*[dim], M6[i] = new int*[dim], M7[i] = new int*[dim];
		for (int j = 0; j < dim; ++j) {
			A[i][j] = new int[dim], B[i][j] = new int[dim], C[i][j] = new int[dim];
			M1[i][j] = new int[dim], M2[i][j] = new int[dim], M3[i][j] = new int[dim];
			M4[i][j] = new int[dim], M5[i][j] = new int[dim], M6[i][j] = new int[dim], M7[i][j] = new int[dim];
		}
		dim /= 2;
	}
}

void reset()
{
	int dim = DIM;
	for (int i = 0; i < MAX_DEPTH && dim > 0; ++i) {
		for (int j = 0; j < dim; ++j) {
			for (int k = 0; k < dim; ++k) {
				A[i][j][k] = rand() % MOD, B[i][j][k] = rand() % MOD;
				M7[i][j][k] = M6[i][j][k] = M5[i][j][k] = M4[i][j][k] = M3[i][j][k] = M2[i][j][k] = M1[i][j][k] = C[i][j][k] = 0;
			}
		}
		dim /= 2;
	}
}


int main(int argc, char** argv)
{
	init();
	srand(time(0));

	/* Warm-up */

	for (int cnt = 0; cnt < N_WARMUP; ++cnt) {
		reset();
		strassen_mm(C, A, B, pii(0,0), pii(0,0), pii(0,0), DIM, 0);
	}
	cout << "Warm-up done.\n" << endl;


	/* Test strassen_mm */

	clock_t tic, toc;
	double time_elapsed[300], avg_time = 0;

	for (int cnt = 0; cnt < N_TEST; ++cnt) {
		reset();

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
		reset();

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