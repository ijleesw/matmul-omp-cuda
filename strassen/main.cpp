// #define DIM 1024
#define MAX_DEPTH 20  // Requirement: 2^MAX_DEPTH > DIM

#define MOD 10

#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif
using namespace std;

#ifdef OMP
#include <omp.h>
#endif

typedef long double ring;

#include "strassen.hpp"
#include "classical.hpp"
#include "utils.hpp"


ring*** A;
ring*** B;
ring*** C;

ring*** M1;
ring*** M2;
ring*** M3;
ring*** M4;
ring*** M5;
ring*** M6;
ring*** M7;


int N_WARMUP = 2;
int N_TEST = 5;


void init_memory()
{
	int dim = DIM;

	A = new ring**[MAX_DEPTH];
	B = new ring**[MAX_DEPTH];
	C = new ring**[MAX_DEPTH];

	M1 = new ring**[MAX_DEPTH];
	M2 = new ring**[MAX_DEPTH];
	M3 = new ring**[MAX_DEPTH];
	M4 = new ring**[MAX_DEPTH];
	M5 = new ring**[MAX_DEPTH];
	M6 = new ring**[MAX_DEPTH];
	M7 = new ring**[MAX_DEPTH];

	for (int i = 0; i < MAX_DEPTH && dim > 0; ++i)
	{
		A[i] = new ring*[dim];
		B[i] = new ring*[dim];

		for (int j = 0; j < dim; ++j) {
			A[i][j] = new ring[dim];
			B[i][j] = new ring[dim];
		}

		if (i == 0) {
			C[i] = new ring*[dim];

			for (int j = 0; j < dim; ++j) {
				C[i][j] = new ring[dim];
			}
		}

		else {  // i >= 1
			M1[i] = new ring*[dim];
			M2[i] = new ring*[dim];
			M3[i] = new ring*[dim];
			M4[i] = new ring*[dim];
			M5[i] = new ring*[dim];
			M6[i] = new ring*[dim];
			M7[i] = new ring*[dim];

			for (int j = 0; j < dim; ++j) {
				M1[i][j] = new ring[dim];
				M2[i][j] = new ring[dim];
				M3[i][j] = new ring[dim];
				M4[i][j] = new ring[dim];
				M5[i][j] = new ring[dim];
				M6[i][j] = new ring[dim];
				M7[i][j] = new ring[dim];
			}
		}

		dim /= 2;
	}
}

void reset()
{
	int dim = DIM;
	for (int i = 0; i < MAX_DEPTH && dim > 0; ++i)
	{
		for (int j = 0; j < dim; ++j) {
			for (int k = 0; k < dim; ++k)
			{
				// A[i][j][k] = rand() % MOD;
				// B[i][j][k] = rand() % MOD;
				A[i][j][k] = 1;
				B[i][j][k] = 1;

				if (i == 0) {
					C[i][j][k] = 0;
				}
				else {  // i >= 1
					M7[i][j][k] = M6[i][j][k] = M5[i][j][k] = M4[i][j][k] = M3[i][j][k] = M2[i][j][k] = M1[i][j][k] = 0;
				}
			}
		}

		dim /= 2;
	}
}


int main(int argc, char** argv)
{
	double time_elapsed[300], avg_time;

	init_memory();
	srand(time(0));


	/* Warm-up */

	for (int cnt = 0; cnt < N_WARMUP; ++cnt)
	{
		reset();
		strassen_mm(C, A, B, DIM, 0);
	}
	cout << "Warm-up done." << endl;


	/* Test strassen_mm */

	avg_time = 0;
	for (int cnt = 0; cnt < N_TEST; ++cnt)
	{
		reset();

		tic();
		strassen_mm(C, A, B, DIM, 0);
		toc();

		time_elapsed[cnt] = get_elapsed_time();
		avg_time += time_elapsed[cnt];
	}

	cout << "[strassen_mm]  Average time : " << avg_time/N_TEST << endl;


	/* Test classical_mm */

	avg_time = 0;
	for (int cnt = 0; cnt < N_TEST; ++cnt)
	{
		reset();

		tic();
		classical_mm(C, A, B, DIM, 0);
		toc();

		time_elapsed[cnt] = get_elapsed_time();
		avg_time += time_elapsed[cnt];
	}

	cout << "[classical_mm] Average time : " << avg_time/N_TEST << endl;


	return 0;
}