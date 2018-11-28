#pragma once


#ifdef __APPLE__
#include "bits/stdc++.h"
#else
#include <bits/stdc++.h>
#endif


#ifdef OMP
#include <omp.h>
#endif


clock_t tic_c, toc_c;
double tic_d, toc_d;


#ifndef OMP
void tic() { tic_c = clock(); }
void toc() { toc_c = clock(); }
double get_elapsed_time() { return (double)(toc_c-tic_c) / CLOCKS_PER_SEC; }


#else
void tic() { tic_d = omp_get_wtime(); }
void toc() { toc_d = omp_get_wtime(); }
double get_elapsed_time() { return toc_d - tic_d; }


#endif
