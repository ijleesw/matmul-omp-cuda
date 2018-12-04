#pragma once

#include <ctime>
#include <omp.h>

double tic_d, toc_d;

void tic() { tic_d = omp_get_wtime(); }
void toc() { toc_d = omp_get_wtime(); }

double get_elapsed_time() { return toc_d - tic_d; }
