# Matrix Multiplication in OpenMP

An OpenMP Implementation of Classical and Strassen Matrix Multiplication.



## Usage

```bash
# On MacOS
$ g++ -O2 -std=c++11 mm-real.cpp -o mm-real -Xpreprocessor -fopenmp -lomp
$ ./mm-real <dim> <threshold>
# Other mm-*'s are analogous to mm-real
```
Data type can be changed by modifying `typedef` in `mm-*.cpp`.



## Experimental Results

Classical matmul is implemented in cache-efficient version. Thresholds in Strassen for using classical matmul are set to 32 w/o OpenMP and 128 w/ OpenMP (Thresholds are chosen with coarse hyperparameter search). `long double` is used for the base ring of matrix. Results are averaged over 5 experiments.



On Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz (unit is second) :

| Dim  | Classical | Strassen  |
| :--: | :-------: | :-------: |
| 512  | 0.0940104 | 0.0781379 |
| 1024 | 0.730832  | 0.557298  |
| 2048 |  6.1555   |  3.98302  |