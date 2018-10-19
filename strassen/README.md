# Strassen Algorithm

A C++ Implementation of Strassen Algorithm with OpenMP.



## Usage

```bash
# Without OpenMP
$ g++ -O2 -DDIM=<dim_of_matrix> -DTHRESHOLD=<threshold> main.cpp -o test
$ ./test

# With OpenMP on Ubuntu
$ g++ -O2 -DDIM=<dim_of_matrix> -DTHRESHOLD=<threshold> main.cpp -o test -DOMP -fopenmp
$ ./test

# With OpenMP on MacOS
$ g++ -O2 -DDIM=<dim_of_matrix> -DTHRESHOLD=<threshold> main.cpp -o test -DOMP -Xpreprocessor -fopenmp -lomp
$ ./test
```



## Experimental Results

Thresholds for using classical matmul are set to 32 w/o OpenMP and 128 w/ OpenMP. (Thresholds are chosen with coarse hyperparameter search.) `long double` is used for the base ring of matrix. Results are averaged over 5 experiments.



On Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz (2 cores 4 threads) :

| Dim  | Strassen (Single) | Classical (Single) | Strassen (Multi) | Classical (Multi) |
| :--: | :---------------: | :----------------: | :--------------: | :---------------: |
| 512  |      0.13551      |      0.202261      |    0.0781379     |     0.0940104     |
| 1024 |      0.95808      |      1.65246       |     0.557298     |     0.730832      |
| 2048 |      6.75949      |      14.0989       |     3.98302      |      6.1555       |