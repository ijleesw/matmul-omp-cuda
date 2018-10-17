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

# With OpenMP on macOS
$ g++ -O2 -DDIM=<dim_of_matrix> -DTHRESHOLD=<threshold> main.cpp -o test -DOMP -Xpreprocessor -fopenmp -lomp
$ ./test
```



## Experimental Results

Thresholds for using classical matmul are set to 32 w/o OpenMP and 128 w/ OpenMP. (Thresholds are chosen with coarse search.) Results are averaged over 5 experiments.



On Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz (macOS 10.13) :

| Dimension | Strassen | Classical | Strassen (OpenMP) |
| :-------: | :------: | :-------: | :---------------: |
|    512    | 0.199279 | 0.0226146 |     0.0304214     |
|   1024    | 1.40317  | 0.233193  |     0.243375      |
|   2048    | 9.75089  |  2.07506  |      1.5483       |
|   4096    | 68.9356  |  19.2629  |      10.6671      |



On Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz (Ubuntu 16.04 LTS) :

| Dimension | Strassen | Classical | Strassen (OpenMP) |
| :-------: | :------: | :-------: | :---------------: |
|    512    | 0.216104 | 0.320117  |     0.189048      |
|   1024    | 1.66836  |  2.54991  |     0.124042      |
|   2048    | 10.8328  |  20.9902  |     0.888446      |
|   4096    | 75.8845  |  166.865  |      6.22707      |
|   8192    |    -     |     -     |      43.9331      |

