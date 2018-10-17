# Strassen Algorithm

A C++ Implementation of Strassen Algorithm.



## Usage

```bash
$ g++ -O2 -DDIM=<dim_of_matrix> -DTHRESHOLD=<threshold> main.cpp -o test
$ ./test
```



## Experimental Results

Threshold is set to 32 in all experiments. (If curr_dim <= threshold, then use classical_mm.) Results are averaged over 5 experiments.



On Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz (macOS 10.13) :

| Dimension | Strassen | Classical |
| :-------: | :------: | :-------: |
|    512    | 0.028164 | 0.0031392 |
|   1024    | 1.40317  | 0.233193  |
|   2048    | 9.75089  |  2.07506  |
|   4096    | 68.9356  |  19.2629  |



On Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz (Ubuntu 16.04 LTS) :

| Dimension | Strassen | Classical |
| :-------: | :------: | :-------: |
|    512    | 0.216104 | 0.320117  |
|   1024    | 1.66836  |  2.54991  |
|   2048    | 10.8328  |  20.9902  |
|   4096    | 75.8845  |  166.865  |

