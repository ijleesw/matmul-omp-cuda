# Classical Matrix Multiplication

A comparison of naive matmul and cache-efficient matmul.

## Usage

```bash
$ g++ -O2 -std=c++14 -DDIM=<dim_of_matrix> main.cpp -o test
$ ./test
```



## Experimental Results

Experimented on Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz. Results are averaged over 200 experiments.

| Dimension | Cache-Efficient (sec) | Naive (sec) |
| :-------: | :-------------------: | :---------: |
|     4     |        8.2e-07        |    8e-07    |
|     8     |        8.7e-07        |  8.05e-07   |
|    16     |       1.49e-06        |  1.285e-06  |
|    32     |       4.12e-06        |  1.315e-05  |
|    64     |       2.88e-05        |  0.000103   |
|    128    |      0.00030897       | 0.00156092  |
|    256    |      0.00234298       |  0.0119725  |
|    512    |       0.0199395       |  0.213967   |

