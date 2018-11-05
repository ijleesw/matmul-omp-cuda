# Classical Matrix Multiplication

A comparison of naive matmul and cache-efficient matmul.



## Usage

```bash
$ g++ -O2 -std=c++14 -DDIM=<dim_of_matrix> main.cpp -o test
$ ./test
```



## Experimental Results

Results are averaged over 200 experiments. 32bits integer is used for the base ring.



On Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz (macOS 10.13) :

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



On Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz (Ubuntu 16.04 LTS) :

| Dimension | Cache-Efficient (sec) | Naive (sec) |
| :-------: | :-------------------: | :---------: |
|    32     |      5.0415e-05       |  3.249e-05  |
|    64     |      0.000415865      | 0.00026043  |
|    128    |      0.00244401       | 0.00206899  |
|    256    |       0.0182932       |  0.0307875  |
|    512    |       0.144318        |  0.282904   |

