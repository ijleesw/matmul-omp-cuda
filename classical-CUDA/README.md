# Matrix Multiplication in CUDA

An implementation and comparison of naive matmul, 'shared' matmul and cublasSgemm(cublasDgemm).





## Usage

```bash
$ nvcc -arch=<gpu_architecture> mm-float.cu -o mm-float -lcublas
$ nvcc -arch=<gpu_architecture> mm-double.cu -o mm-double -lcublas
$ nvprof ./mm-float <dim_of_matrix>
$ nvprof ./mm-double <dim_of_matrix>
```

Use `./mm-float <dim_of_matrix> 1` if you want to check correctness.





## Experimental Results

Results are averaged over 10 experiments. GeForce GTX TITAN X is used for the experiment with `-arch=sm_52` option.

'Shared' matmul means a matrix multiplication using shared memory. Block size in shared memory is set to 16x16 in all experiments. See [here](http://cseweb.ucsd.edu/classes/wi12/cse260-a/Lectures/Lec08.pdf) for more information.



With `float` data type:

| Dimension | Naive MM | 'Shared' MM | cublasSgemm |
| :-------: | :------: | :---------: | :---------: |
|    256    | 112.42us |  41.400us   |  11.843us   |
|    512    | 841.29us |  295.14us   |  89.776us   |
|   1024    | 7.4259ms |  2.2608ms   |  399.05us   |
|   2048    | 50.219ms |  17.031ms   |  2.7991ms   |
|   4096    | 479.67ms |  136.61ms   |  20.126ms   |
|   8192    | 5.67334s |  1.24943s   |  207.43ms   |
|   16384   | 53.7824s |  11.1537s   |  1.75428s   |



With `double` data type:

| Dimension | Naive MM | 'Shared' MM | cublasDgemm |
| :-------: | :------: | :---------: | :---------: |
|    256    | 247.04us |  298.55us   |  312.77us   |
|    512    | 16.467ms |  2.3271ms   |  1.8489ms   |
|   1024    | 12.097ms |  15.955ms   |  11.975ms   |
|   2048    | 89.551ms |  120.87ms   |  87.764ms   |
|   4096    | 708.91ms |  975.03ms   |  692.37ms   |
|   8192    | 6.28788s |  8.35359s   |  5.79862s   |

