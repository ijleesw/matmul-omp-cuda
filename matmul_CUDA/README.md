# Matrix Multiplication in CUDA

A CUDA implementation of naive matmul, 'shared' matmul, cublasSgemm(cublasDgemm) and strassen algorithm.





## Usage

```bash
$ nvcc -arch=<gpu_architecture> mm-float.cu cudaTimer.cc -o mm-float -lcublas
$ ./mm-float <dim_of_matrix> <check>
# mm-double is analogous to mm-float

$ nvcc -arch=<gpu_architecture> strassen-float.cu cudaTimer.cc -o strassen-float -lcublas
$ ./strassen-float <dim_of_matrix> <threshold> <check>
# strassen-double is analogous to strassen-float
```





## Experimental Results

GeForce GTX TITAN X is used for the experiment with `-arch=sm_52` option. Threshold for Strassen algorithm is set to 1024 in both data types with a coarse hyperparameter search. Results are averaged over 50 experiments. 

'Shared' matmul means a matrix multiplication using shared memory. Block size in shared memory is set to 16x16 in all experiments. See [here](http://cseweb.ucsd.edu/classes/wi12/cse260-a/Lectures/Lec08.pdf) for more information.



With `float` data type:

| Dimension | Naive MM | 'Shared' MM | cublasSgemm | Strassen (w/ 'shared') |
| :-------: | :------: | :---------: | :---------: | :--------------------: |
|    256    | 120.46us |   48.18us   |   14.62us   |           -            |
|    512    | 850.30us |  302.64us   |   92.29us   |           -            |
|   1024    | 6.972ms  |   2.004ms   |  358.40us   |           -            |
|   2048    | 50.745ms |  17.182ms   |   2.857ms   |        17.124ms        |
|   4096    | 497.00ms |  148.430ms  |  23.773ms   |       126.019ms        |
|   8192    | 5.6956s  |   1.2485s   |  208.507ms  |        958.83ms        |



With `double` data type:

| Dimension | Naive MM | 'Shared' MM | cublasDgemm | Strassen (w/ naive) |
| :-------: | :------: | :---------: | :---------: | :-----------------: |
|    256    | 263.05us |  305.71us   |  314.43us   |          -          |
|    512    | 1.6514ms |  2.2105ms   |  1.6622ms   |          -          |
|   1024    | 11.147ms |  15.180ms   |  11.943ms   |          -          |
|   2048    | 88.416ms |  121.91ms   |  88.793ms   |      76.319ms       |
|   4096    | 747.89ms |   1.0468s   |  699.253ms  |      598.50ms       |
|   8192    | 5.6286s  |   7.4545s   |   5.3832s   |      4.3077ms       |





## References

http://cseweb.ucsd.edu/classes/wi12/cse260-a/Lectures/Lec08.pdf

https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

https://github.com/paiweilai/strassen-cuda

