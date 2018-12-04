#ifndef UTILS_H_
#define UTILS_H_


#include <cuda_runtime.h>
#include <stdio.h>


#define CHECK_(call)	\
{	\
	const cudaError_t error = call;	\
	if (error != cudaSuccess)	\
	{	\
		printf("Error: %s:%d, ", __FILE__, __LINE__);	\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));	\
		fflush(stdout);		\
		exit(-10*error);	\
	}	\
}


#define setDeviceAndGetInfo(dev)	\
{	\
	cudaDeviceProp deviceProp;	\
	CHECK_(cudaGetDeviceProperties(&deviceProp, dev));	\
	printf("Using Device %d: %s\n", dev, deviceProp.name);	\
	CHECK_(cudaSetDevice(dev));	\
	fflush(stdout);	\
}


#endif
