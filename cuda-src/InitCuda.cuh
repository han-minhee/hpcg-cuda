#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstdio>

int getWarpSize(void);
int getThreadsPerBlock(void);
int getRegsPerBlock(void);
int getMultiProcessorCount(void);

int gpuAssert(cudaError_t code);
int initDevice(void);
int gpuCheckError(void);
int getDeviceProp(cudaDeviceProp deviceProp);