#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstdio>

// SMs contain SPs
int numSM;
int numSP;

int threadsPerBlock;
int regsPerBlock;
int multiProcessorCount;
int deviceWarpSize;

int gpuAssert(cudaError_t code);
int initDevice(void);
int getDeviceProp(cudaDeviceProp deviceProp);