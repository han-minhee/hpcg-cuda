#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstdio>

// SMs contain SPs
extern int numSM;
extern int numSP;

extern int threadsPerBlock;
extern int regsPerBlock;
extern int multiProcessorCount;
extern int deviceWarpSize;

int gpuAssert(cudaError_t code);
int initDevice(void);
int getDeviceProp(cudaDeviceProp deviceProp);