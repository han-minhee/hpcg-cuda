#include "InitCuda.cuh"
#include <cassert>
#include <cudart>

// SMs contain SPs
extern int numSM;
extern int numSP;

extern int threadsPerBlock;
extern int warpSize;
extern int regsPerBlock;
extern int multiProcessorCount;

inline int gpuAssert(cudaError_t code);
int initDevice(void);
int getDeviceProp(cudaDeviceProp deviceProp);