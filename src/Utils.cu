#include "Utils.cuh"

int numSM;
int numSP;

cudaDeviceProp deviceProp;
int threadsPerBlock;
int regsPerBlock;
int multiProcessorCount;
int deviceWarpSize;

int getWarpSize(void) { return deviceWarpSize; }
int getThreadsPerBlock(void) { return threadsPerBlock; }
int getRegsPerBlock(void) { return regsPerBlock; }
int getMultiProcessorCount(void) { return multiProcessorCount; }

int gpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    printf("CUDA Error (%d): %s\n", code, cudaGetErrorString(code));
    return -1;
  } else {
    return 0;
  }
}

int gpuCheckError(void) { return gpuAssert(cudaPeekAtLastError()); }

int initDevice(void) {
  
  if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
    printf("CUDA device property request failed\n");
    return -1;
  }

  deviceWarpSize = deviceProp.warpSize;
  regsPerBlock = deviceProp.regsPerBlock;
  multiProcessorCount = deviceProp.multiProcessorCount;
  printf("\n\n\n");
  printf("=== CUDA Platform Information ===\n");
  printf("deviceName : %s\n", deviceProp.name);
  printf("deviceWarpSize : %d\n", deviceWarpSize);
  printf("regsPerBlock : %d\n", regsPerBlock);
  printf("multiProcessorCount : %d\n", multiProcessorCount);
  printf("maxThreadsPerBlock : %d\n", deviceProp.maxThreadsPerBlock);

  printf("=== CUDA Platform Information ===\n\n\n");

  return 0;
}