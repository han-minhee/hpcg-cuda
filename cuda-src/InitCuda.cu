#include "InitCuda.cuh"
// macro ref:
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

int numSM;
int numSP;

int threadsPerBlock;
int regsPerBlock;
int multiProcessorCount;
int deviceWarpSize;

int getWarpSize(void){
  return deviceWarpSize;
}
int getThreadsPerBlock(void){
  return threadsPerBlock;
}
int getRegsPerBlock(void){
  return regsPerBlock;
}
int getMultiProcessorCount(void){
  return multiProcessorCount;
}

int gpuAssert(cudaError_t code) {
    if (code != cudaSuccess) {
      printf("GPUassert: %s\n", cudaGetErrorString(code));
      printf("Error Code: %d\n", code);
      return -1;
    } else {
      return 0;
    }
  }

int getDeviceProp(cudaDeviceProp deviceProp) {
  if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
    return 0;
  } else {
    return 1;
  }
}
int gpuCheckError(void){
  return gpuAssert(cudaPeekAtLastError());
}


int initDevice(void) {
  cudaDeviceProp deviceProp;

  int getDevicePropSuccess = getDeviceProp(deviceProp);
  if (!getDevicePropSuccess) {
    return -1;
  }

  deviceWarpSize = deviceProp.warpSize;
  regsPerBlock = deviceProp.regsPerBlock;
  multiProcessorCount = deviceProp.multiProcessorCount;
  return 0;
}