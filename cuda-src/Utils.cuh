#include "cuda_runtime.h"
#include <cassert>
#include <cstdio>

#define debug_message true

#define printFileLine                                                          \
  if (debug_message)                                                           \
    printf("line passed %s %d \n", __FILE__, __LINE__);

#define checkGPUandPrintLine                                                   \
  if (debug_message) {                                                         \
    if (cudaPeekAtLastError() != cudaSuccess) {                                \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at line%s %d \n", __FILE__, __LINE__);                           \
    } else {                                                                   \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
    }                                                                          \
  }

#define vectorMemcpyFromDeviceToHost(vector)                                   \
  cudaMemcpy(vector.values, vector.d_values,                                   \
             sizeof(double) * vector.localLength, cudaMemcpyDeviceToHost);

#define vectorMemcpyFromHostToDevice(vector)                                   \
  cudaMemcpy(vector.d_values, vector.values,                                   \
             sizeof(double) * vector.localLength, cudaMemcpyHostToDevice);

int getWarpSize(void);
int getThreadsPerBlock(void);
int getRegsPerBlock(void);
int getMultiProcessorCount(void);

int gpuAssert(cudaError_t code);
int initDevice(void);
int gpuCheckError(void);
int getDeviceProp(cudaDeviceProp deviceProp);
