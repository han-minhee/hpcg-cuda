
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

// Streams
extern cudaStream_t stream_interior;
extern cudaStream_t stream_halo;
// Workspace
extern void *workspace;
// Memory allocator

#define debug_message true

#define cudaHostRealloc(dst, temp, size)                                       \
  cudaMalloc((void **)&temp, sizeof(dst));                                     \
  cudaMemcpy(temp, dst, sizeof(dst), cudaMemcpyHostToHost);                    \
  cudaFree(dst);                                                               \
  cudaMalloc((void **)&dst, size);                                             \
  cudaMemcpy(dst, temp, sizeof(temp), cudaMemcpyHostToHost);                   \
  cudaFree(temp);

#define printFileLine                                                          \
  if (debug_message)                                                           \
    printf("line passed %s %d \n", __FILE__, __LINE__);

#define CUDA_CHECK_COMMAND(command)                                            \
  if (command != cudaSuccess) {                                                \
    if (debug_message) {                                                       \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at line%s %d \n", __FILE__, __LINE__);                           \
    }                                                                          \
  } else {                                                                     \
    if (debug_message)                                                         \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
  }

#define CUDA_RETURN_IFF_ERROR(command)                                         \
  if (command != cudaSuccess) {                                                \
    if (debug_message) {                                                       \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at line%s %d \n", __FILE__, __LINE__);                           \
    }                                                                          \
    return -1;                                                                 \
  } else {                                                                     \
    if (debug_message)                                                         \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
  }

#define CUDA_CHECK_ERROR                                                       \
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

#define NULL_CHECK(ptr)                                                        \
  {                                                                            \
    if (ptr == NULL) {                                                         \
      fprintf(stderr, "ERROR in file %s ; line %d\n", __FILE__, __LINE__);     \
                                                                               \
      cudaDeviceReset();                                                       \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CudaVectorCopyHostToDevice(vector)                                     \
  cudaMemcpy(vector.d_values, vector.values, sizeof(vector.values),            \
             cudaMemcpyHostToDevice);

#define CudaVectorCopyDeviceToHost(vector)                                     \
  cudaMemcpy(vector.values, vector.d_values, sizeof(vector.d_values),          \
             cudaMemcpyDeviceToHost);

#define RNG_SEED 0x586744
#define MAX_COLORS 128

int getWarpSize(void);
int getThreadsPerBlock(void);
int getRegsPerBlock(void);
int getMultiProcessorCount(void);

int gpuAssert(cudaError_t code);
int initDevice(void);
int gpuCheckError(void);
int getDeviceProp(cudaDeviceProp deviceProp);
