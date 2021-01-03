
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "nccl.h"

// Streams
extern cudaStream_t stream_interior;
extern cudaStream_t stream_halo;
// Workspace
extern void *workspace;
// Memory allocator

#define debug_message false

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define cudaRealloc(dst, temp, smallsize, size)                                \
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&temp, smallsize));                   \
  CUDA_CHECK_COMMAND(                                                          \
      cudaMemcpy(temp, dst, smallsize, cudaMemcpyDeviceToDevice));             \
  CUDA_CHECK_COMMAND(cudaFree(dst));                                           \
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&dst, size));                         \
  CUDA_CHECK_COMMAND(                                                          \
      cudaMemcpy(dst, temp, smallsize, cudaMemcpyDeviceToDevice));             \
  CUDA_CHECK_COMMAND(cudaFree(temp));

#define printFileLine                                                          \
  if (debug_message)                                                           \
    printf("line passed %s %d \n", __FILE__, __LINE__);

#define CUDA_CHECK_COMMAND(command)                                            \
  if (command != cudaSuccess) {                                                \
    printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                      \
           cudaGetErrorString(cudaPeekAtLastError()));                         \
    printf("at file, line %s %d \n", __FILE__, __LINE__);                      \
    exit(EXIT_FAILURE);                                                        \
  } else {                                                                     \
    if (debug_message)                                                         \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
  }

#define CUDA_RETURN_IFF_ERROR(command)                                         \
  if (command != cudaSuccess) {                                                \
    if (debug_message) {                                                       \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at file, line %s %d \n", __FILE__, __LINE__);                    \
    }                                                                          \
    return -1;                                                                 \
  } else {                                                                     \
    if (debug_message)                                                         \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
  }

#define CUDA_RETURN_VOID_IF_ERROR(command)                                     \
  if (command != cudaSuccess) {                                                \
    if (debug_message) {                                                       \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at file, line %s %d \n", __FILE__, __LINE__);                    \
    }                                                                          \
    return;                                                                    \
  } else {                                                                     \
    if (debug_message)                                                         \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
  }

#define CUDA_CHECK_ERROR                                                       \
  if (debug_message) {                                                         \
    if (cudaPeekAtLastError() != cudaSuccess) {                                \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at file, line %s %d \n", __FILE__, __LINE__);                    \
    } else {                                                                   \
      printf("line passed %s %d \n", __FILE__, __LINE__);                      \
    }                                                                          \
  }

#define CUDA_CHECK_ERROR_AND_RETURN_VOID                                       \
  if (debug_message) {                                                         \
    if (cudaPeekAtLastError() != cudaSuccess) {                                \
      printf("CUDA Error %d : %s\n", cudaPeekAtLastError(),                    \
             cudaGetErrorString(cudaPeekAtLastError()));                       \
      printf("at file, line %s %d \n", __FILE__, __LINE__);                    \
      return;                                                                  \
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

#define EXIT_IF_HPCG_ERROR(err)                                                \
  {                                                                            \
    if (err != 0) {                                                            \
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
