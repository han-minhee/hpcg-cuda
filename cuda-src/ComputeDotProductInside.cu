#ifndef HPCG_NO_MPI
#include "../src/mytimer.hpp"
#include <mpi.h>
#endif

#include "ComputeDotProductInside.cuh"
#include "Utils.cuh"

#include <cuda_runtime.h>

__device__ void kernelDeviceReduceSum(local_int_t tid, double *data) {
  unsigned int BLOCKSIZE = blockDim.x;
  __syncthreads();

  if (BLOCKSIZE > 512) {
    if (tid < 512 && tid + 512 < BLOCKSIZE) {
      data[tid] += data[tid + 512];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 256) {
    if (tid < 256 && tid + 256 < BLOCKSIZE) {
      data[tid] += data[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 128) {
    if (tid < 128 && tid + 128 < BLOCKSIZE) {
      data[tid] += data[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 64) {
    if (tid < 64 && tid + 64 < BLOCKSIZE) {
      data[tid] += data[tid + 64];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 32) {
    if (tid < 32 && tid + 32 < BLOCKSIZE) {
      data[tid] += data[tid + 32];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 16) {
    if (tid < 16 && tid + 16 < BLOCKSIZE) {
      data[tid] += data[tid + 16];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 8) {
    if (tid < 8 && tid + 8 < BLOCKSIZE) {
      data[tid] += data[tid + 8];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 4) {
    if (tid < 4 && tid + 4 < BLOCKSIZE) {
      data[tid] += data[tid + 4];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 2) {
    if (tid < 2 && tid + 2 < BLOCKSIZE) {
      data[tid] += data[tid + 2];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 1) {
    if (tid < 1 && tid + 1 < BLOCKSIZE) {
      data[tid] += data[tid + 1];
    }
    __syncthreads();
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_dot1_part1(local_int_t n, const double *x, double *workspace) {
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
  local_int_t inc = gridDim.x * BLOCKSIZE;

  double sum = 0.0;
  for (local_int_t idx = gid; idx < n; idx += inc) {
    double val = x[idx];
    sum = fma(val, val, sum);
  }

  __shared__ double sdata[BLOCKSIZE /*BLOCKSIZE*/];
  sdata[tid] = sum;

  kernelDeviceReduceSum(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_dot2_part1(local_int_t n, const double *x, const double *y,
                           double *workspace) {
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
  local_int_t inc = gridDim.x * BLOCKSIZE;

  double sum = 0.0;
  for (local_int_t idx = gid; idx < n; idx += inc) {
    sum = fma(y[idx], x[idx], sum);
  }

  __shared__ double sdata[BLOCKSIZE /*BLOCKSIZE*/];
  sdata[tid] = sum;

  kernelDeviceReduceSum(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_dot_part2(double *workspace) {
  __shared__ double sdata[BLOCKSIZE /*BLOCKSIZE*/];
  sdata[threadIdx.x] = workspace[threadIdx.x];

  kernelDeviceReduceSum(threadIdx.x, sdata);

  if (threadIdx.x == 0) {
    workspace[0] = sdata[0];
  }
}

int ComputeDotProductInside(local_int_t n, const Vector &x, const Vector &y,
                            double &result, double &time_allreduce,
                            bool &isOptimized) {
  assert(x.localLength >= n);
  assert(y.localLength >= n);

  double *tmp = reinterpret_cast<double *>(workspace);

#define DOT_DIM 256
  dim3 dot_blocks(DOT_DIM);
  dim3 dot_threads(DOT_DIM);

  if (x.d_values == y.d_values) {
    kernel_dot1_part1<DOT_DIM><<<dot_blocks, dot_threads>>>(n, x.d_values, tmp);
  } else {
    kernel_dot2_part1<DOT_DIM>
        <<<dot_blocks, dot_threads>>>(n, x.d_values, y.d_values, tmp);
  }
  kernel_dot_part2<DOT_DIM><<<dim3(1), dot_threads>>>(tmp);
#undef DOT_DIM

  double local_result;
  CUDA_CHECK_COMMAND(
      cudaMemcpy(&local_result, tmp, sizeof(double), cudaMemcpyDeviceToHost));

#ifndef HPCG_NO_MPI
  double t0 = mytimer();
  double global_result = 0.0;

  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  result = local_result;
#endif

  return 0;
}
