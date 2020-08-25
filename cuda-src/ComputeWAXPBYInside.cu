#ifndef HPCG_NO_MPI
#include "mytimer.hpp"
#include <mpi.h>
#endif

#include <cassert>
#include <cuda_runtime.h>

#include "ComputeWAXPBYInside.cuh"

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_waxpby(local_int_t size, double alpha,
                       const double * x, double beta,
                       const double * y, double * w) {
  local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) {
    return;
  }

  if (alpha == 1.0) {
    w[gid] = fma(beta, y[gid], x[gid]);
  } else if (beta == 1.0) {
    w[gid] = fma(alpha, x[gid], y[gid]);
  } else {
    w[gid] = fma(alpha, x[gid], beta * y[gid]);
  }
}

int ComputeWAXPBY(local_int_t n, double alpha, const Vector &x, double beta,
                  const Vector &y, Vector &w, bool &isOptimized) {
  assert(x.localLength >= n);
  assert(y.localLength >= n);
  assert(w.localLength >= n);

  kernel_waxpby<512><<<dim3((n - 1) / 512 + 1), dim3(512)>>>(
      n, alpha, x.d_values, beta, y.d_values, w.d_values);

  return 0;
}

template <unsigned int BLOCKSIZE>
__device__ void kernelDeviceReduceSum(local_int_t tid, double *data) {
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
    void kernel_fused_waxpby_dot_part1(local_int_t size, double alpha,
                                       const double * x,
                                       double * y,
                                       double * workspace) {
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * blockDim.x + tid;

  __shared__ double sdata[BLOCKSIZE];
  sdata[tid] = 0.0;

  for (local_int_t idx = gid; idx < size; idx += gridDim.x * blockDim.x) {
    double val = fma(alpha, x[idx], y[idx]);

    y[idx] = val;
    sdata[tid] = fma(val, val, sdata[tid]);
  }

  kernelDeviceReduceSum<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_fused_waxpby_dot_part2(local_int_t size,
                                       double * workspace) {
  local_int_t tid = threadIdx.x;

  __shared__ double sdata[BLOCKSIZE];
  sdata[tid] = 0.0;

  for (local_int_t idx = tid; idx < size; idx += BLOCKSIZE) {
    sdata[tid] += workspace[idx];
  }

  __syncthreads();

  kernelDeviceReduceSum<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[0] = sdata[0];
  }
}

int ComputeFusedWAXPBYDot(local_int_t n, double alpha, const Vector &x,
                          Vector &y, double &result, double &time_allreduce) {
  assert(x.localLength >= n);
  assert(y.localLength >= n);

  double *tmp = reinterpret_cast<double *>(workspace);

#define WAXPBY_DOT_DIM 256
  kernel_fused_waxpby_dot_part1<WAXPBY_DOT_DIM>
      <<<dim3(WAXPBY_DOT_DIM), dim3(WAXPBY_DOT_DIM)>>>(n, alpha, x.d_values,
                                                       y.d_values, tmp);

  kernel_fused_waxpby_dot_part2<WAXPBY_DOT_DIM>
      <<<dim3(1), dim3(WAXPBY_DOT_DIM)>>>(WAXPBY_DOT_DIM, tmp);
#undef WAXPBY_DOT_DIM

  double local_result;
  CUDA_CHECK_COMMAND(
      cudaMemcpy(&local_result, tmp, sizeof(double), cudaMemcpyDeviceToHost));

#ifndef HPCG_NO_MPI
  double t0 = mytimer();
  double global_result = 0.0;

  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  result = local_result;
#endif

  return 0;
}
