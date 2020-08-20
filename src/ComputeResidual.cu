
#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include "ComputeResidual.cuh"

#include <cuda_runtime.h>

template <unsigned int BLOCKSIZE>
__device__ void reduce_max(local_int_t tid, double *data) {
  __syncthreads();

  if (BLOCKSIZE > 512) {
    if (tid < 512 && tid + 512 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 512]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 256) {
    if (tid < 256 && tid + 256 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 256]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 128) {
    if (tid < 128 && tid + 128 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 128]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 64) {
    if (tid < 64 && tid + 64 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 64]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 32) {
    if (tid < 32 && tid + 32 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 32]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 16) {
    if (tid < 16 && tid + 16 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 16]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 8) {
    if (tid < 8 && tid + 8 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 8]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 4) {
    if (tid < 4 && tid + 4 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 4]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 2) {
    if (tid < 2 && tid + 2 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 2]);
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 1) {
    if (tid < 1 && tid + 1 < BLOCKSIZE) {
      data[tid] = max(data[tid], data[tid + 1]);
    }
    __syncthreads();
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_residual_part1(local_int_t n, const double *__restrict__ v1,
                               const double *__restrict__ v2,
                               double *__restrict__ workspace) {
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
  local_int_t inc = gridDim.x * BLOCKSIZE;

  __shared__ double sdata[BLOCKSIZE];
  sdata[tid] = 0.0;

  for (local_int_t idx = gid; idx < n; idx += inc) {
    sdata[tid] = max(sdata[tid], fabs(v1[idx] - v2[idx]));
  }

  reduce_max<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_residual_part2(local_int_t n, double *__restrict__ workspace) {
  local_int_t tid = threadIdx.x;

  __shared__ double sdata[BLOCKSIZE];
  sdata[tid] = 0.0;

  for (local_int_t idx = tid; idx < n; idx += BLOCKSIZE) {
    sdata[tid] = max(sdata[tid], workspace[idx]);
  }

  __syncthreads();

  reduce_max<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[0] = sdata[0];
  }
}

int ComputeResidual(local_int_t n, const Vector &v1, const Vector &v2,
                    double &residual) {
  double *tmp = reinterpret_cast<double *>(workspace);

#define RES_DIM 256
  kernel_residual_part1<RES_DIM>
      <<<dim3(RES_DIM), dim3(RES_DIM)>>>(n, v1.d_values, v2.d_values, tmp);

  kernel_residual_part2<RES_DIM><<<dim3(1), dim3(RES_DIM)>>>(RES_DIM, tmp);
#undef RES_DIM

  double local_residual;
  cudaMemcpy(&local_residual, tmp, sizeof(double), cudaMemcpyDeviceToHost);

#ifndef HPCG_NO_MPI
  double global_residual = 0.0;
  MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  residual = global_residual;
#else
  residual = local_residual;
#endif

  return 0;
}
