#include "ComputeDotProduct_cuda.cuh"

__device__ void reduce_sum(local_int_t tid, double *data) {
  size_t BLOCK_SIZE = blockDim.x;
  __syncthreads();

  if (BLOCK_SIZE > 512) {
    if (tid < 512 && tid + 512 < BLOCK_SIZE) {
      data[tid] += data[tid + 512];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 256) {
    if (tid < 256 && tid + 256 < BLOCK_SIZE) {
      data[tid] += data[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 128) {
    if (tid < 128 && tid + 128 < BLOCK_SIZE) {
      data[tid] += data[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 64) {
    if (tid < 64 && tid + 64 < BLOCK_SIZE) {
      data[tid] += data[tid + 64];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 32) {
    if (tid < 32 && tid + 32 < BLOCK_SIZE) {
      data[tid] += data[tid + 32];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 16) {
    if (tid < 16 && tid + 16 < BLOCK_SIZE) {
      data[tid] += data[tid + 16];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 8) {
    if (tid < 8 && tid + 8 < BLOCK_SIZE) {
      data[tid] += data[tid + 8];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 4) {
    if (tid < 4 && tid + 4 < BLOCK_SIZE) {
      data[tid] += data[tid + 4];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 2) {
    if (tid < 2 && tid + 2 < BLOCK_SIZE) {
      data[tid] += data[tid + 2];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE > 1) {
    if (tid < 1 && tid + 1 < BLOCK_SIZE) {
      data[tid] += data[tid + 1];
    }
    __syncthreads();
  }
}

__global__ void kernel_dodProduct1(local_int_t n, const double *x,
                                   double *workspace) {
  size_t BLOCK_SIZE = 32;
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCK_SIZE + tid;
  local_int_t inc = gridDim.x * BLOCK_SIZE;

  double sum = 0.0;
  for (local_int_t idx = gid; idx < n; idx += inc) {
    double val = x[idx];
    sum = fma(val, val, sum);
  }
  // shared
  double sdata[32];
  sdata[tid] = sum;

  reduce_sum(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

__global__ void kernel_dodProduct2(local_int_t n, const double *x,
                                   const double *y, double *workspace) {
  size_t BLOCK_SIZE = 32;
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCK_SIZE + tid;
  local_int_t inc = gridDim.x * BLOCK_SIZE;

  double sum = 0.0;
  for (local_int_t idx = gid; idx < n; idx += inc) {
    double val = x[idx];
    double val1 = y[idx];
    sum = fma(val, val1, sum);
  }
  // shared
  double sdata[32];
  sdata[tid] = sum;

  reduce_sum(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

int ComputeDotProduct_cuda(const local_int_t n, const Vector &x,
                           const Vector &y, double &result,
                           double &time_allreduce) {

  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
  double *xv = x.values;
  double *yv = y.values;

  double *xv_d;
  double *yv_d;
  double *workspace_d;

  size_t n_size = n * sizeof(double);
  printFileLine;

  cudaMalloc((void **)&xv_d, n_size);
  checkGPUandPrintLine;

  cudaMalloc((void **)&yv_d, n_size);
  checkGPUandPrintLine;

  cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);
  checkGPUandPrintLine;

  cudaMemcpy(yv_d, yv, n_size, cudaMemcpyHostToDevice);
  checkGPUandPrintLine;

  cudaMalloc((void **)&workspace_d, n_size);
  checkGPUandPrintLine;

  // double endMemcpy = mytimer();

  size_t blockDim = (n - 1) / 512 + 1;
  size_t globalDim = 512;

  kernel_dodProduct1<<<blockDim, globalDim>>>(x.localLength, xv_d, workspace_d);
  checkGPUandPrintLine;

  // cudaMemcpy(result, workspace_d, sizeof(double), cudaMemcpyDeviceToHost);

  checkGPUandPrintLine;

  cudaFree(xv_d);
  cudaFree(yv_d);
  cudaFree(workspace_d);

  return 0;
}

int ComputeDotProduct_ref_cuda(const local_int_t n, const Vector &x,
                               const Vector &y, double &result,
                               double &time_allreduce) {
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
  double *xv = x.values;
  double *yv = y.values;
  if (yv == xv) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t i = 0; i < n; i++)
      local_result += xv[i] * xv[i];
  } else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t i = 0; i < n; i++)
      local_result += xv[i] * yv[i];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}