#include "ComputeDotProduct_cuda.cuh"

// notice: originally, for the use of MPI, there should be result,
// time_allreduce variable, but currently omitted. instead of changing a value,
// it returns a double var.
#define BLOCK_SIZE 32

__global__ void kernel_dot2_part1(local_int_t n, const double *x,
                                  const double *y, double *workspace) {
  local_int_t tid = hipThreadIdx_x;
  local_int_t gid = hipBlockIdx_x * BLOCK_SIZE + tid;
  local_int_t inc = hipGridDim_x * BLOCK_SIZE;

  double sum = 0.0;
  for (local_int_t idx = gid; idx < n; idx += inc) {
    sum = fma(y[idx], x[idx], sum);
  }

  __shared__ double sdata[BLOCK_SIZE];
  sdata[tid] = sum;

  reduce_sum<BLOCK_SIZE>(tid, sdata);

  if (tid == 0) {
    workspace[hipBlockIdx_x] = sdata[0];
  }
}

template <unsigned int BLOCK_SIZE>
__launch_bounds__(256) __global__ void kernel_dot_part2(double *workspace) {
  __shared__ double sdata[BLOCK_SIZE];
  sdata[hipThreadIdx_x] = workspace[hipThreadIdx_x];

  reduce_sum<BLOCK_SIZE>(hipThreadIdx_x, sdata);

  if (hipThreadIdx_x == 0) {
    workspace[0] = sdata[0];
  }
}

int ComputeDotProduct_cuda(const local_int_t n, const Vector &x,
                           const Vector &y, double &result,
                           double &time_allreduce) {
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double *xv = x.values;
  double *yv = y.values;
  double *local_results = new double[n]();

  double *xv_d;
  double *yv_d;
  double *local_results_d;

  size_t n_size = n * sizeof(double);

  result = 0.0;

  cudaMalloc((void **)&xv_d, n_size);
  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMalloc((void **)&yv_d, n_size);

  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMalloc((void **)&local_results_d, n_size);

  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMemcpy(yv_d, yv, n_size, cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMemcpy(local_results_d, local_results, n_size, cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaDeviceSynchronize();

  size_t deviceWarpSize = 32;
  int numBlocks = (n + deviceWarpSize - 1) / deviceWarpSize;
  // kernelDotProduct<<<numBlocks, deviceWarpSize>>>(
  //    n, xv_d, yv_d, local_results_d, deviceWarpSize);

  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);

  if (gpuCheckError() == -1) {
    return -1;
  }

  cublasDdot(h, n, xv_d, 1, yv_d, 1, local_results_d);

  cudaDeviceSynchronize();

  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMemcpy(local_results, local_results_d, n_size, cudaMemcpyDeviceToHost);
  if (gpuCheckError() == -1) {
    return -1;
  }
  cudaDeviceSynchronize();

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : result)
#endif
  for (int i = 0; i < n; i++) {
    result += local_results[i];
  }

  cudaFree(xv_d);
  cudaFree(yv_d);
  cudaFree(local_results_d);
  free(local_results);

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