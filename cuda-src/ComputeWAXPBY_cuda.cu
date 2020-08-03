#include "ComputeWAXPBY_cuda.cuh"

#define BLOCK_SIZE 32

__global__ void kernel_waxpby(local_int_t size, double alpha, const double *x,
                              double beta, const double *y, double *w) {
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

int ComputeWAXPBY_cuda(const local_int_t n, const double alpha, const Vector &x,
                       const double beta, const Vector &y, Vector &w) {

  assert(x.localLength >= n);
  assert(y.localLength >= n);

  double *xv = x.values;
  double *yv = y.values;
  double *wv = w.values;

  double *xv_d;
  double *yv_d;
  double *wv_d;
  assert(x.localLength >= n);
  assert(y.localLength >= n);
  assert(w.localLength >= n);

  size_t n_size = n * sizeof(double);

  cudaMalloc((void **)&xv_d, n_size);

  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMalloc((void **)&yv_d, n_size);

  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMalloc((void **)&wv_d, n_size);
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMemcpy(yv_d, yv, n_size, cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  size_t blockDim = (n - 1) / 512 + 1;
  size_t globalDim = 512;

  kernel_waxpby<<<blockDim, globalDim>>>(n, alpha, xv_d, beta, yv_d, wv_d);

  // printf("line passed %d\n", __LINE__);
  cudaDeviceSynchronize();
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMemcpy(wv, wv_d, n_size, cudaMemcpyDeviceToHost);
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);
  cudaDeviceSynchronize();
  cudaFree(wv_d);
  cudaFree(xv_d);
  cudaFree(yv_d);

  return 0;
}

int ComputeWAXPBY_ref_cuda(const local_int_t n, const double alpha,
                           const Vector &x, const double beta, const Vector &y,
                           Vector &w) {

  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  const double *const xv = x.values;
  const double *const yv = y.values;
  double *const wv = w.values;

  if (alpha == 1.0) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = 0; i < n; i++)
      wv[i] = xv[i] + beta * yv[i];
  } else if (beta == 1.0) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = 0; i < n; i++)
      wv[i] = alpha * xv[i] + yv[i];
  } else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = 0; i < n; i++)
      wv[i] = alpha * xv[i] + beta * yv[i];
  }

  return 0;
}
