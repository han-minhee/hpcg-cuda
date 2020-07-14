#include "ComputeWAXPBY_cuda.cuh"

__global__ void kernelWAXPBY(int n, double alpha, double beta, double *xv,
                             double *yv, double *wv, int deviceWarpSize) {
  int localIndex;
  int elemsPerThreads = deviceWarpSize;

  int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
  globalIndex *= elemsPerThreads;
  if (globalIndex + elemsPerThreads >= n)
    return;

  for (localIndex = globalIndex; localIndex < globalIndex + elemsPerThreads;
       localIndex++) {
    if (alpha == 1.0) {
      wv[localIndex] = xv[localIndex] + beta * yv[localIndex];
    } else if (beta == 1.0) {
      wv[localIndex] = alpha * xv[localIndex] + yv[localIndex];
    } else {
      wv[localIndex] = alpha * xv[localIndex] + beta * yv[localIndex];
    }
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

  // cudaMalloc
  size_t n_size = n * sizeof(double);
  size_t deviceWarpSize = 32;


  cudaMalloc((void **)&xv_d, n_size);

  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }

  cudaMalloc((void **)&yv_d, n_size);

  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }

  cudaMalloc((void **)&wv_d, n_size);
  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }

  cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);
  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }
  cudaMemcpy(yv_d, yv, n_size, cudaMemcpyHostToDevice);
  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }

  int numBlocks = (n + deviceWarpSize - 1) / deviceWarpSize;

  kernelWAXPBY<<<numBlocks, deviceWarpSize>>>(n, alpha, beta, xv_d, yv_d, wv_d,
                                        deviceWarpSize);
  cudaDeviceSynchronize();
  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }

  cudaMemcpy(wv, wv_d, n_size, cudaMemcpyDeviceToHost);
  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }

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
