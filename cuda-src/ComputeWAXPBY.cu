#include "ComputeWAXPBY_cuda.cuh"

__global__ void kernelWAXPBY(int n, double alpha, double beta, double *xv,
                             double *yv, double *wv) {
  int localIndex;
  int elemsPerThreads = warpSize;

  int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
  globalIndex *= elemsPerThreads;
  if (globalIndex + elemsPerThreads >= n)
    return;

  for (localIndex = globalIndex; localIndex < globalIndex + elemsPerThreads;
       localIndex++) {
    if (alpha == 1.0) {
      wv[x] = xv[localIndex] + beta * yv[localIndex];
    } else if (beta == 1.0) {
      wv[x] = alpha * xv[localIndex] + yv[localIndex];
    } else {
      wv[x] = alpha * xv[localIndex] + beta * yv[localIndex];
    }
  }
}

int ComputeWAXPBY_cuda(const local_int_t n, const double alpha, const Vector &x,
                       const double beta, const Vector &y, Vector &w) {

  assert(x.localLength >= n);
  assert(y.localLength >= n);

  double *const xv = x.values;
  double *const yv = y.values;
  double *const wv = w.values;

  double *xv_d;
  double *yv_d;
  double *wv_d;

  // cudaMalloc
  size_t n_size = n * sizeof(double);
  cudaMalloc(xv_d, n_size);
  cudaMalloc(yv_d, n_size);
  cudaMalloc(wv_d, n_size);

  cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);
  cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);

  int numBlocks = (n + warpSize - 1) / warpSize;
  kernelWAXPBY<<<numBlocks, warpSize>>>(n, alpha, beta, xv_d, yv_d, wv_d);
  cudaMemcpy(wv_d, wv, n_size, cudaMemcpyDeviceToHost);

  if (gpuAssert(cudaPeekAtLastError()) == -1) {
    return -1;
  }
  if (gpuAssert(cudaDeviceSynchronize()) == -1) {
    return -1;
  }

  return 0;
}
