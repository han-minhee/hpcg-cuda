#include "VectorInside.cuh"
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <vector>


void CudaInitializeVectorInside(Vector &v, local_int_t localLength) {
  v.localLength = localLength;
  v.optimizationData = 0;
  cudaMalloc((void **)&v.d_values, localLength * sizeof(double));
}

void CudaZeroVectorInside(Vector &v) {
  // local_int_t localLength = v.localLength;
  // cudaFree(v.d_values);
  // cudaMalloc((void **)&v.d_values, v.localLength * sizeof(double));
  thrust::device_ptr<double> dev_ptr(v.d_values);
  thrust::fill(dev_ptr, dev_ptr + v.localLength, 0);
  // cudaMemset(v.d_values, 0, v.localLength * sizeof(double));
    // cudaMemset((void **)&v.d_values, 0, v.localLength * sizeof(double));
      // cudaMemset((void **)&v.d_values, 0.0f, v.localLength * sizeof(double));
}

void CudaScaleVectorValueInside(Vector &v, local_int_t index, double value) {
  assert(index >= 0 && index < v.localLength);
  double *vv = v.values;
  vv[index] *= value;
  vectorMemcpyFromHostToDevice(v);
  return;
}

// TODO: chage to Curand later
void CudaFillRandomVectorInside(Vector &v) {
  std::vector<double> rng(v.localLength);
  for(int i = 0; i < v.localLength; ++i)
  {
    rng[i] = rand() / (double)(RAND_MAX) + 1.0;
  }

  cudaMemcpy(v.d_values,
                      rng.data(),
                      sizeof(double) * v.localLength,
                      cudaMemcpyHostToDevice);
}

void CudaCopyVectorInside(const Vector &v, Vector &w) {
  cudaMemcpy(w.d_values, v.d_values, v.localLength * sizeof(double),
             cudaMemcpyDeviceToDevice);
}

void CudaDeleteVectorInside(Vector &v) {
  cudaFree(v.d_values);
  v.localLength = 0;
}
