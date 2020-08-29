#include "VectorInside.cuh"
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

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
  // cudaMemset((void **)&v.d_values, 0, v.localLength * sizeof(double));
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
  local_int_t localLength = v.localLength;
  double *vv = v.values;
  for (int i = 0; i < localLength; ++i)
    vv[i] = rand() / (double)(RAND_MAX) + 1.0;

  cudaMemcpy(v.d_values, vv, localLength * sizeof(double),
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
