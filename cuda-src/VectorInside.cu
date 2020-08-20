#include "VectorInside.cuh"
#include <cuda_runtime.h>

 void CudaInitializeVectorInside(Vector &v, local_int_t localLength) {
  v.localLength = localLength;
  v.optimizationData = 0;
  cudaMalloc((void **)&v.d_values, localLength * sizeof(double));
}

 void CudaZeroVectorInside(Vector &v) {
  cudaMemset((void **)&v.d_values, 0.0, v.localLength * sizeof(double));
}

 void CudaScaleVectorValueInside(Vector &v, local_int_t index, double value) {
  assert(index >= 0 && index < v.localLength);
  ScaleVectorValue(v, index, value);
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
