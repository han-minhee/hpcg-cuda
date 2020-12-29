#include "VectorInside.cuh"
#include <Utils.cuh>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vector>

void CudaInitializeVectorInside(Vector &v, local_int_t localLength) {
  v.localLength = localLength;
  v.optimizationData = 0;
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&v.d_values, localLength * sizeof(double)));
//  // printf("Allocated %p in CudaInitializeVectorInside\n", v.d_values);
//  CUDA_CHECK_COMMAND(cudaMemset(v.d_values, 0, localLength * sizeof(double)));
}

void CudaZeroVectorInside(Vector &v) {
  // local_int_t localLength = v.localLength;
  // cudaFree(v.d_values);
  // // cudaMalloc((void **)&v.d_values, v.localLength * sizeof(double));
  // thrust::device_ptr<double> dev_ptr(v.d_values);
  // thrust::fill(dev_ptr, dev_ptr + v.localLength, 0);
  CUDA_CHECK_COMMAND(cudaMemset(v.d_values, 0, v.localLength * sizeof(double)));
  //// printf("cudaMemset %p in CudaZeroVectorInside\n", v.d_values);
  // cudaMemset(v.d_values, 0, v.localLength * sizeof(double));
  // // cudaMemset((void **)&v.d_values, 0.0f, v.localLength * sizeof(double));
  // double *vv = new double[v.localLength];
  // for (int i = 0; i < v.localLength; ++i)
  //   vv[i] = 0.0;

  // cudaMemcpy(v.d_values, vv, sizeof(double) * v.localLength,
  // cudaMemcpyHostToDevice); delete[] vv;

  double *zeros = new double[100];
  CUDA_CHECK_COMMAND(cudaMemcpy(zeros, v.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost));
  //// printf("cudaMemcpyD2H %p in CudaZeroVectorInside\n", v.d_values);
  // printf("after ZeroVector Memset\n");
  bool has_nonzero = false;
  int nums = 0;

  for (int i = 0; i < 10; i++) {
      // printf("CudaZeroVectorInside[%d] : %x\n", i, zeros[i]);
      nums++;
  }

  delete[] zeros;
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
  for (int i = 0; i < v.localLength; ++i) {
    rng[i] = rand() / (double)(RAND_MAX) + 1.0;
  }

  cudaMemcpy(v.d_values, rng.data(), sizeof(double) * v.localLength,
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
