#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "Geometry.hpp"
#include "Utils.cuh"
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#define CudaVectorCopyHostToDevice(vector)                                     \
  cudaMemcpy(vector.d_values, vector.values, sizeof(vector.values),            \
             cudaMemcpyHostToDevice);

#define CudaVectorCopyDeviceToHost(vector)                                     \
  cudaMemcpy(vector.values, vector.d_values, sizeof(vector.d_values),          \
             cudaMemcpyDeviceToHost);

struct Vector_STRUCT {
  local_int_t localLength; //!< length of local portion of the vector
  double *values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void *optimizationData;

  // added for CUDA
  double *d_values;
};
typedef struct Vector_STRUCT Vector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector &v, local_int_t localLength) {
  v.localLength = localLength;
  v.values = new double[localLength];
  v.optimizationData = 0;
  return;
}

inline void CudaInitializeVector(Vector &v, local_int_t localLength) {
  v.localLength = localLength;
  v.optimizationData = 0;
  cudaMalloc((void **)&v.d_values, localLength * sizeof(double));
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are
  zero.
 */
inline void ZeroVector(Vector &v) {
  local_int_t localLength = v.localLength;
  double *vv = v.values;
  for (int i = 0; i < localLength; ++i)
    vv[i] = 0.0;

  return;
}

inline void CudaZeroVector(Vector &v) {
  cudaMemset((void **)&v.d_values, 0.0, v.localLength * sizeof(double));
}

/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */

// Should be implemented in some different way?
inline void ScaleVectorValue(Vector &v, local_int_t index, double value) {
  assert(index >= 0 && index < v.localLength);
  double *vv = v.values;
  vv[index] *= value;
  return;
}

// __global__ void scale_elem(double *arr, int idx, double val) {
//   arr[idx] *= val;
// }
// //  scale_elem<<<1, 1>>>(v.d_values, index, value);

inline void CudaScaleVectorValue(Vector &v, local_int_t index, double value) {
  assert(index >= 0 && index < v.localLength);
  ScaleVectorValue(v, index, value);
  vectorMemcpyFromHostToDevice(v);
  return;
}

/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector &v) {
  local_int_t localLength = v.localLength;
  double *vv = v.values;
  for (int i = 0; i < localLength; ++i)
    vv[i] = rand() / (double)(RAND_MAX) + 1.0;

  cudaMemcpy(v.d_values, vv, localLength * sizeof(double),
             cudaMemcpyHostToDevice);
  return;
}

// TODO: chage to Curand later
inline void CudaFillRandomVector(Vector &v) {
  local_int_t localLength = v.localLength;
  double *vv = v.values;
  for (int i = 0; i < localLength; ++i)
    vv[i] = rand() / (double)(RAND_MAX) + 1.0;

  cudaMemcpy(v.d_values, vv, localLength * sizeof(double),
             cudaMemcpyHostToDevice);
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(const Vector &v, Vector &w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  double *vv = v.values;
  double *wv = w.values;
  for (int i = 0; i < localLength; ++i)
    wv[i] = vv[i];
  return;
}

inline void CudaCopyVector(const Vector &v, Vector &w) {
  cudaMemcpy(w.d_values, v.d_values, v.localLength * sizeof(double),
             cudaMemcpyDeviceToDevice);
}

/*!
  Deallocates the members of the data structure of the known system matrix
  provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector &v) {

  delete[] v.values;
  v.localLength = 0;
  return;
}

inline void CudaDeleteVector(Vector &v) {
  cudaFree(v.d_values);
  v.localLength = 0;
}

#endif // VECTOR_HPP
