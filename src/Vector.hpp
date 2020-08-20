#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "../cuda-src/VectorInside.cuh"
#include "Geometry.hpp"
#include <cassert>
#include <cstdlib>


inline void InitializeVector(Vector &v, local_int_t localLength) {
  v.localLength = localLength;
  v.values = new double[localLength];
  v.optimizationData = 0;
  return;
}

inline void CudaInitializeVector(Vector &v, local_int_t localLength) {
  return CudaInitializeVectorInside(v, localLength);
}

inline void ZeroVector(Vector &v) {
  local_int_t localLength = v.localLength;
  double *vv = v.values;
  for (int i = 0; i < localLength; ++i)
    vv[i] = 0.0;
  return;
}

inline void CudaZeroVector(Vector &v) { return CudaZeroVectorInside(v); }

inline void ScaleVectorValue(Vector &v, local_int_t index, double value) {
  assert(index >= 0 && index < v.localLength);
  double *vv = v.values;
  vv[index] *= value;
  return;
}

inline void CudaScaleVectorValue(Vector &v, local_int_t index, double value) {
  return CudaScaleVectorValueInside(v, index, value);
}

inline void FillRandomVector(Vector &v) {
  local_int_t localLength = v.localLength;
  double *vv = v.values;
  for (int i = 0; i < localLength; ++i)
    vv[i] = rand() / (double)(RAND_MAX) + 1.0;
  return;
}

inline void CudaFillRandomVector(Vector &v) {
  return CudaFillRandomVectorInside(v);
}

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
  return CudaCopyVectorInside(v, w);
}

inline void DeleteVector(Vector &v) {
  delete[] v.values;
  v.localLength = 0;
  return;
}

inline void CudaDeleteVector(Vector &v) { return CudaDeleteVectorInside(v); }

#endif // VECTOR_HPP
