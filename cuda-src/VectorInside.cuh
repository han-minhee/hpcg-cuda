
#include "../src/hpcg.hpp"
#include "Utils.cuh"
#include <cassert>
#include <cstdlib>

struct Vector_STRUCT {
  local_int_t localLength; //!< length of local portion of the vector
  double *values;          //!< array of values
  void *optimizationData;

  // added for CUDA
  double *d_values;
};
typedef struct Vector_STRUCT Vector;

void CudaInitializeVectorInside(Vector &v, local_int_t localLength);

void CudaZeroVectorInside(Vector &v);

void CudaScaleVectorValueInside(Vector &v, local_int_t index, double value);

// TODO: chage to Curand later
void CudaFillRandomVectorInside(Vector &v);

void CudaCopyVectorInside(const Vector &v, Vector &w);

void CudaDeleteVectorInside(Vector &v);
