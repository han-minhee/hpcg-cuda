
#ifndef COMPUTEPROLONGATION_HPP
#define COMPUTEPROLONGATION_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include "../cuda-src/ComputeProlongationInside.cuh"

inline int ComputeProlongation(const SparseMatrix &Af, Vector &xf) {
  return ComputeProlongationInside(Af, xf);
}
#endif