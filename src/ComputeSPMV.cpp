#include "ComputeSPMV.hpp"
#include "../cuda-src/ComputeSPMVInside.cuh"

int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y) {
  return ComputeSPMVInside(A, x, y);
}
