#include "ComputeProlongation.hpp"

int ComputeProlongation(const SparseMatrix &Af, Vector &xf) {
  return ComputeProlongationInside(Af, xf);
}
