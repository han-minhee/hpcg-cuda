
#include "GenerateCoarseProblem.hpp"
#include "../cuda-src/GenerateCoarseProblemInside.cuh"

void GenerateCoarseProblem(const SparseMatrix &Af) {
  return GenerateCoarseProblemInside(Af);
}

void CopyCoarseProblemToHost(SparseMatrix &A) {
  return CopyCoarseProblemToHostInside(A);
}
