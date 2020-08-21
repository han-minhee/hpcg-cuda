
#include "GenerateProblem.hpp"
#include "../cuda-src/GenerateProblemInside.cuh"

void GenerateProblem(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact) {
  return GenerateProblemInside(A, b, x, xexact);
}

void CopyProblemToHost(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact) {
  return CopyProblemToHostInside(A, b, x, xexact);
}
