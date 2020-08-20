#include "ComputeSYMGS.hpp"
#include "../cuda-src/ComputeSYMGSInside.cuh"

int ComputeSYMGS(const SparseMatrix &A, const Vector &r, Vector &x) {
  return ComputeSYMGSInside(A, r, x);
}

int ComputeSYMGSZeroGuess(const SparseMatrix &A, const Vector &r, Vector &x) {
  return ComputeSYMGSZeroGuessInside(A, r, x);
}
