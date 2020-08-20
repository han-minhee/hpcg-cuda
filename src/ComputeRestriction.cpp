
#include "ComputeRestriction.hpp"
#include "ExchangeHalo.hpp"

int ComputeRestriction(const SparseMatrix &A, const Vector &rf) {
  return ComputeRestrictionInside(A, rf);
}

int ComputeFusedSpMVRestriction(const SparseMatrix &A, const Vector &rf,
                                Vector &xf) {
  return ComputeFusedSpMVRestrictionInside(A, rf, xf);
}
