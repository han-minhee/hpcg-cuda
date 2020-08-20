
#include "../cuda-src/ComputeMGInside.cuh"
#include "ComputeMG.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as
  the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix &A, const Vector &r, Vector &x) {
  return ComputeMGInside(A, r, x);
}
