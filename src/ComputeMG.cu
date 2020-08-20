
#include "ComputeMG.cuh"
#include "ComputeProlongation.cuh"
#include "ComputeRestriction.cuh"
#include "ComputeSPMV.cuh"
#include "ComputeSYMGS.cuh"
#include "Utils.cuh"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as
  the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix &A, const Vector &r, Vector &x) {
  assert(x.localLength == A.localNumberOfColumns);

  if (A.mgData != 0) {
    CUDA_RETURN_IFF_ERROR(ComputeSYMGSZeroGuess(A, r, x));

    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;

    for (int i = 1; i < numberOfPresmootherSteps; ++i) {
      CUDA_RETURN_IFF_ERROR(ComputeSYMGS(A, r, x));
    }

#ifndef HPCG_REFERENCE
    CUDA_RETURN_IFF_ERROR(ComputeFusedSpMVRestriction(A, r, x));
#else
    CUDA_RETURN_IFF_ERROR(ComputeSPMV(A, x, *A.mgData->Axf));
    CUDA_RETURN_IFF_ERROR(ComputeRestriction(A, r));
#endif

    CUDA_RETURN_IFF_ERROR(ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc));
    CUDA_RETURN_IFF_ERROR(ComputeProlongation(A, x));

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;

    for (int i = 0; i < numberOfPostsmootherSteps; ++i) {
      CUDA_RETURN_IFF_ERROR(ComputeSYMGS(A, r, x));
    }
  } else {
    CUDA_RETURN_IFF_ERROR(ComputeSYMGSZeroGuess(A, r, x));
  }

  return 0;
}
