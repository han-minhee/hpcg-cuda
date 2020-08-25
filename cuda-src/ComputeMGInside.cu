
#include "../src/ComputeProlongation.hpp"
#include "../src/ComputeRestriction.hpp"
#include "../src/ComputeSPMV.hpp"
#include "../src/ComputeSYMGS.hpp"
#include "ComputeMGInside.cuh"
#include "Utils.cuh"

int ComputeMGInside(const SparseMatrix &A, const Vector &r, Vector &x) {

  printf("===entering MG Inside ===\n");
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

    CUDA_RETURN_IFF_ERROR(ComputeMGInside(*A.Ac, *A.mgData->rc, *A.mgData->xc));
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
