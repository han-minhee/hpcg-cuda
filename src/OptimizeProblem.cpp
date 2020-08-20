#include "OptimizeProblem.hpp"
#include "../cuda-src/MultiColoring.cuh"
#include "../cuda-src/Permute.cuh"
#include "SparseMatrix.hpp"

int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
                    Vector &xexact) {
  // Perform matrix coloring
  // JPLColoring(A);

  // Permute matrix columns
  PermuteColumns(A);

  // Convert matrix to ELL format
  ConvertToELL(A);

  // Defrag permutation vector
  // CUDA_CHECK_COMMAND(deviceDefrag((void**)&A.perm, sizeof(local_int_t) *
  // A.localNumberOfRows));

  // Permute matrix rows
  PermuteRows(A);

  // Extract diagonal indices and inverse values
  ExtractDiagonal(A);

  // Defrag
  // CUDA_CHECK_COMMAND(deviceDefrag((void**)&A.diag_idx, sizeof(local_int_t) *
  // A.localNumberOfRows)); CUDA_CHECK_COMMAND(deviceDefrag((void**)&A.inv_diag,
  // sizeof(double) * A.localNumberOfRows));
#ifndef HPCG_NO_MPI
  // CUDA_CHECK_COMMAND(deviceDefrag((void**)&A.d_send_buffer, sizeof(double) *
  // A.totalToBeSent));
  // CUDA_CHECK_COMMAND(deviceDefrag((void**)&A.d_elementsToSend,
  // sizeof(local_int_t) * A.totalToBeSent));
#endif

  // Permute vectors
  PermuteVector(A.localNumberOfRows, b, A.perm);
  PermuteVector(A.localNumberOfRows, xexact, A.perm);

  // Initialize CG structures
  CudaInitializeSparseCGData(A, data);

  // Process all coarse level matrices
  SparseMatrix *M = A.Ac;

  while (M != NULL) {
    // Perform matrix coloring
    JPLColoring(*M);

    // Permute matrix columns
    PermuteColumns(*M);

    // Convert matrix to ELL format
    ConvertToELL(*M);

    // Defrag matrix arrays and permutation vector
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->ell_col_ind,
    // sizeof(local_int_t) * M->ell_width * M->localNumberOfRows));
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->ell_val, sizeof(double) *
    // M->ell_width * M->localNumberOfRows));
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->perm, sizeof(local_int_t) *
    // M->localNumberOfRows));

    // Permute matrix rows
    PermuteRows(*M);

    // Extract diagonal indices and inverse values
    ExtractDiagonal(*M);

    // Defrag
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->diag_idx, sizeof(local_int_t)
    // * M->localNumberOfRows));
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->inv_diag, sizeof(double) *
    // M->localNumberOfRows));
#ifndef HPCG_NO_MPI
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->d_send_buffer, sizeof(double)
    // * M->totalToBeSent));
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&M->d_elementsToSend,
    // sizeof(local_int_t) * M->totalToBeSent));
#endif

    // Go to next level in hierarchy
    M = M->Ac;
  }

  // Defrag hierarchy structures
  M = &A;
  MGData *mg = M->mgData;

  while (mg != NULL) {
    M = M->Ac;

    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&mg->d_f2cOperator,
    // sizeof(local_int_t) * M->localNumberOfRows));
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&mg->rc->d_values, sizeof(double)
    // * mg->rc->localLength));
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&mg->xc->d_values, sizeof(double)
    // * mg->xc->localLength));
#ifdef HPCG_REFERENCE
    // CUDA_CHECK_COMMAND(deviceDefrag((void**)&mg->Axf->d_values,
    // sizeof(double) * mg->Axf->localLength));
#endif

    mg = M->mgData;
  }

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
// double OptimizeProblemMemoryUse(const SparseMatrix &A) { return 0.0; }
