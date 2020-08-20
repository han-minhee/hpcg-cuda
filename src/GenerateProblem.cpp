#include "../cuda-src/GenerateProblemInside.cuh"
#include "GenerateProblem.hpp"

void GenerateProblem(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact) {
  return GenerateProblemInside(A, b, x, xexact);
}

void CopyProblemToHost(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact) {
  // Allocate host structures
  A.nonzerosInRow = new char[A.localNumberOfRows];
  A.mtxIndG = new global_int_t *[A.localNumberOfRows];
  A.mtxIndL = new local_int_t *[A.localNumberOfRows];
  A.matrixValues = new double *[A.localNumberOfRows];
  A.matrixDiagonal = new double *[A.localNumberOfRows];
  local_int_t *mtxDiag = new local_int_t[A.localNumberOfRows];
  A.localToGlobalMap.resize(A.localNumberOfRows);

  // Now allocate the arrays pointed to
  A.mtxIndL[0] =
      new local_int_t[A.localNumberOfRows * A.numberOfNonzerosPerRow];
  A.matrixValues[0] =
      new double[A.localNumberOfRows * A.numberOfNonzerosPerRow];
  A.mtxIndG[0] =
      new global_int_t[A.localNumberOfRows * A.numberOfNonzerosPerRow];

  // Copy GPU data to host
  CUDA_CHECK_COMMAND(cudaMemcpy(A.nonzerosInRow, A.d_nonzerosInRow,
                                sizeof(char) * A.localNumberOfRows,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaMemcpy(A.mtxIndG[0], A.d_mtxIndG,
                                sizeof(global_int_t) * A.localNumberOfRows *
                                    A.numberOfNonzerosPerRow,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaMemcpy(A.matrixValues[0], A.d_matrixValues,
                                sizeof(double) * A.localNumberOfRows *
                                    A.numberOfNonzerosPerRow,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaMemcpy(mtxDiag, A.d_matrixDiagonal,
                                sizeof(local_int_t) * A.localNumberOfRows,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaMemcpy(A.localToGlobalMap.data(), A.d_localToGlobalMap,
                                sizeof(global_int_t) * A.localNumberOfRows,
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_COMMAND(cudaFree(A.d_nonzerosInRow));
  CUDA_CHECK_COMMAND(cudaFree(A.d_matrixDiagonal));

  // Initialize pointers
  A.matrixDiagonal[0] = A.matrixValues[0] + mtxDiag[0];
  for (local_int_t i = 1; i < A.localNumberOfRows; ++i) {
    A.mtxIndL[i] = A.mtxIndL[0] + i * A.numberOfNonzerosPerRow;
    A.matrixValues[i] = A.matrixValues[0] + i * A.numberOfNonzerosPerRow;
    A.mtxIndG[i] = A.mtxIndG[0] + i * A.numberOfNonzerosPerRow;
    A.matrixDiagonal[i] = A.matrixValues[i] + mtxDiag[i];
  }

  delete[] mtxDiag;

  // Create global to local map
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
    A.globalToLocalMap[A.localToGlobalMap[i]] = i;
  }

  // Allocate and copy vectors, if available
  if (b != NULL) {
    InitializeVector(*b, A.localNumberOfRows);
    CUDA_CHECK_COMMAND(cudaMemcpy(b->values, b->d_values,
                                  sizeof(double) * b->localLength,
                                  cudaMemcpyDeviceToHost));
  }

  if (x != NULL) {
    InitializeVector(*x, A.localNumberOfRows);
    CUDA_CHECK_COMMAND(cudaMemcpy(x->values, x->d_values,
                                  sizeof(double) * x->localLength,
                                  cudaMemcpyDeviceToHost));
  }

  if (xexact != NULL) {
    InitializeVector(*xexact, A.localNumberOfRows);
    CUDA_CHECK_COMMAND(cudaMemcpy(xexact->values, xexact->d_values,
                                  sizeof(double) * xexact->localLength,
                                  cudaMemcpyDeviceToHost));
  }
}
