#ifndef SPARSEMATRIXInside_HPP
#define SPARSEMATRIXInside_HPP
#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"
#include <cuda_runtime.h>

inline void InitializeSparseMatrixInside(SparseMatrix &A, Geometry *geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized = true;
  A.isMgOptimized = true;
  A.isWaxpbyOptimized = true;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;

  A.recv_request = NULL;
  A.send_request = NULL;
  A.d_elementsToSend = NULL;
  A.recv_buffer = NULL;
  A.send_buffer = NULL;
  A.d_send_buffer = NULL;

  A.halo_row_ind = NULL;
  A.halo_col_ind = NULL;
  A.halo_val = NULL;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac = 0;

  A.ell_width = 0;
  A.ell_col_ind = NULL;
  A.ell_val = NULL;
  A.diag_idx = NULL;
  A.inv_diag = NULL;

  A.nblocks = 0;
  A.ublocks = 0;
  A.sizes = NULL;
  A.offsets = NULL;
  A.perm = NULL;

  return;
}

void ConvertToELLInside(SparseMatrix &A);
void ExtractDiagonalInside(SparseMatrix &A);

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before
  call to this function).
 */
void CudaCopyMatrixDiagonalInside(const SparseMatrix &A, Vector &diagonal);
inline void CopyMatrixDiagonalInside(SparseMatrix &A, Vector &diagonal) {
  double **curDiagA = A.matrixDiagonal;
  double *dv = diagonal.values;
  assert(A.localNumberOfRows == diagonal.localLength);
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
    dv[i] = *(curDiagA[i]);
  CudaCopyMatrixDiagonalInside(A, diagonal);
  return;
}

/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing
  matrix diagonal values.
 */
void CudaReplaceMatrixDiagonalInside(SparseMatrix &A, const Vector &diagonal);
inline void ReplaceMatrixDiagonalInside(SparseMatrix &A, Vector &diagonal) {
  double **curDiagA = A.matrixDiagonal;
  double *dv = diagonal.values;
  assert(A.localNumberOfRows == diagonal.localLength);
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
    *(curDiagA[i]) = dv[i];
  CudaReplaceMatrixDiagonalInside(A, diagonal);
  return;
}

void DeleteMatrixInside(SparseMatrix &A);

#endif