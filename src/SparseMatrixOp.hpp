#ifndef SPARSEMATRIXOPS_HPP
#define SPARSEMATRIXOPS_HPP

#include "../cuda-src/SparseMatrixInside.cuh"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

inline void InitializeSparseMatrix(SparseMatrix &A, Geometry *geom) { 
  return InitializeSparseMatrixInside(A, geom);
}

inline void ConvertToELL(SparseMatrix &A) { ConvertToELLInside(A); }
inline void ExtractDiagonal(SparseMatrix &A) { ExtractDiagonalInside(A); }

inline void CudaCopyMatrixDiagonal(const SparseMatrix &A, Vector &diagonal) {
  CudaCopyMatrixDiagonalInside(A, diagonal);
}
inline void CopyMatrixDiagonal(SparseMatrix &A, Vector &diagonal) {
  double **curDiagA = A.matrixDiagonal;
  double *dv = diagonal.values;
  assert(A.localNumberOfRows == diagonal.localLength);
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
    dv[i] = *(curDiagA[i]);
  CudaCopyMatrixDiagonal(A, diagonal);
}

inline void CudaReplaceMatrixDiagonal(SparseMatrix &A, const Vector &diagonal) {
  CudaReplaceMatrixDiagonalInside(A, diagonal);
}

inline void ReplaceMatrixDiagonal(SparseMatrix &A, Vector &diagonal) {
  double **curDiagA = A.matrixDiagonal;
  double *dv = diagonal.values;
  assert(A.localNumberOfRows == diagonal.localLength);
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
    *(curDiagA[i]) = dv[i];
  CudaReplaceMatrixDiagonal(A, diagonal);
}

inline void DeleteMatrix(SparseMatrix &A) { return DeleteMatrixInside(A); }
#endif