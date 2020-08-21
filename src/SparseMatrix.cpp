#include "SparseMatrix.hpp"
#include "../cuda-src/SparseMatrixInside.cuh"

inline void CudaCopyMatrixDiagonal(const SparseMatrix &A, Vector &diagonal) {
  CudaCopyMatrixDiagonalInside(A, diagonal);
}

inline void CudaReplaceMatrixDiagonal(SparseMatrix &A, const Vector &diagonal) {
  CudaReplaceMatrixDiagonalInside(A, diagonal);
}
inline void ConvertToELL(SparseMatrix &A) { ConvertToELLInside(A); }
inline void ExtractDiagonal(SparseMatrix &A) { ExtractDiagonalInside(A); }