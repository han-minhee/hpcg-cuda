#include "../cuda-src/SparseMatrixInside.cuh"

void CudaCopyMatrixDiagonal(const SparseMatrix &A, Vector &diagonal) {
  return CudaCopyMatrixDiagonalInside(A, diagonal);
}

void CudaReplaceMatrixDiagonal(SparseMatrix &A, const Vector &diagonal) {
  return CudaReplaceMatrixDiagonalInside(A, diagonal);
}
// void ConvertToELL(SparseMatrix &A) { return ConvertToELLInside(A); }

void DeleteMatrix(SparseMatrix &A) { return DeleteMatrixInside(A); }