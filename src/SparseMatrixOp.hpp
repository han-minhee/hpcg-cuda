#include "../cuda-src/SparseMatrixInside.cuh"

void CudaCopyMatrixDiagonal(const SparseMatrix &A, Vector &diagonal) {
  CudaCopyMatrixDiagonalInside(A, diagonal);
}

void CudaReplaceMatrixDiagonal(SparseMatrix &A, const Vector &diagonal) {
  CudaReplaceMatrixDiagonalInside(A, diagonal);
}
void ConvertToELL(SparseMatrix &A) { ConvertToELLInside(A); }
void ExtractDiagonal(SparseMatrix &A) { ExtractDiagonalInside(A); }