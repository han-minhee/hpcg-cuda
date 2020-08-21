#include "../src/SparseMatrix.hpp"

void CudaCopyMatrixDiagonalInside(const SparseMatrix &A, Vector &diagonal);
void CudaReplaceMatrixDiagonalInside(SparseMatrix &A, const Vector &diagonal);
void ConvertToELLInside(SparseMatrix &A);
void ExtractDiagonalInside(SparseMatrix &A);