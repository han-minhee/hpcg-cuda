#include "../src/SparseMatrix.hpp"
#include "../src/Utils.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>

void CudaCopyMatrixDiagonalInside(const SparseMatrix &A, Vector &diagonal);
void CudaReplaceMatrixDiagonalInside(SparseMatrix &A, const Vector &diagonal);
void ConvertToELLInside(SparseMatrix &A);
void ExtractDiagonalInside(SparseMatrix &A);