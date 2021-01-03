#include "OptimizeProblem.hpp"
#include "../cuda-src/MultiColoring.cuh"
#include "../cuda-src/Permute.cuh"
#include "SparseMatrixOp.hpp"


int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
                    Vector &xexact) {
  // Perform matrix coloring
  JPLColoring(A);

  // Permute matrix columns
  PermuteColumns(A);

  // Convert matrix to ELL format
  ConvertToELL(A);

  // Permute matrix rows
  PermuteRows(A);

  // Extract diagonal indices and inverse values
  ExtractDiagonal(A);

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

    // Permute matrix rows
    PermuteRows(*M);

    // Extract diagonal indices and inverse values
    ExtractDiagonal(*M);

    // Go to next level in hierarchy
    M = M->Ac;
  }
  return 0;
}
