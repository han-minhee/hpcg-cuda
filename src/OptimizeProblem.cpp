#include "OptimizeProblem.hpp"
#include "../cuda-src/MultiColoring.cuh"
#include "../cuda-src/Permute.cuh"
#include "SparseMatrixOp.hpp"


int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
                    Vector &xexact) {
  // Perform matrix coloring
  JPLColoring(A);
  // printf("finished JPLCoring @ opt\n");

  // Permute matrix columns
  PermuteColumns(A);
  // printf("finished PermuteColumns @ opt\n");

  // Convert matrix to ELL format
  ConvertToELL(A);
  // printf("finished Convert to ELL @ opt\n");

  // Permute matrix rows
  PermuteRows(A);
  // printf("finished Permute Rows @ opt\n");

  // Extract diagonal indices and inverse values
  ExtractDiagonal(A);
    // printf("finished Extract Diagonal @ opt\n");


  // Permute vectors
  PermuteVector(A.localNumberOfRows, b, A.perm);
  PermuteVector(A.localNumberOfRows, xexact, A.perm);
  // printf("finished PermuteVector @ opt\n");

  // Initialize CG structures
  CudaInitializeSparseCGData(A, data);

  double * ApVal = new double[100];
  CUDA_CHECK_COMMAND(cudaMemcpy(ApVal, data.Ap.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost));
  //// printf("cudaMemcpyD2H %p in OptimizeProblem\n", data.Ap.d_values);
     for(int i = 0; i < 10; i++)
    {
        // printf("after initialization, ApVal [%d] %x \n", i, ApVal[i]);
    }

  // Process all coarse level matrices
  SparseMatrix *M = A.Ac;

  while (M != NULL) {
    // Perform matrix coloring
    JPLColoring(*M);
  // printf("finished JPL inner @ opt\n");

    // Permute matrix columns
    PermuteColumns(*M);
  // printf("finished Permute Columns Inner @ opt\n");

    // Convert matrix to ELL format
    ConvertToELL(*M);
  // printf("finished ELL Inner @ opt\n");

    // Permute matrix rows
    PermuteRows(*M);
  // printf("finished Rows Inner @ opt\n");

    // Extract diagonal indices and inverse values
    ExtractDiagonal(*M);
  // printf("finished Extract Diagonal Inner @ opt\n");

    // Go to next level in hierarchy
    M = M->Ac;
  }
  return 0;
}
