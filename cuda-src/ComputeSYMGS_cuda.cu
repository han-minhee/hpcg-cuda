
#include "ComputeSYMGS.cuh"

int ComputeSYMGS_cuda(const SparseMatrix &A, const Vector &r, Vector &x) {
  // cusparseDcsrsv_analysis
  /*
  We must perform the analysis for both the upper and lower triangular portions
  of the matrix. This analysis phase maps nicely to the optimization phase of
  the benchmark, and the time spent here is recorded in the optimization timing.

  r <-- rhs cublasDcopy
  r <-- r - A*x cusparseDcsrmv (SPMV)
  y <-- L*y=r cusparseDcsrsv_solve
  y <-- y*D cublasDaxpy
  dx <-- U*dx=y cusparseDcsrsv_solve
  x <-- x+dx cublasDaxpy

*/

  // cusparseDcsrsv_solve
}

int ComputeSYMGS_ref_cuda(const SparseMatrix &A, const Vector &r, Vector &x) {

  assert(x.localLength ==
         A.localNumberOfColumns); // Make sure x contain space for halo values

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal; // An array of pointers to the
                                              // diagonal entries A.matrixValues
  const double *const rv = r.values;
  double *const xv = x.values;

  for (local_int_t i = 0; i < nrow; i++) {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal =
        matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i];       // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] *
           currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum / currentDiagonal;
  }

  // Now the back sweep.

  for (local_int_t i = nrow - 1; i >= 0; i--) {
    const double *const currentValues = A.matrixValues[i];
    const local_int_t *const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal =
        matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i];       // RHS value

    for (int j = 0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] *
           currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum / currentDiagonal;
  }

  return 0;
}
