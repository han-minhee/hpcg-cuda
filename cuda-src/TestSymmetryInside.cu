#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif
#include <cfloat>
#include <fstream>
#include <iostream>
using std::endl;
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/hpcg.hpp"

#include "../src/ComputeDotProduct.hpp"
#include "../src/ComputeMG.hpp"
#include "../src/ComputeResidual.hpp"
#include "../src/ComputeSPMV.hpp"
#include "../src/Geometry.hpp"
#include "TestSymmetryInside.cuh"

int TestSymmetryInside(SparseMatrix &A, Vector &b, Vector &xexact,
                       TestSymmetryData &testsymmetry_data) {
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_ncol, y_ncol, z_ncol;
  CudaInitializeVector(x_ncol, ncol);
  CudaInitializeVector(y_ncol, ncol);
  CudaInitializeVector(z_ncol, ncol);

  double t4 = 0.0; // Needed for dot-product call, otherwise unused
  testsymmetry_data.count_fail = 0;

  // Test symmetry of matrix
  // First load vectors with random values
  CudaFillRandomVector(x_ncol);
  CudaFillRandomVector(y_ncol);

  double xNorm2, yNorm2;
  double ANorm = 2 * 26.0;

  // Next, compute x'*A*y
  ComputeDotProduct(nrow, y_ncol, y_ncol, yNorm2, t4, A.isDotProductOptimized);
  int ierr = ComputeSPMV(A, y_ncol, z_ncol); // z_nrow = A*y_overlap
  if (ierr)
    HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
  double xtAy = 0.0;
  ierr = ComputeDotProduct(nrow, x_ncol, z_ncol, xtAy, t4,
                           A.isDotProductOptimized); // x'*A*y
  if (ierr)
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

  // Next, compute y'*A*x
  ComputeDotProduct(nrow, x_ncol, x_ncol, xNorm2, t4, A.isDotProductOptimized);
  ierr = ComputeSPMV(A, x_ncol, z_ncol); // b_computed = A*x_overlap
  if (ierr)
    HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
  double ytAx = 0.0;
  ierr = ComputeDotProduct(nrow, y_ncol, z_ncol, ytAx, t4,
                           A.isDotProductOptimized); // y'*A*x
  if (ierr)
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

  testsymmetry_data.depsym_spmv =
      std::fabs((long double)(xtAy - ytAx)) /
      ((xNorm2 * ANorm * yNorm2 + yNorm2 * ANorm * xNorm2) * (DBL_EPSILON));
  if (testsymmetry_data.depsym_spmv > 1.0)
    ++testsymmetry_data.count_fail; // If the difference is > 1, count it wrong
  if (A.geom->rank == 0)
    HPCG_fout
        << "Departure from symmetry (scaled) for SpMV abs(x'*A*y - y'*A*x) = "
        << testsymmetry_data.depsym_spmv << endl;

  // Test symmetry of multi-grid

  // Compute x'*Minv*y
  ierr = ComputeMG(A, y_ncol, z_ncol); // z_ncol = Minv*y_ncol
  if (ierr)
    HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  double xtMinvy = 0.0;
  ierr = ComputeDotProduct(nrow, x_ncol, z_ncol, xtMinvy, t4,
                           A.isDotProductOptimized); // x'*Minv*y
  if (ierr)
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

  // Next, compute z'*Minv*x
  ierr = ComputeMG(A, x_ncol, z_ncol); // z_ncol = Minv*x_ncol
  if (ierr)
    HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  double ytMinvx = 0.0;
  ierr = ComputeDotProduct(nrow, y_ncol, z_ncol, ytMinvx, t4,
                           A.isDotProductOptimized); // y'*Minv*x
  if (ierr)
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

  testsymmetry_data.depsym_mg =
      std::fabs((long double)(xtMinvy - ytMinvx)) /
      ((xNorm2 * ANorm * yNorm2 + yNorm2 * ANorm * xNorm2) * (DBL_EPSILON));
  if (testsymmetry_data.depsym_mg > 1.0)
    ++testsymmetry_data.count_fail; // If the difference is > 1, count it wrong
  if (A.geom->rank == 0)
    HPCG_fout << "Departure from symmetry (scaled) for MG abs(x'*Minv*y - "
                 "y'*Minv*x) = "
              << testsymmetry_data.depsym_mg << endl;

  CudaCopyVector(xexact, x_ncol); // Copy exact answer into overlap vector

  int numberOfCalls = 2;
  double residual = 0.0;
  for (int i = 0; i < numberOfCalls; ++i) {
    ierr = ComputeSPMV(A, x_ncol, z_ncol); // b_computed = A*x_overlap
    if (ierr)
      HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    if ((ierr = ComputeResidual(A.localNumberOfRows, b, z_ncol, residual)))
      HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n"
                << endl;
    if (A.geom->rank == 0)
      HPCG_fout << "SpMV call [" << i << "] Residual [" << residual << "]"
                << endl;
  }
  CudaDeleteVector(x_ncol);
  CudaDeleteVector(y_ncol);
  CudaDeleteVector(z_ncol);

  return 0;
}
