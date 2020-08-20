#ifndef TESTSYMMETRY_HPP
#define TESTSYMMETRY_HPP

#include "CGData.cuh"
#include "SparseMatrix.cuh"
#include "hpcg.cuh"

struct TestSymmetryData_STRUCT {
  double depsym_spmv; //!< departure from symmetry for the SPMV kernel
  double depsym_mg;   //!< departure from symmetry for the MG kernel
  int count_fail;     //!< number of failures in the symmetry tests
};
typedef struct TestSymmetryData_STRUCT TestSymmetryData;

extern int TestSymmetry(SparseMatrix &A, Vector &b, Vector &xexact,
                        TestSymmetryData &testsymmetry_data);

#endif // TESTSYMMETRY_HPP
