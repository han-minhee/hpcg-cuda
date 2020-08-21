#include "TestSymmetry.hpp"
#include "../cuda-src/TestSymmetryInside.cuh"

int TestSymmetry(SparseMatrix &A, Vector &b, Vector &xexact,
                 TestSymmetryData &testsymmetry_data) {

  return TestSymmetryInside(A, b, xexact, testsymmetry_data);
}
