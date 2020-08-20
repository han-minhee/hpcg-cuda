#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif
#include <cfloat>
#include <fstream>
#include <iostream>
using std::endl;
#include <cmath>
#include <vector>

#include "hpcg.hpp"

#include "../cuda-src/TestSymmetryInside.cuh"
#include "ComputeDotProduct.hpp"
#include "ComputeMG.cpp"
#include "ComputeResidual.hpp"
#include "ComputeSPMV.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "TestSymmetry.hpp"

int TestSymmetry(SparseMatrix &A, Vector &b, Vector &xexact,
                 TestSymmetryData &testsymmetry_data) {

  return TestSymmetryInside(A, b, xexact, testsymmetry_data);
}
