#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP
#include "Vector.cuh"
#include "SparseMatrix.cuh"

int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y);

#endif  // COMPUTESPMV_HPP
