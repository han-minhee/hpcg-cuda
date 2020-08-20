#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y);

#endif // COMPUTESPMV_HPP
