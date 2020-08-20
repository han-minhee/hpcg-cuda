
#ifndef COMPUTEMG_HPP
#define COMPUTEMG_HPP
#include "SparseMatrix.cuh"
#include "Vector.cuh"

int ComputeMG(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTEMG_HPP
