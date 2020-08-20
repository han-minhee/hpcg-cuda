#ifndef COMPUTERESTRICTION_HPP
#define COMPUTERESTRICTION_HPP

#include "Vector.cuh"
#include "SparseMatrix.cuh"

int ComputeRestriction(const SparseMatrix& A, const Vector& rf);
int ComputeFusedSpMVRestriction(const SparseMatrix& A, const Vector& rf, Vector& xf);

#endif // COMPUTERESTRICTION_HPP
