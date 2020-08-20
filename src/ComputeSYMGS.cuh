
#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP

#include "SparseMatrix.cuh"
#include "Vector.cuh"

int ComputeSYMGS(const SparseMatrix & A, const Vector& r, Vector& x);
int ComputeSYMGSZeroGuess(const SparseMatrix & A, const Vector& r, Vector& x);

#endif // COMPUTESYMGS_HPP
