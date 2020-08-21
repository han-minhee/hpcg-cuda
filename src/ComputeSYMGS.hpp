
#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeSYMGS(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGSZeroGuess(const SparseMatrix & A, const Vector& r, Vector& x);

#endif // COMPUTESYMGS_HPP
