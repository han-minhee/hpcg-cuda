#ifndef COMPUTEMG_HPP
#define COMPUTEMG_HPP

#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"

int ComputeMGInside(const SparseMatrix &A, const Vector &r, Vector &x);

#endif // COMPUTEMG_HPP
