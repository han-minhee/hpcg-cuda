
#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "CGData.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
                    Vector &xexact);
// double OptimizeProblemMemoryUse(const SparseMatrix &A);

#endif // OPTIMIZEPROBLEM_HPP
