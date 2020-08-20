#ifndef COMPUTERESTRICTION_HPP
#define COMPUTERESTRICTION_HPP

#include "../cuda-src/ComputeRestrictionInside.cuh"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeRestriction(const SparseMatrix &A, const Vector &rf);
int ComputeFusedSpMVRestriction(const SparseMatrix &A, const Vector &rf,
                                Vector &xf);

#endif // COMPUTERESTRICTION_HPP
