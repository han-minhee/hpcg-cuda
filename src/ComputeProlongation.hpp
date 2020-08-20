
#ifndef COMPUTEPROLONGATION_HPP
#define COMPUTEPROLONGATION_HPP

#include "../cuda-src/ComputeProlongationInside.cuh"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeProlongation(const SparseMatrix &Af, Vector &xf);

#endif