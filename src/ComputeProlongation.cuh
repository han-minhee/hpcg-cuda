
#ifndef COMPUTEPROLONGATION_HPP
#define COMPUTEPROLONGATION_HPP

#include "Vector.cuh"
#include "SparseMatrix.cuh"

int ComputeProlongation(const SparseMatrix& Af, Vector& xf);

#endif 