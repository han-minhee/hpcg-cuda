
#ifndef COMPUTEDOTPRODUCT_HPP
#define COMPUTEDOTPRODUCT_HPP
#include "Vector.hpp"
#include "../cuda-src/ComputeDotProductInside.cuh"

int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized);

#endif // COMPUTEDOTPRODUCT_HPP
