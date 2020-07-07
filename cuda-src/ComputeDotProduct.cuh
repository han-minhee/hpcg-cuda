#include "Vector.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

int ComputeDotProduct_cuda(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce);