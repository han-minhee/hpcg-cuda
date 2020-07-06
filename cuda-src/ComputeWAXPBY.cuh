#include "Vector.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

int ComputeWAXPBY_cuda(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized);

