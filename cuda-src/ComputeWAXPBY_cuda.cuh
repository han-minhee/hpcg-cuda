
/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
#include "../src/Vector.hpp"
#include "InitCuda.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstdio>

int ComputeWAXPBY_cuda(const local_int_t n, const double alpha, const Vector &x,
                       const double beta, const Vector &y, Vector &w);

int ComputeWAXPBY_ref_cuda(const local_int_t n, const double alpha,
                           const Vector &x, const double beta, const Vector &y,
                           Vector &w);
