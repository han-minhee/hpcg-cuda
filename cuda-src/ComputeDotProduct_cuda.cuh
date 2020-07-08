/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for
  the purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication
  between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/

#include "../src/Vector.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>

int ComputeDotProduct_cuda(const local_int_t n, const Vector &x,
                           const Vector &y, double &result,
                           double &time_allreduce);