

#include "ComputeResidual.hpp"
#include "../cuda-src/ComputeResidualInside.cuh"

int ComputeResidual(local_int_t n, const Vector &v1, const Vector &v2,
                    double &residual) {
  return ComputeResidualInside(n, v1, v2, residual);
}
