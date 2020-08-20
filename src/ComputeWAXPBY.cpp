
#ifndef HPCG_NO_MPI
#include "mytimer.hpp"
#include <mpi.h>
#endif

#include <cassert>

#include "../cuda-src/ComputeWAXPBYInside.cuh"
#include "ComputeWAXPBY.hpp"

int ComputeWAXPBY(local_int_t n, double alpha, const Vector &x, double beta,
                  const Vector &y, Vector &w, bool &isOptimized) {
  return ComputeWAXPBYInside(n, alpha, x, beta, y, w, isOptimized);
}

int ComputeFusedWAXPBYDot(local_int_t n, double alpha, const Vector &x,
                          Vector &y, double &result, double &time_allreduce) {
  return ComputeFusedWAXPBYDotInside(n, alpha, x, y, result, time_allreduce);
}
