#ifndef HPCG_NO_MPI
#include "mytimer.hpp"
#include <mpi.h>
#endif

#include "ComputeDotProduct.hpp"

int ComputeDotProduct(local_int_t n, const Vector &x, const Vector &y,
                      double &result, double &time_allreduce,
                      bool &isOptimized) {
  return ComputeDotProductInside(n, x, y, result, time_allreduce, isOptimized);
}
