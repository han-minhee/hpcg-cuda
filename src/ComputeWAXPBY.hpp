#ifndef COMPUTEWAXPBY_HPP
#define COMPUTEWAXPBY_HPP
#include "Vector.hpp"

inline int ComputeWAXPBY(local_int_t n, double alpha, const Vector &x, double beta,
                  const Vector &y, Vector &w, bool &isOptimized);

inline int ComputeFusedWAXPBYDot(local_int_t n, double alpha, const Vector &x,
                          Vector &y, double &result, double &time_allreduce);

#endif // COMPUTEWAXPBY_HPP
