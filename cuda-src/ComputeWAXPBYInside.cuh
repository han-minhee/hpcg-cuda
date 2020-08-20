#include "../src/Vector.hpp"

int ComputeWAXPBYInside(local_int_t n, double alpha, const Vector &x,
                        double beta, const Vector &y, Vector &w,
                        bool &isOptimized);

int ComputeFusedWAXPBYDotInside(local_int_t n, double alpha, const Vector &x,
                                Vector &y, double &result,
                                double &time_allreduce);
