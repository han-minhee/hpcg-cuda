#include "../src/TestCG.hpp"
#include <cuda_runtime.h>

int TestCGInside(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
                 TestCGData &testcg_data);
