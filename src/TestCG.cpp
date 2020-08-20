#include "TestCG.hpp"

#include "../cuda-src/TestCGInside.cuh"


int TestCG(SparseMatrix &A, CGData &data, Vector &b, Vector &x,
           TestCGData &testcg_data) {
             return TestCGInside(A, data, b, x, testcg_data);
}
