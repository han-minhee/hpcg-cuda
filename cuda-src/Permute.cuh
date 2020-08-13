#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"
#include "cuda_runtime.h"

void PermuteColumns(SparseMatrix &A);
void PermuteRows(SparseMatrix &A);
void PermuteVector(local_int_t size, Vector &v, const local_int_t *perm);
