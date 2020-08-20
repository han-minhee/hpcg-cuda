
#ifndef PERMUTE_HPP
#define PERMUTE_HPP

#include "../src/SparseMatrix.hpp"

void PermuteColumns(SparseMatrix &A);
void PermuteRows(SparseMatrix &A);
void PermuteVector(local_int_t size, Vector &v, const local_int_t *perm);

#endif // PERMUTE_HPP
