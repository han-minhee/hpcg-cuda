#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"

int ComputeSYMGSInside(const SparseMatrix &A, const Vector &r, Vector &x);
int ComputeSYMGSZeroGuessInside(const SparseMatrix &A, const Vector &r,
                                Vector &x);