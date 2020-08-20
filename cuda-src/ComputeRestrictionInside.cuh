
#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"

int ComputeRestrictionInside(const SparseMatrix &A, const Vector &rf);
int ComputeFusedSpMVRestrictionInside(const SparseMatrix &A, const Vector &rf,
                                      Vector &xf);

#endif // COMPUTERESTRICTION_HPP
