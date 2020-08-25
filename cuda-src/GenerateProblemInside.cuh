
#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"

void GenerateProblemInside(SparseMatrix &A, Vector *b, Vector *x,
                           Vector *xexact);
                           
void CopyProblemToHostInside(SparseMatrix &A, Vector *b, Vector *x,
                             Vector *xexact);
