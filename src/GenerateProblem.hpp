#ifndef GENERATEPROBLEM_HPP
#define GENERATEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

void GenerateProblem(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact);
void CopyProblemToHost(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact);

#endif // GENERATEPROBLEM_HPP
