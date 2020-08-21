#ifndef GENERATEPROBLEM_HPP
#define GENERATEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

inline void GenerateProblem(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact);
inline void CopyProblemToHost(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact);

#endif // GENERATEPROBLEM_HPP
