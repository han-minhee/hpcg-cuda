#ifndef GENERATECOARSEPROBLEM_HPP
#define GENERATECOARSEPROBLEM_HPP

#include "SparseMatrix.cuh"

void GenerateCoarseProblem(const SparseMatrix& A);
void CopyCoarseProblemToHost(SparseMatrix& A);

#endif // GENERATECOARSEPROBLEM_HPP
