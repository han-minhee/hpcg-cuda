
#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

void ExchangeHalo(const SparseMatrix &A, Vector &x);

void PrepareSendBuffer(const SparseMatrix &A, const Vector &x);
void ExchangeHaloAsync(const SparseMatrix &A);
void ObtainRecvBuffer(const SparseMatrix &A, Vector &x);

#endif // EXCHANGEHALO_HPP
