#ifndef SETUPHALO_HPP
#define SETUPHALO_HPP

#include "SparseMatrix.hpp"

void SetupHalo(SparseMatrix &A);
void CopyHaloToHost(SparseMatrix &A);

#endif // SETUPHALO_HPP
