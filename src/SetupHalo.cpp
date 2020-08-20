
#include "SetupHalo.hpp"
#include "../cuda-src/SetupHaloInside.cuh"

void SetupHalo(SparseMatrix &A) { return SetupHaloInside(A); }

void CopyHaloToHost(SparseMatrix &A) { return CopyHaloToHostInside(A); }
