#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"

void ExchangeHaloInside(const SparseMatrix &A, Vector &x);

void PrepareSendBufferInside(const SparseMatrix &A, const Vector &x);
void ExchangeHaloAsyncInside(const SparseMatrix &A);
void ObtainRecvBufferInside(const SparseMatrix &A, Vector &x);
