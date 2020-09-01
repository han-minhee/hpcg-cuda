#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#include "../cuda-src/ExchangeHaloInside.cuh"
#include "Geometry.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>

/*!
  Communicates data that is at the border of the part of the domain assigned to
  this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be
  communicated; on exit: the vector with non-local entries updated by other
  processors
 */
void ExchangeHalo(const SparseMatrix &A, Vector &x) {
  return ExchangeHaloInside(A, x);
}

void ExchangeHaloAsync(const SparseMatrix &A) {
  return ExchangeHaloAsyncInside(A);
}

void PrepareSendBuffer(const SparseMatrix &A, const Vector &x){
  return PrepareSendBufferInside(A, x);
}

void ObtainRecvBuffer(const SparseMatrix &A, Vector &x) {
  return ObtainRecvBufferInside(A, x);
}
#endif
// ifndef HPCG_NO_MPI
