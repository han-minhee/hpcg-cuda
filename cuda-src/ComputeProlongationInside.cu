#include "ComputeProlongation.hpp"
#include <cuda_runtime.h>

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_prolongation(local_int_t size,
                             const local_int_t *__restrict__ f2cOperator,
                             const double *__restrict__ coarse,
                             double *__restrict__ fine,
                             const local_int_t *__restrict__ perm_fine,
                             const local_int_t *__restrict__ perm_coarse) {
  local_int_t idx_coarse = blockIdx.x * 1024 + threadIdx.x;

  if (idx_coarse >= size) {
    return;
  }

  local_int_t idx_fine = f2cOperator[idx_coarse];

  fine[perm_fine[idx_fine]] += coarse[perm_coarse[idx_coarse]];
}

int ComputeProlongationInside(const SparseMatrix &Af, Vector &xf) {
  kernel_prolongation<1024>
      <<<dim3((Af.mgData->rc->localLength - 1) / 1024 + 1), dim3(1024)>>>(
          Af.mgData->rc->localLength, Af.mgData->d_f2cOperator,
          Af.mgData->xc->d_values, xf.d_values, Af.perm, Af.Ac->perm);

  return 0;
}
