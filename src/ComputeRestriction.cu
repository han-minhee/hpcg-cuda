
#include "ComputeRestriction.cuh"
#include "ExchangeHalo.cuh"

#include <cuda_runtime.h>

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_restrict(local_int_t size,
                         const local_int_t *__restrict__ f2cOperator,
                         const double *__restrict__ fine,
                         const double *__restrict__ data,
                         double *__restrict__ coarse,
                         const local_int_t *__restrict__ perm_fine,
                         const local_int_t *__restrict__ perm_coarse) {
  local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (idx_coarse >= size) {
    return;
  }

  local_int_t idx_fine = perm_fine[f2cOperator[idx_coarse]];

  coarse[perm_coarse[idx_coarse]] = fine[idx_fine] - data[idx_fine];
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__ void kernel_fused_restrict_spmv(
    local_int_t size, const local_int_t *__restrict__ f2cOperator,
    const double *__restrict__ fine, local_int_t m, local_int_t n,
    local_int_t ell_width, const local_int_t *__restrict__ ell_col_ind,
    const double *__restrict__ ell_val, const double *__restrict__ xf,
    double *__restrict__ coarse, const local_int_t *__restrict__ perm_fine,
    const local_int_t *__restrict__ perm_coarse) {
  local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (idx_coarse >= size) {
    return;
  }

  local_int_t idx_f2c = __builtin_nontemporal_load(f2cOperator + idx_coarse);
  local_int_t idx_fine = __builtin_nontemporal_load(perm_fine + idx_f2c);

  double sum = 0.0;

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + idx_fine;
    local_int_t col = __builtin_nontemporal_load(ell_col_ind + idx);

    if (col >= 0 && col < n) {
      sum =
          fma(__builtin_nontemporal_load(ell_val + idx), __ldg(xf + col), sum);
    } else {
      break;
    }
  }

  local_int_t idx_perm = __builtin_nontemporal_load(perm_coarse + idx_coarse);
  double val_fine = __builtin_nontemporal_load(fine + idx_fine);
  __builtin_nontemporal_store(val_fine - sum, coarse + idx_perm);
}

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf,
  the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into
  corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction(const SparseMatrix &A, const Vector &rf) {
  kernel_restrict<128>
      <<<dim3((A.mgData->rc->localLength - 1) / 128 + 1), dim3(128)>>>(
          A.mgData->rc->localLength, A.mgData->d_f2cOperator, rf.d_values,
          A.mgData->Axf->d_values, A.mgData->rc->d_values, A.perm, A.Ac->perm);

  return 0;
}

int ComputeFusedSpMVRestriction(const SparseMatrix &A, const Vector &rf,
                                Vector &xf) {
#ifndef HPCG_NO_MPI
  if (A.geom->size > 1) {
    PrepareSendBuffer(A, xf);
    ExchangeHaloAsync(A);
    ObtainRecvBuffer(A, xf);
  }
#endif

  kernel_fused_restrict_spmv<1024>
      <<<dim3((A.mgData->rc->localLength - 1) / 1024 + 1), dim3(1024)>>>(
          A.mgData->rc->localLength, A.mgData->d_f2cOperator, rf.d_values,
          A.localNumberOfRows, A.localNumberOfColumns, A.ell_width,
          A.ell_col_ind, A.ell_val, xf.d_values, A.mgData->rc->d_values, A.perm,
          A.Ac->perm);

  return 0;
}
