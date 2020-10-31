
#include "../src/ExchangeHalo.hpp"
#include "ComputeRestrictionInside.cuh"

#include <cuda_runtime.h>

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_restrict(local_int_t size, const local_int_t *f2cOperator,
                         const double *fine, const double *data, double *coarse,
                         const local_int_t *perm_fine,
                         const local_int_t *perm_coarse) {
  local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (idx_coarse >= size) {
    return;
  }

  local_int_t idx_fine = perm_fine[f2cOperator[idx_coarse]];

  coarse[perm_coarse[idx_coarse]] = fine[idx_fine] - data[idx_fine];
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__ void kernel_fused_restrict_spmv(
    local_int_t size, const local_int_t *f2cOperator, const double *fine,
    local_int_t m, local_int_t n, local_int_t ell_width,
    const local_int_t *ell_col_ind, const double *ell_val, const double *xf,
    double *coarse, const local_int_t *perm_fine,
    const local_int_t *perm_coarse) {
  local_int_t idx_coarse = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (idx_coarse >= size) {
    return;
  }

  local_int_t idx_f2c = f2cOperator[idx_coarse];
  local_int_t idx_fine = perm_fine[idx_f2c];

  double sum = 0.0;

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + idx_fine;
    local_int_t col = ell_col_ind[idx];

    if (col >= 0 && col < n) {
      sum = fma(ell_val[idx], __ldg(xf + col), sum);
    } else {
      break;
    }
  }

  local_int_t idx_perm = perm_coarse[idx_coarse];
  double val_fine = fine[idx_fine];
  coarse[idx_perm] = val_fine - sum;
}

int ComputeRestriction(const SparseMatrix &A, const Vector &rf) {

  printf("entering restriction\n");
  
  double *rfv = new double[50 /*length*/];
  double *rcv = new double[50 /*length*/];
  double *Axfv = new double[50 /*length*/];

  cudaMemcpy(rfv, rf.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  cudaMemcpy(rcv, A.mgData->rc->d_values, sizeof(double) * 50 /*length*/,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Axfv, A.mgData->Axf->d_values, sizeof(double) * 50 /*length*/,
             cudaMemcpyDeviceToHost);

  for (int j = 0; j < 50 /*length*/; j++) {
    printf(
        "inside before restriction: rfv[%d]: %f, rcv[%d]: %f, Axfv[%d] : %f\n",
        j, rfv[j], j, rcv[j], j, Axfv[j]);
  }

  kernel_restrict<128>
      <<<dim3((A.mgData->rc->localLength - 1) / 128 + 1), dim3(128)>>>(
          A.mgData->rc->localLength, A.mgData->d_f2cOperator, rf.d_values,
          A.mgData->Axf->d_values, A.mgData->rc->d_values, A.perm, A.Ac->perm);

  cudaMemcpy(rfv, rf.d_values, sizeof(double) * 50, cudaMemcpyDeviceToHost);
  cudaMemcpy(rcv, A.mgData->rc->d_values, sizeof(double) * 50,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Axfv, A.mgData->Axf->d_values, sizeof(double) * 50,
             cudaMemcpyDeviceToHost);

  for (int j = 0; j < 50 /*length*/; j++) {
    printf(
        "inside after restriction: rfv[%d]: %f, rcv[%d]: %f, Axfv[%d] : %f\n",
        j, rfv[j], j, rcv[j], j, Axfv[j]);
  }

  delete [] rfv;
  delete [] rcv;
  delete [] Axfv;
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

  printf("entering SpMVrestriction\n");
  double *rfv = new double[50 /*length*/];
  double *rcv = new double[50 /*length*/];
  double *Axfv = new double[50 /*length*/];
  double *ellVals = new double[50 /*length*/];
  cudaMemcpy(rfv, rf.d_values, sizeof(double) * 50, cudaMemcpyDeviceToHost);
  cudaMemcpy(rcv, A.mgData->rc->d_values, sizeof(double) * 50,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Axfv, A.mgData->Axf->d_values, sizeof(double) * 50,
             cudaMemcpyDeviceToHost);
             
  cudaMemcpy(ellVals, A.ell_val, sizeof(double) * 50,
  cudaMemcpyDeviceToHost);
  

  for (int j = 0; j < 50 /*length*/; j++) {
    printf(
        "inside before SpMVrestriction: rfv[%d]: %f, rcv[%d]: %f, Axfv[%d]: %f, ellVal[%d] : %f\n",
        j, rfv[j], j, rcv[j], j, Axfv[j], j, ellVals[j]);
  }

  kernel_fused_restrict_spmv<1024>
      <<<dim3((A.mgData->rc->localLength - 1) / 1024 + 1), dim3(1024)>>>(
          A.mgData->rc->localLength, A.mgData->d_f2cOperator, rf.d_values,
          A.localNumberOfRows, A.localNumberOfColumns, A.ell_width,
          A.ell_col_ind, A.ell_val, xf.d_values, A.mgData->rc->d_values, A.perm,
          A.Ac->perm);

          cudaMemcpy(rfv, rf.d_values, sizeof(double) * 50, cudaMemcpyDeviceToHost);
          cudaMemcpy(rcv, A.mgData->rc->d_values, sizeof(double) * 50,
                     cudaMemcpyDeviceToHost);
          cudaMemcpy(Axfv, A.mgData->Axf->d_values, sizeof(double) * 50,
                     cudaMemcpyDeviceToHost);
        
          for (int j = 0; j < 50 /*length*/; j++) {
            printf(
                "inside after SpMVrestriction: rfv[%d]: %f, rcv[%d]: %f, Axfv[%d] : %f\n",
                j, rfv[j], j, rcv[j], j, Axfv[j]);
          }
        

  delete [] rfv;
  delete [] rcv;
  delete [] Axfv;

  return 0;
}
