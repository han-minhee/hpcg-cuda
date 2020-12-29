#include "../src/ExchangeHalo.hpp"
#include "ComputeSPMVInside.cuh"

#include <cstddef>
#include <cuda_runtime.h>

#define LAUNCH_SPMV_ELL(blocksize)                                             \
  kernel_spmv_ell<blocksize><<<dim3((A.localNumberOfRows - 1) / blocksize + 1),  \
                             dim3(blocksize), 0, stream_interior>>>(           \
      A.localNumberOfRows, A.nblocks, A.localNumberOfRows / A.nblocks,         \
      A.ell_width, A.ell_col_ind, A.ell_val, x.d_values, y.d_values)
// 0, stream_interior,

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_spmv_ell_coarse(local_int_t size, local_int_t m, local_int_t n,
                             local_int_t ell_width,
                             const local_int_t *ell_col_ind,
                             const double *ell_val, const local_int_t *perm,
                             const local_int_t *f2cOperator, const double *x,
                             double *y) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  local_int_t f2c = f2cOperator[gid];
  local_int_t row = perm[f2c];

  double sum = 0.0;

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];

    if (col >= 0 && col < n) {
      sum = fma(ell_val[idx], __ldg(x + col), sum);
    } else {
      break;
    }
  }

  y[row] = sum;
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_spmv_ell(local_int_t m, int nblocks, local_int_t rows_per_block,
                       local_int_t ell_width, const local_int_t *ell_col_ind,
                       const double *ell_val, const double *x, double *y) {
  // Applies for chunks of blockDim.x * nblocks
  local_int_t color_block_offset = BLOCKSIZE * (blockIdx.x / nblocks);

  // Applies for chunks of blockDim.x and restarts for each
  // color_block_offset
  local_int_t thread_block_offset =
      (blockIdx.x & (nblocks - 1)) * rows_per_block;

  // Row entry point
  local_int_t row = color_block_offset + thread_block_offset + threadIdx.x;

  if (row >= m) {
    return;
  }

  double sum = 0.0;

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];

    if (col >= 0 && col < m) {
      sum = fma(ell_val[idx], x[col], sum);
    } else {
      break;
    }
  }

  y[row] = sum;
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_spmv_halo(local_int_t m, local_int_t n, local_int_t halo_width,
                          const local_int_t *halo_row_ind,
                          const local_int_t *halo_col_ind,
                          const double *halo_val, const local_int_t *perm,
                          const double *x, double *y) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= m) {
    return;
  }

  double sum = 0.0;

  for (local_int_t p = 0; p < halo_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = halo_col_ind[idx];

    if (col >= 0 && col < n) {
      sum = fma(halo_val[idx], x[col], sum);
    }
  }

  y[perm[halo_row_ind[row]]] += sum;
}

int ComputeSPMVInside(const SparseMatrix &A, Vector &x, Vector &y) {
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  if (A.geom->size > 1) {
    PrepareSendBuffer(A, x);
  }
#endif

  // double* ellVals = new double[10];
  // cudaMemcpy(ellVals, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);

  // for (int i = 0; i<10; i++){
  //   // printf("input ellval[%d] : %f\n", i, ellVals[i]);
  // }

  // free(ellVals);

  // double * AVals = new double[10];
  // double * xVals = new double[10];
  // double * yVals = new double[10];

  // cudaMemcpy(AVals, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
  // cudaMemcpy(xVals, x.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
  // cudaMemcpy(yVals, y.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

  // // printf("before MV\n");
  // for (int i = 0 ; i<10 ; i++){
  //   // printf("A, x, y [%d] : %f %f %f\n",i, AVals[i], xVals[i], yVals[i]);
  // }

  if (&y != A.mgData->Axf) {
    // Number of rows per block
    local_int_t rows_per_block = A.localNumberOfRows / A.nblocks;

    // Determine blocksize
    unsigned int blocksize = 512;

    while (rows_per_block & (blocksize - 1)) {
      blocksize >>= 1;
    }

    if (blocksize == 512)
      LAUNCH_SPMV_ELL(512);
    else if (blocksize == 256)
      LAUNCH_SPMV_ELL(256);
    else if (blocksize == 128)
      LAUNCH_SPMV_ELL(128);
    else
      LAUNCH_SPMV_ELL(64);
  }

#ifndef HPCG_NO_MPI
  if (A.geom->size > 1) {
    ExchangeHaloAsync(A);
    ObtainRecvBuffer(A, x);

    if (&y != A.mgData->Axf) {
      kernel_spmv_halo<128><<<dim3((A.halo_rows - 1) / 128 + 1), dim3(128)>>>(
          A.halo_rows, A.localNumberOfColumns, A.ell_width, A.halo_row_ind,
          A.halo_col_ind, A.halo_val, A.perm, x.d_values, y.d_values);
    }
  }
#endif

  // double* ellVals = new double[10];
  // cudaMemcpy(ellVals, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);

  // for (int i = 0; i<10; i++){
  //   // printf("input ellval[%d] : %f\n", i, ellVals[i]);
  // }

  // free(ellVals);

  if (&y == A.mgData->Axf) {
    kernel_spmv_ell_coarse<1024>
        <<<dim3((A.mgData->rc->localLength - 1) / 1024 + 1), dim3(1024)>>>(
            A.mgData->rc->localLength, A.localNumberOfRows,
            A.localNumberOfColumns, A.ell_width, A.ell_col_ind, A.ell_val,
            A.perm, A.mgData->d_f2cOperator, x.d_values, y.d_values);
  }

  // cudaMemcpy(y.values, y.d_values, sizeof(double) * y.localLength, cudaMemcpyDeviceToHost);

  // cudaMemcpy(AVals, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
  // cudaMemcpy(xVals, x.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
  // cudaMemcpy(yVals, y.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

  // // printf("after MV\n");
  // for (int i = 0 ; i<10 ; i++){
  //   // printf("A, x, y [%d] : %f %f %f\n",i, AVals[i], xVals[i], yVals[i]);
  // }

  // bool caught = false;
  // for (int i = 0; i<32*32*32; i++){
  //   if(y.values[i] != 0){
  //     // printf("Ap %d val: %f \n", i, y.values[i]);
  //     caught = true;
  //     break;
  //   }
  // }
  // if(caught) return -1;
  // // printf("finished SPMV\n");



  // free(AVals);
  // free(xVals);
  // free(yVals);

  return 0;
}
