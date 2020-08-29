#include "Permute.cuh"
#include "Utils.cuh"

#include <cuda_runtime.h>

#define LAUNCH_PERM_COLS(blockSizeX, blockSizeY)                               \
  kernel_perm_cols<blockSizeX, blockSizeY>                                     \
      <<<dim3((A.localNumberOfRows - 1) / blockSizeY + 1),                     \
         dim3(blockSizeX, blockSizeY)>>>(                                      \
          A.localNumberOfRows, A.localNumberOfColumns,                         \
          A.numberOfNonzerosPerRow, A.perm, A.d_mtxIndL, A.d_matrixValues)

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_permute_ell_rows(local_int_t m, local_int_t p,
                                 const local_int_t *tmp_cols,
                                 const double *tmp_vals,
                                 const local_int_t *perm,
                                 local_int_t *ell_col_ind, double *ell_val) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= m) {
    return;
  }

  local_int_t idx = p * m + perm[row];
  local_int_t col = tmp_cols[row];

  ell_col_ind[idx] = col;
  ell_val[idx] = tmp_vals[row];
}

__device__ void swap(local_int_t &key, double &val, int mask, int dir) {

  // 32 is for CUDA, temporarily
  local_int_t key1 = __shfl_xor(key, mask);
  __syncthreads();
  //__shfl_xor_sync()(key, mask);
  double val1 = __shfl_xor(val, mask);
  //__shfl_xor_sync(val, mask);
  __syncthreads();
  if (key < key1 == dir) {
    key = key1;
    val = val1;
  }
}

__device__ int get_bit(int x, int i) { return (x >> i) & 1; }

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__
    void kernel_perm_cols(local_int_t m, local_int_t n,
                          local_int_t nonzerosPerRow, const local_int_t *perm,
                          local_int_t *mtxIndL, double *matrixValues) {
  local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;
  local_int_t idx = row * nonzerosPerRow + threadIdx.x;
  local_int_t key = n;
  double val = 0.0;

  if (threadIdx.x < nonzerosPerRow && row < m) {
    local_int_t col = mtxIndL[idx];
    val = matrixValues[idx];

    if (col >= 0 && col < m) {
      key = perm[col];
    } else if (col >= m && col < n) {
      key = col;
    }
  }

  swap(key, val, 1, get_bit(threadIdx.x, 1) ^ get_bit(threadIdx.x, 0));

  swap(key, val, 2, get_bit(threadIdx.x, 2) ^ get_bit(threadIdx.x, 1));
  swap(key, val, 1, get_bit(threadIdx.x, 2) ^ get_bit(threadIdx.x, 0));

  swap(key, val, 4, get_bit(threadIdx.x, 3) ^ get_bit(threadIdx.x, 2));
  swap(key, val, 2, get_bit(threadIdx.x, 3) ^ get_bit(threadIdx.x, 1));
  swap(key, val, 1, get_bit(threadIdx.x, 3) ^ get_bit(threadIdx.x, 0));

  swap(key, val, 8, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 3));
  swap(key, val, 4, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 2));
  swap(key, val, 2, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 1));
  swap(key, val, 1, get_bit(threadIdx.x, 4) ^ get_bit(threadIdx.x, 0));

  swap(key, val, 16, get_bit(threadIdx.x, 4));
  swap(key, val, 8, get_bit(threadIdx.x, 3));
  swap(key, val, 4, get_bit(threadIdx.x, 2));
  swap(key, val, 2, get_bit(threadIdx.x, 1));
  swap(key, val, 1, get_bit(threadIdx.x, 0));

  if (threadIdx.x < nonzerosPerRow && row < m) {
    mtxIndL[idx] = (key == n) ? -1 : key;
    matrixValues[idx] = val;
  }
}

void PermuteColumns(SparseMatrix &A) {
  // Determine blocksize in x direction
  unsigned int dim_x = A.numberOfNonzerosPerRow;

  // Compute next power of two
  dim_x |= dim_x >> 1;
  dim_x |= dim_x >> 2;
  dim_x |= dim_x >> 4;
  dim_x |= dim_x >> 8;
  dim_x |= dim_x >> 16;
  ++dim_x;

  // Determine blocksize
  unsigned int dim_y = 512 / dim_x;

  // Compute next power of two
  dim_y |= dim_y >> 1;
  dim_y |= dim_y >> 2;
  dim_y |= dim_y >> 4;
  dim_y |= dim_y >> 8;
  dim_y |= dim_y >> 16;
  ++dim_y;

  // Shift right until we obtain a valid blocksize
  while (dim_x * dim_y > 512) {
    dim_y >>= 1;
  }

  // double *perm = new double[10];
  // double *vals = new double[10];
  // local_int_t *mtxIndl = new local_int_t[10];

  // cudaMemcpy(perm, A.perm, sizeof(double) * 10, cudaMemcpyDeviceToHost);
  // cudaMemcpy(vals, A.d_matrixValues, sizeof(double) * 10,
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(mtxIndl, A.d_mtxIndL, sizeof(local_int_t) * 10,
  //            cudaMemcpyDeviceToHost);

  // printf("before launch perm\n");
  // for (int i = 0; i < 10; i++) {
  //   printf("perm, vals, mtxIndl [%d] : %f %f %d\n", i, perm[i], vals[i],
  //          mtxIndl[i]);
  // }

  if (dim_y == 32)
    LAUNCH_PERM_COLS(32, 32);
  else if (dim_y == 16)
    LAUNCH_PERM_COLS(32, 16);
  else if (dim_y == 8)
    LAUNCH_PERM_COLS(32, 8);
  else
    LAUNCH_PERM_COLS(32, 4);

  // cudaMemcpy(perm, A.perm, sizeof(double) * 10, cudaMemcpyDeviceToHost);
  // cudaMemcpy(vals, A.d_matrixValues, sizeof(double) * 10,
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(mtxIndl, A.d_mtxIndL, sizeof(local_int_t) * 10,
  //            cudaMemcpyDeviceToHost);

  // // printf("after launch perm\n");
  // // for (int i = 0; i < 10; i++) {
  // //   printf("perm, vals, mtxIndl [%d] : %f %f %d\n", i, perm[i], vals[i],
  // //          mtxIndl[i]);
  // // }

  // free(perm);
  // free(vals);
  // free(mtxIndl);
}

void PermuteRows(SparseMatrix &A) {
  local_int_t m = A.localNumberOfRows;

  // Temporary structures for row permutation
  local_int_t *tmp_cols;
  double *tmp_vals;

  CUDA_CHECK_COMMAND(cudaMalloc((void **)&tmp_cols, sizeof(local_int_t) * m));
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&tmp_vals, sizeof(double) * m));

  // Permute ELL rows
  for (local_int_t p = 0; p < A.ell_width; ++p) {
    local_int_t offset = p * m;

    CUDA_CHECK_COMMAND(cudaMemcpy(tmp_cols, A.ell_col_ind + offset,
                                  sizeof(local_int_t) * m,
                                  cudaMemcpyDeviceToDevice));
    CUDA_CHECK_COMMAND(cudaMemcpy(tmp_vals, A.ell_val + offset,
                                  sizeof(double) * m,
                                  cudaMemcpyDeviceToDevice));

    kernel_permute_ell_rows<1024><<<dim3((m - 1) / 1024 + 1), dim3(1024)>>>(
        m, p, tmp_cols, tmp_vals, A.perm, A.ell_col_ind, A.ell_val);
  }

  CUDA_CHECK_COMMAND(cudaFree(tmp_cols));
  CUDA_CHECK_COMMAND(cudaFree(tmp_vals));
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_permute(local_int_t size, const local_int_t *perm,
                        const double *in, double *out) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  out[perm[gid]] = in[gid];
}

void PermuteVector(local_int_t size, Vector &v, const local_int_t *perm) {
  double *buffer;
  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&buffer, sizeof(double) * v.localLength));

  kernel_permute<1024><<<dim3((size - 1) / 1024 + 1), dim3(1024)>>>(
      size, perm, v.d_values, buffer);

  CUDA_CHECK_COMMAND(cudaFree(v.d_values));
  v.d_values = buffer;
}
