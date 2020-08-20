#include "SparseMatrixInside.cuh"

#define LAUNCH_TO_ELL_COL(blocksizex, blocksizey)                              \
  kernel_to_ell_col<blocksizex, blocksizey>                                    \
      <<<dim3((A.localNumberOfRows - 1) / blocksizey + 1),                     \
         dim3(blocksizex, blocksizey)>>>(A.localNumberOfRows, A.ell_width,     \
                                         A.d_mtxIndL, A.ell_col_ind,           \
                                         d_halo_rows, A.halo_row_ind)

#define LAUNCH_TO_ELL_VAL(blocksizex, blocksizey)                              \
  kernel_to_ell_val<blocksizex, blocksizey>                                    \
      <<<dim3((A.localNumberOfRows - 1) / blocksizey + 1),                     \
         dim3(blocksizex, blocksizey)>>>(A.localNumberOfRows,                  \
                                         A.numberOfNonzerosPerRow,             \
                                         A.d_matrixValues, A.ell_val)

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_copy_diagonal(local_int_t m, local_int_t n,
                              local_int_t ell_width,
                              const local_int_t *__restrict__ ell_col_ind,
                              const double *__restrict__ ell_val,
                              double *__restrict__ diagonal) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= m) {
    return;
  }

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];

    if (col >= 0 && col < n) {
      if (col == row) {
        diagonal[row] = ell_val[idx];
        break;
      }
    } else {
      break;
    }
  }
}

void CudaCopyMatrixDiagonalInside(const SparseMatrix &A, Vector &diagonal) {
  kernel_copy_diagonal<1024>
      <<<dim3((A.localNumberOfRows - 1) / 1024 + 1), dim3(1024)>>>(
          A.localNumberOfRows, A.localNumberOfColumns, A.ell_width,
          A.ell_col_ind, A.ell_val, diagonal.d_values);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_replace_diagonal(local_int_t m, local_int_t n,
                                 const double *__restrict__ diagonal,
                                 local_int_t ell_width,
                                 const local_int_t *__restrict__ ell_col_ind,
                                 double *__restrict__ ell_val,
                                 double *__restrict__ inv_diag) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= m) {
    return;
  }

  double diag = diagonal[row];

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];

    if (col >= 0 && col < n) {
      if (col == row) {
        ell_val[idx] = diag;
        break;
      }
    } else {
      break;
    }
  }

  inv_diag[row] = 1.0 / diag;
}

void CudaReplaceMatrixDiagonalInside(SparseMatrix &A, const Vector &diagonal) {
  kernel_replace_diagonal<1024>
      <<<dim3((A.localNumberOfRows - 1) / 1024 + 1), dim3(1024)>>>(
          A.localNumberOfRows, A.localNumberOfColumns, diagonal.d_values,
          A.ell_width, A.ell_col_ind, A.ell_val, A.inv_diag);
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__
    void kernel_to_ell_col(local_int_t m, local_int_t nonzerosPerRow,
                           const local_int_t *__restrict__ mtxIndL,
                           local_int_t *__restrict__ ell_col_ind,
                           local_int_t *__restrict__ halo_rows,
                           local_int_t *__restrict__ halo_row_ind) {
  local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

#ifndef HPCG_NO_MPI
  __shared__ bool sdata[BLOCKSIZEY];
  sdata[threadIdx.y] = false;
  cudaDeviceSynchronize();
  __synchtreads();
#endif

  if (row >= m) {
    return;
  }

  local_int_t col = __ldg(mtxIndL + row * nonzerosPerRow + threadIdx.x);
  ell_col_ind[threadIdx.x * m + row] = col;

#ifndef HPCG_NO_MPI
  if (col >= m) {
    sdata[threadIdx.y] = true;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (sdata[threadIdx.y] == true) {
      halo_row_ind[atomicAdd(halo_rows, 1)] = row;
    }
  }
#endif
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__
    void kernel_to_ell_val(local_int_t m, local_int_t nnz_per_row,
                           const double *__restrict__ matrixValues,
                           double *__restrict__ ell_val) {
  local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

  if (row >= m) {
    return;
  }

  local_int_t idx = threadIdx.x * m + row;
  ell_val[idx] = matrixValues[row * nnz_per_row + threadIdx.x];
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_to_halo(local_int_t halo_rows, local_int_t m, local_int_t n,
                        local_int_t ell_width,
                        const local_int_t *__restrict__ ell_col_ind,
                        const double *__restrict__ ell_val,
                        const local_int_t *__restrict__ halo_row_ind,
                        local_int_t *__restrict__ halo_col_ind,
                        double *__restrict__ halo_val) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= halo_rows) {
    return;
  }

  local_int_t row = halo_row_ind[gid];

  int q = 0;
  for (int p = 0; p < ell_width; ++p) {
    local_int_t ell_idx = p * m + row;
    local_int_t col = ell_col_ind[ell_idx];

    if (col >= m && col < n) {
      local_int_t halo_idx = q++ * halo_rows + gid;

      halo_col_ind[halo_idx] = col;
      halo_val[halo_idx] = ell_val[ell_idx];
    }
  }

  for (; q < ell_width; ++q) {
    local_int_t idx = q * halo_rows + gid;
    halo_col_ind[idx] = -1;
  }
}

void ConvertToELLInside(SparseMatrix &A) {
  // We can re-use mtxIndL array for ELL values
  A.ell_val = reinterpret_cast<double *>(A.d_mtxIndG);
  A.d_mtxIndG = NULL;

  // Resize
  // TODO: implement cudaRealloc
  double *tempRealloc;
  cudaHostRealloc(A.ell_val, tempRealloc,
                  sizeof(double) * A.ell_width * A.localNumberOfRows);

  // Determine blocksize
  unsigned int blocksize = 1024 / A.ell_width;

  // Compute next power of two
  blocksize |= blocksize >> 1;
  blocksize |= blocksize >> 2;
  blocksize |= blocksize >> 4;
  blocksize |= blocksize >> 8;
  blocksize |= blocksize >> 16;
  ++blocksize;

  // Shift right until we obtain a valid blocksize
  while (blocksize * A.ell_width > 1024) {
    blocksize >>= 1;
  }

  if (blocksize == 32)
    LAUNCH_TO_ELL_VAL(27, 32);
  else if (blocksize == 16)
    LAUNCH_TO_ELL_VAL(27, 16);
  else if (blocksize == 8)
    LAUNCH_TO_ELL_VAL(27, 8);
  else
    LAUNCH_TO_ELL_VAL(27, 4);

  // We can re-use mtxIndG array for the ELL column indices
  A.ell_col_ind = reinterpret_cast<local_int_t *>(A.d_matrixValues);
  A.d_matrixValues = NULL;

  // Resize the array

  cudaHostRealloc(A.ell_col_ind, tempRealloc,
                  sizeof(local_int_t) * A.ell_width * A.localNumberOfRows);

  // Convert mtxIndL into ELL column indices
  local_int_t *d_halo_rows = reinterpret_cast<local_int_t *>(workspace);

#ifndef HPCG_NO_MPI
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&A.halo_row_ind,
                                sizeof(local_int_t) * A.totalToBeSent));

  CUDA_CHECK_COMMAND(cudaMemset(d_halo_rows, 0, sizeof(local_int_t)));
#endif

  if (blocksize == 32)
    LAUNCH_TO_ELL_COL(27, 32);
  else if (blocksize == 16)
    LAUNCH_TO_ELL_COL(27, 16);
  else if (blocksize == 8)
    LAUNCH_TO_ELL_COL(27, 8);
  else
    LAUNCH_TO_ELL_COL(27, 4);

  // Free old matrix indices
  CUDA_CHECK_COMMAND(cudaFree(A.d_mtxIndL));

#ifndef HPCG_NO_MPI
  CUDA_CHECK_COMMAND(cudaMemcpy(&A.halo_rows, d_halo_rows, sizeof(local_int_t),
                                cudaMemcpyDeviceToHost));
  assert(A.halo_rows <= A.totalToBeSent);

  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&A.halo_col_ind,
                 sizeof(local_int_t) * A.ell_width * A.halo_rows));
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&A.halo_val,
                                sizeof(double) * A.ell_width * A.halo_rows));

  size_t rocprim_size;
  void *rocprim_buffer = NULL;

  // TODO: FIX radix sort
  CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortKeys(
      rocprim_buffer, rocprim_size, A.halo_row_ind, A.halo_row_ind,
      A.halo_rows));
  CUDA_CHECK_COMMAND(cudaMalloc(&rocprim_buffer, rocprim_size));
  CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortKeys(
      rocprim_buffer, rocprim_size, A.halo_row_ind,
      A.halo_row_ind, // TODO inplace!
      A.halo_rows));
  CUDA_CHECK_COMMAND(cudaFree(rocprim_buffer));

  kernel_to_halo<128><<<dim3((A.halo_rows - 1) / 128 + 1), dim3(128)>>>(
      A.halo_rows, A.localNumberOfRows, A.localNumberOfColumns, A.ell_width,
      A.ell_col_ind, A.ell_val, A.halo_row_ind, A.halo_col_ind, A.halo_val);
#endif
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_extract_diag_index(local_int_t m, local_int_t ell_width,
                                   const local_int_t *__restrict__ ell_col_ind,
                                   const double *__restrict__ ell_val,
                                   local_int_t *__restrict__ diag_idx,
                                   double *__restrict__ inv_diag) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= m) {
    return;
  }

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];

    if (col == row) {
      diag_idx[row] = p;
      inv_diag[row] = 1.0 / ell_val[idx];
      break;
    }
  }
}

void ExtractDiagonalInside(SparseMatrix &A) {
  local_int_t m = A.localNumberOfRows;
  // Allocate memory to extract diagonal entries
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&A.diag_idx, sizeof(local_int_t) * m));
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&A.inv_diag, sizeof(double) * m));
  // Extract diagonal entries
  kernel_extract_diag_index<1024><<<dim3((m - 1) / 1024 + 1), dim3(1024)>>>(
      m, A.ell_width, A.ell_col_ind, A.ell_val, A.diag_idx, A.inv_diag);
}
