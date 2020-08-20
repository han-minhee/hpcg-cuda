#include "ComputeSYMGS.cuh"
#include "ExchangeHalo.cuh"

#include <cuda_runtime.h>

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__ void kernel_symgs_sweep(
    local_int_t m, local_int_t n, local_int_t block_nrow, local_int_t offset,
    local_int_t ell_width, const local_int_t *__restrict__ ell_col_ind,
    const double *__restrict__ ell_val, const double *__restrict__ inv_diag,
    const double *__restrict__ x, double *__restrict__ y) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= block_nrow) {
    return;
  }

  local_int_t row = gid + offset;

  double sum = x[row];

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];

    if (col >= 0 && col < n && col != row) {
      sum =
          fma(-ell_val[idx], __ldg(y + col), sum);
    }
  }

  y[row] = sum * inv_diag[row];
  // (sum * __builtin_nontemporal_load(inv_diag + row),
  //                             y + row);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_symgs_interior(local_int_t m, local_int_t block_nrow,
                               local_int_t ell_width,
                               const local_int_t *__restrict__ ell_col_ind,
                               const double *__restrict__ ell_val,
                               const double *__restrict__ inv_diag,
                               const double *__restrict__ x,
                               double *__restrict__ y) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= block_nrow) {
    return;
  }

  double sum =  x[row];
  // __builtin_nontemporal_load(x + row);

  for (local_int_t p = 0; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];
    // __builtin_nontemporal_load(ell_col_ind + idx);

    if (col >= 0 && col < m && col != row) {
      sum =
          fma(-ell_val[idx], __ldg(y + col), sum);
    }
  }

  y[row] = sum * inv_diag[row];
  // __builtin_nontemporal_store(sum * __builtin_nontemporal_load(inv_diag + row),
  //                             y + row);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_symgs_halo(local_int_t m, local_int_t n, local_int_t block_nrow,
                           local_int_t halo_width,
                           const local_int_t *__restrict__ halo_row_ind,
                           const local_int_t *__restrict__ halo_col_ind,
                           const double *__restrict__ halo_val,
                           const double *__restrict__ inv_diag,
                           const local_int_t *__restrict__ perm,
                           const double *__restrict__ x,
                           double *__restrict__ y) {
  local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row >= m) {
    return;
  }

  local_int_t halo_idx = halo_row_ind[row];
  local_int_t perm_idx = perm[halo_idx];

  if (perm_idx >= block_nrow) {
    return;
  }

  double sum = 0.0;

  for (local_int_t p = 0; p < halo_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = halo_col_ind[idx];

    if (col >= 0 && col < n) {
      sum = fma(-halo_val[idx], y[col], sum);
    }
  }

  y[perm_idx] = fma(sum, inv_diag[halo_idx], y[perm_idx]);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_pointwise_mult(local_int_t size, const double *__restrict__ x,
                               const double *__restrict__ y,
                               double *__restrict__ out) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  out[gid] = x[gid] * y[gid];
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_forward_sweep_0(local_int_t m, local_int_t block_nrow,
                                local_int_t offset, local_int_t ell_width,
                                const local_int_t *__restrict__ ell_col_ind,
                                const double *__restrict__ ell_val,
                                const local_int_t *__restrict__ diag_idx,
                                const double *__restrict__ x,
                                double *__restrict__ y) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= block_nrow) {
    return;
  }

  local_int_t row = gid + offset;

  double sum = x[row];
  // __builtin_nontemporal_load(x + row);
  local_int_t diag = diag_idx[row];
  // __builtin_nontemporal_load(diag_idx + row);
  double diag_val = ell_val[diag*m + row];
  // __builtin_nontemporal_load(ell_val + diag * m + row);

  for (local_int_t p = 0; p < diag; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];
    // __builtin_nontemporal_load(ell_col_ind + idx);

    // Every entry above offset is zero
    if (col >= 0 && col < offset) {
      sum =
          fma(-ell_val[idx], __ldg(y + col), sum);
    }
  }

  y[row] = sum * __drcp_rn(diag_val);
  // __builtin_nontemporal_store(sum * __drcp_rn(diag_val), y + row);
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_backward_sweep_0(local_int_t m, local_int_t block_nrow,
                                 local_int_t offset, local_int_t ell_width,
                                 const local_int_t *__restrict__ ell_col_ind,
                                 const double *__restrict__ ell_val,
                                 const local_int_t *__restrict__ diag_idx,
                                 double *__restrict__ x) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= block_nrow) {
    return;
  }

  local_int_t row = gid + offset;
  local_int_t diag = diag_idx[row];
  double sum = x[row];
  double diag_val = ell_val[diag * m + row];

  // Scale result with diagonal entry
  sum *= diag_val;

  for (local_int_t p = diag + 1; p < ell_width; ++p) {
    local_int_t idx = p * m + row;
    local_int_t col = ell_col_ind[idx];
    // __builtin_nontemporal_load(ell_col_ind + idx);

    // Every entry below offset should not be taken into account
    if (col >= offset && col < m) {
      sum =
          fma(-ell_val[idx], __ldg(x + col), sum);
    }
  }

  x[row] = sum * __drcp_rn(diag_val);
  // __builtin_nontemporal_store(sum * __drcp_rn(diag_val), x + row);
}

int ComputeSYMGS(const SparseMatrix &A, const Vector &r, Vector &x) {
  assert(x.localLength == A.localNumberOfColumns);

  local_int_t i = 0;

#ifndef HPCG_NO_MPI
  if (A.geom->size > 1) {
    PrepareSendBuffer(A, x);

    ///                             stream_interior,

    kernel_symgs_interior<128><<<dim3((A.sizes[0] - 1) / 128 + 1), dim3(128)>>>(
        A.localNumberOfRows, A.sizes[0], A.ell_width, A.ell_col_ind, A.ell_val,
        A.inv_diag, r.d_values, x.d_values);

    ExchangeHaloAsync(A);
    ObtainRecvBuffer(A, x);

    kernel_symgs_halo<128><<<dim3((A.halo_rows - 1) / 128 + 1), dim3(128)>>>(
        A.halo_rows, A.localNumberOfColumns, A.sizes[0], A.ell_width,
        A.halo_row_ind, A.halo_col_ind, A.halo_val, A.inv_diag, A.perm,
        r.d_values, x.d_values);

    ++i;
  }
#endif

  // Solve L
  for (; i < A.nblocks; ++i) {
    kernel_symgs_sweep<128><<<dim3((A.sizes[i] - 1) / 128 + 1), dim3(128)>>>(
        A.localNumberOfRows, A.localNumberOfColumns, A.sizes[i], A.offsets[i],
        A.ell_width, A.ell_col_ind, A.ell_val, A.inv_diag, r.d_values,
        x.d_values);
  }

  // Solve U
  for (i = A.ublocks; i >= 0; --i) {
    kernel_symgs_sweep<128><<<dim3((A.sizes[i] - 1) / 128 + 1), dim3(128)>>>(
        A.localNumberOfRows, A.localNumberOfColumns, A.sizes[i], A.offsets[i],
        A.ell_width, A.ell_col_ind, A.ell_val, A.inv_diag, r.d_values,
        x.d_values);
  }

  return 0;
}

int ComputeSYMGSZeroGuess(const SparseMatrix &A, const Vector &r, Vector &x) {
  assert(x.localLength == A.localNumberOfColumns);

  // Solve L
  kernel_pointwise_mult<256><<<dim3((A.sizes[0] - 1) / 256 + 1), dim3(256)>>>(
      A.sizes[0], r.d_values, A.inv_diag, x.d_values);

  for (local_int_t i = 1; i < A.nblocks; ++i) {
    kernel_forward_sweep_0<128>
        <<<dim3((A.sizes[i] - 1) / 128 + 1), dim3(128)>>>(
            A.localNumberOfRows, A.sizes[i], A.offsets[i], A.ell_width,
            A.ell_col_ind, A.ell_val, A.diag_idx, r.d_values, x.d_values);
  }

  // Solve U
  for (local_int_t i = A.ublocks; i >= 0; --i) {
    kernel_backward_sweep_0<128>
        <<<dim3((A.sizes[i] - 1) / 128 + 1), dim3(128)>>>(
            A.localNumberOfRows, A.sizes[i], A.offsets[i], A.ell_width,
            A.ell_col_ind, A.ell_val, A.diag_idx, x.d_values);
  }

  return 0;
}
