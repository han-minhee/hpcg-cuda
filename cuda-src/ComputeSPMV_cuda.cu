#include "ComputeSPMV_cuda.cuh"

__global__ void kernel_spmv_ell(local_int_t m, int nblocks,
                                local_int_t rows_per_block,
                                local_int_t ell_width,
                                const local_int_t *__restrict__ ell_col_ind,
                                const double *__restrict__ ell_val,
                                const double *__restrict__ x,
                                double *__restrict__ y) {
  // Applies for chunks of hipBlockDim_x * nblocks
  local_int_t color_block_offset = gridDim.x * (blockIdx.x / nblocks);

  // Applies for chunks of hipBlockDim_x and restarts for each
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

int ComputeSPMV_ref_cuda(const SparseMatrix &A, Vector &x, Vector &y) {
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

  //*const : can only alter the object being pointed
  // const double *const: cannot alter pointer and the object

  const double *const xv = x.values;
  double *const yv = y.values;

  const local_int_t nrow = A.localNumberOfRows;
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    const double *const cur_vals = A.matrixValues[i];
    const local_int_t *const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++)
      sum += cur_vals[j] * xv[cur_inds[j]];
    yv[i] = sum;
  }
  return 0;
}
int ComputeSPMV_cuda(SparseMatrix &A, Vector &x, Vector &y) {
  // convert to ELLPACK?
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

  //*const : can only alter the object being pointed
  // const double *const: cannot alter pointer and the object

  const double *const xv = x.values;
  double *const yv = y.values;

  const local_int_t nrow = A.localNumberOfRows;
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    const double *const cur_vals = A.matrixValues[i];
    const local_int_t *const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j = 0; j < cur_nnz; j++)
      sum += cur_vals[j] * xv[cur_inds[j]];
    yv[i] = sum;
  }

  vectorMemcpyFromHostToDevice(y);

  return 0;
}
