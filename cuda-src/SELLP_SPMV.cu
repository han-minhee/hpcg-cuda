__global__ void kernelSellpSpmv(int n, int b, int t, double alpha, int *rowptr, int *colind,
                double *values, double *x, double beta, double *y) {

  // t threads assigned to each row
  int idx = threadIdx.x;                         // thread in row
  int idy = threadIdx.y;                         // local row
  int ldx = idy * blocksize + idx;               // first element to be accessed
  int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
  int row = bdx * blocksize + idx;               // global row index
  extern shared double shared[];
  if (row < n) {
    double dot = 0.0;
    int offset = rowptr[bdx];
    int block = blocksize * t; // total number of threads
                               // number of elements each thread handles
    int max_ = (rowptr[bdx + 1] - offset) / block;
    // partial product loop unrolled to blocks of two
    int kk, i1, i2;
    double x1, x2, v1, v2;
    d_colind += offset + ldx;
    d_val += offset + ldx;
    for (kk = 0; kk < max_ - 1; kk += 2) {
      i1 = colind[block * kk];
      i2 = colind[block * kk + block];
      x1 = x[i1];
      x2 = x[i2];
      v1 = values[block * kk];
      v2 = values[block * kk + block];
      dot += v1 * x1;
      dot += v2 * x2;
    }
    // maybe one additional step
    if (kk < max_) {
      x1 = d_x[d_colind[block * kk]];
      v1 = d_val[block * kk];
      dot += v1 * x1;
    }
    // write result to shared memory
    shared[ldx] = dot;
    // reduction
    syncthreads();
    if (idy < 4) {
      shared[ldx] += shared[ldx + blocksize * 4];
      s y n c t h r e a d s();
      if (idy < 2)
        shared[ldx] += shared[ldx + blocksize * 2];
      s y n c t h r e a d s();
      if (idy == 0) {
        y[row] =
            (shared[ldx] + shared[ldx + blocksize * 1]) * alpha + beta * y[row];
      }
    }
  }
}