#include "ComputeSPMV_cuda.cuh"

__global__ void kernelSPMV(int n, double *xv, double *yv, int blockSize) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int x;
  int i;
  int offset = row * blockSize;
  double sum;
  for (x = offset; x < offset + blockSize; x++) {
    if (x < num_rows) {
      sum = 0;
      for (i = 0; i < AnonzerosInRow[x]; i++) {
        sum += AmatrixValues[x * 27 + i] * xMatrix[AmtxIndL[x * 27 + i]];
      }

      yMatrix[x] = sum;
    }
  }
}

void ConvertToCRS(const SparseMatrix &A, CRSArrays &csr) {
  csr.nnz = A.localNumberOfNonzeros;
  csr.m = A.localNumberOfRows;
  csr.cu_csrValA = *A.matrixValues;
  csr.cu_csrColIndA = (int *)A.nonzerosInRow;
  csr.cu_csrRowPtrA = *A.mtxIndL;
  return;
}

int ComputeSPMV_cuda(const SparseMatrix &A, Vector &x, Vector &y) {
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

  double *xv = x.values;
  double *yv = y.values;

  double *xv_d;
  double *yv_d;

  cudaMalloc((void **)&xv_d, x.localLength * sizeof(double));
  if (gpuCheckError() == -1) {
    return -1;
  }
  printf("malloc 1\n");

  cudaMalloc((void **)&yv_d, y.localLength * sizeof(double));
  if (gpuCheckError() == -1) {
    return -1;
  }
  printf("malloc 2\n");

  cudaMemcpy(xv_d, xv, x.localLength * sizeof(double), cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }
  printf("cpy 1\n");

  cudaMemcpy(yv_d, yv, y.localLength * sizeof(double), cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }
  printf("cpy 2\n");

  cudaDeviceSynchronize();

  return 0;
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
