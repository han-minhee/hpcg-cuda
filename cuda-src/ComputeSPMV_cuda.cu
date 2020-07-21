#include "ComputeSPMV_cuda.cuh"

__global__ void kernelSPMV(int n, double *xv, double *yv) {}

// ref:
// https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f

int ComputeSPMV_cuda(const SparseMatrix &A, Vector &x, Vector &y) {
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

  double *xv = x.values;
  double *yv = y.values;

  double *xv_d;
  double *yv_d;

  size_t nRow = A.localNumberOfRows;
  size_t sizeY = nRow * sizeof(double);

  cudaMalloc((void **)&yv_d, sizeY);

  if (gpuCheckError() == -1) {
    return -1;
  }

  for (int i = 0; i <nRow; i++){
    double sum = 0.0f;
    double *currentValues = A.matrixValues[i];
    int *currentIndices = A.mtxIndL[i];
    int currentNumberOfNoneZeros = A.nonzerosInRow[i];


  }

  return 0;
}

int ComputeSPMV_ref(const SparseMatrix &A, Vector &x, Vector &y) {

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
