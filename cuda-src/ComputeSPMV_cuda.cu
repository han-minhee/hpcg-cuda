#include "ComputeSPMV_cuda.cuh"
__global__ void kernelSPMV(int n, double *x, double *y, int localNumberOfRows,
                           double **matrixValues, int **mtxIndL,
                           int **nonzerosInRow) {}

int ComputeSPMV_cuda_cusparse(const SparseMatrix &A, Vector &x, Vector &y) {
  assert(x.localLength >= A.localNumberOfColumns);
  assert(y.localLength >= A.localNumberOfRows);

  double *xv = x.values;
  double *yv = y.values;

  double *xv_d;
  double *yv_d;

  size_t M = A.localNumberOfRows;
  size_t N = A.localNumberOfColumns;

  double *flatMatrixVal = new double[M * 27];
  int *flatColIndices = new int[M * 27];
  int *flatRowOffsets = new int[M + 1];

  clock_t start;

  start = clock();
  #pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < 27; j++) {
      flatMatrixVal[i * 27 + j] = A.matrixValues[i][j];
      flatColIndices[i * 27 + j] = A.mtxIndL[i][j];
    }
    flatRowOffsets[i] = 27 * i;
  }

  flatRowOffsets[M] = flatRowOffsets[M - 1] + 1;

  cudaMalloc((void **)&xv_d, N * sizeof(double));
  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMalloc((void **)&yv_d, M * sizeof(double));
  if (gpuCheckError() == -1) {
    return -1;
  }

  cudaMemcpy(xv_d, xv, N * sizeof(double), cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {
    return -1;
  }
  
  cudaDeviceSynchronize();
  // --- Device side dense matrix

  cusparseHandle_t handle;
  cusparseCreate(&handle);

  if (gpuCheckError() == -1) {
    return -1;
  }

  cusparseDnVecDescr_t vecX, vecY;
  cusparseCreateDnVec(&vecX, N, xv_d, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, M, yv_d, CUDA_R_64F);

  if (gpuCheckError() == -1) {
    return -1;
  }

  int *csrRowOffsets_d;
  int *csrColInd_d;
  double *csrValues_d;

  cudaMalloc((void **)&csrRowOffsets_d, (M + 1) * sizeof(int));
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMalloc((void **)&csrColInd_d, (M * 27) * sizeof(int));
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMalloc((void **)&csrValues_d, (M * 27) * sizeof(double));
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMemcpy(csrColInd_d, flatColIndices, (M * 27) * sizeof(int),
             cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMemcpy(csrRowOffsets_d, flatRowOffsets, (M + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMemcpy(csrValues_d, flatMatrixVal, (M * 27) * sizeof(double),
             cudaMemcpyHostToDevice);
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, M, N, M * 27, csrRowOffsets_d, csrColInd_d,
                    csrValues_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (gpuCheckError() == -1) {

    return -1;
  }
  const double alpha = 1.0;
  const double beta = 0.0;

  size_t bufferSize;
  double *buffer_d;

  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, &beta, vecY, CUDA_R_64F,
                          CUSPARSE_CSRMV_ALG1, &bufferSize);
  if (gpuCheckError() == -1) {

    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaMalloc((void **)&buffer_d, bufferSize);
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, buffer_d);

  cudaMemcpy(yv, yv_d, M * sizeof(double), cudaMemcpyDeviceToHost);
  if (gpuCheckError() == -1) {
    return -1;
  }
  // printf("line passed %d\n", __LINE__);

  cudaFree(xv_d);
  cudaFree(yv_d);
  cudaFree(buffer_d);
  cudaFree(csrRowOffsets_d);
  cudaFree(csrColInd_d);
  cudaFree(csrValues_d);
  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);
  free(flatColIndices);
  free(flatMatrixVal);
  free(flatRowOffsets);

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
int ComputeSPMV_cuda(const SparseMatrix &A, Vector &x, Vector &y) {

}
