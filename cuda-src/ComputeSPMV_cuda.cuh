
/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/

#include "../src/SparseMatrix.hpp"
#include "../src/Vector.hpp"
#include "InitCuda.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cassert>
#include <cstdlib>
#include <cublas_v2.h>
#include <cusparse_v2.h>

struct CRSArrays {
  int m;              //< the dim of the matrix
  int nnz;            //< the number of nnz (== ia[m])
  double *cu_csrValA; //< the values (of size NNZ)
  int *cu_csrRowPtrA; //< the usual rowptr (of size m+1)
  int *cu_csrColIndA; //< the colidx of each NNZ (of size nnz)

  cudaStream_t streamId;
  cusparseHandle_t cusparseHandle;

  cusparseMatDescr_t descrA;

  CRSArrays() {
    cu_csrValA = NULL;
    cu_csrRowPtrA = NULL;
    cu_csrColIndA = NULL;

    // Create sparse handle (needed to call sparse functions
    streamId = 0;
    cusparseCreate(&cusparseHandle);
    cusparseSetStream(cusparseHandle, streamId);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  }

  ~CRSArrays() {
    cudaFree(cu_csrValA);
    cudaFree(cu_csrRowPtrA);
    cudaFree(cu_csrColIndA);

    // Destroy sparse handle
    cusparseDestroy(cusparseHandle);
  }
};

void ConvertToCRS(const SparseMatrix &A, CRSArrays &csr);
int ComputeSPMV_cuda(const SparseMatrix &A, Vector &x, Vector &y);
int ComputeSPMV_ref_cuda(const SparseMatrix &A, Vector &x, Vector &y);