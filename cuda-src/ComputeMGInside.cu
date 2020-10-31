
#include "../src/ComputeProlongation.hpp"
#include "../src/ComputeRestriction.hpp"
#include "../src/ComputeSPMV.hpp"
#include "../src/ComputeSYMGS.hpp"
#include "ComputeMGInside.cuh"
#include "Utils.cuh"

int ComputeMGInside(const SparseMatrix &A, const Vector &r, Vector &x) {

  printf("Entering MGInside\n");
  double *rv = new double[50 /*length*/];
  double *xv = new double[50 /*length*/];
  
  // printf("===entering MG Inside ===\n");
  assert(x.localLength == A.localNumberOfColumns);

  cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);

  for (int i = 0; i<50 /*length*/; i++){
    printf("before begin: x[%d]: %x, r[%d]: %x\n", i, xv[i], i, rv[i]);
  }

  if (A.mgData != 0) {
    CUDA_RETURN_IFF_ERROR(ComputeSYMGSZeroGuess(A, r, x));

    cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  
    for (int i = 0; i<50 /*length*/; i++){
      printf("after SYMGSZeroGuess: x[%d]: %x, r[%d]: %x\n", i, xv[i], i, rv[i]);
    }

    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;

    for (int i = 1; i < numberOfPresmootherSteps; ++i) {
      CUDA_RETURN_IFF_ERROR(ComputeSYMGS(A, r, x));
      cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
      cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    
      for (int j = 0; j<50 /*length*/; j++){
        printf("after SYMGS: x[%d]: %x, r[%d]: %x\n", j, xv[j], j, rv[j]);
      }
    }

#ifndef HPCG_REFERENCE
    CUDA_RETURN_IFF_ERROR(ComputeFusedSpMVRestriction(A, r, x));

#else
    CUDA_RETURN_IFF_ERROR(ComputeSPMV(A, x, *A.mgData->Axf));
    cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  
    for (int j = 0; j<50 /*length*/; j++){
      printf("after SPMV: x[%d]: %x, r[%d]: %x\n", j, xv[j], j, rv[j]);
    }
    CUDA_RETURN_IFF_ERROR(ComputeRestriction(A, r));
    // cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    // cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  
    // for (int j = 0; j<50 /*length*/; j++){
    //   printf("after restriction: x[%d]: %x, r[%d]: %x\n", j, xv[j], j, rv[j]);
    // }
#endif


    CUDA_RETURN_IFF_ERROR(ComputeMGInside(*A.Ac, *A.mgData->rc, *A.mgData->xc));

    CUDA_RETURN_IFF_ERROR(ComputeProlongation(A, x));
    cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  
    for (int j = 0; j<50 /*length*/; j++){
      printf("after prolongation: x[%d]: %x, r[%d]: %x\n", j, xv[j], j, rv[j]);
    }

    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;

    for (int i = 0; i < numberOfPostsmootherSteps; ++i) {
      CUDA_RETURN_IFF_ERROR(ComputeSYMGS(A, r, x));
      cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
      cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    
      for (int j = 0; j<50 /*length*/; j++){
        printf("after symgs: x[%d]: %x, r[%d]: %x\n", j, xv[j], j, rv[j]);
      }
    }
  } else {
    CUDA_RETURN_IFF_ERROR(ComputeSYMGSZeroGuess(A, r, x));
    cudaMemcpy(rv, r.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
    cudaMemcpy(xv, x.d_values, sizeof(double) * 50 /*length*/, cudaMemcpyDeviceToHost);
  
    for (int j = 0; j<50 /*length*/; j++){
      printf("after SYMGSZeroGuess: x[%d]: %x, r[%d]: %x\n", j, xv[j], j, rv[j]);
    }
  }

  delete [] xv; 
  delete [] rv;

  return 0;
}
