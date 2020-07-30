
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
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cusparse.h"
#include "time.h"
#include <cassert>
#include <cstdlib>

int ComputeSPMV_cuda(const SparseMatrix &A, Vector &x, Vector &y);
int ComputeSPMV_ref_cuda(const SparseMatrix &A, Vector &x, Vector &y);