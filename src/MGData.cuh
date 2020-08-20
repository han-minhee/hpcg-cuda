#ifndef MGDATA_HPP
#define MGDATA_HPP

#include "SparseMatrix.cuh"
#include "Vector.cuh"
#include <cassert>
#include <cuda_runtime.h>

struct MGData_STRUCT {
  int numberOfPresmootherSteps;  // Call ComputeSYMGS this many times prior to
                                 // coarsening
  int numberOfPostsmootherSteps; // Call ComputeSYMGS this many times after
                                 // coarsening
  local_int_t *f2cOperator; //!< 1D array containing the fine operator local IDs
                            //!< that will be injected into coarse space.
  Vector *rc;               // coarse grid residual vector
  Vector *xc;               // coarse grid solution vector
  Vector *Axf;              // fine grid residual vector
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void *optimizationData;

  local_int_t *d_f2cOperator; //!< f2cOperator on device
};
typedef struct MGData_STRUCT MGData;

/*!
 Constructor for the data structure of CG vectors.

 @param[in] Ac - Fully-formed coarse matrix
 @param[in] f2cOperator -
 @param[out] data the data structure for CG vectors that will be allocated to
 get it ready for use in CG iterations
 */
inline void InitializeMGData(local_int_t *f2cOperator, Vector *rc, Vector *xc,
                             Vector *Axf, MGData &data) {
  data.numberOfPresmootherSteps = 1;
  data.numberOfPostsmootherSteps = 1;
  data.f2cOperator = f2cOperator; // Space for injection operator
  data.rc = rc;
  data.xc = xc;
  data.Axf = Axf;
  return;
}

/*!
 Destructor for the CG vectors data.

 @param[inout] data the MG data structure whose storage is deallocated
 */
inline void DeleteMGData(MGData &data) {

  delete[] data.f2cOperator;
  DeleteVector(*data.Axf);
  DeleteVector(*data.rc);
  DeleteVector(*data.xc);
  delete data.Axf;
  delete data.rc;
  delete data.xc;

#ifdef HPCG_REFERENCE
  CudaDeleteVector(*data.Axf);
#endif

  CudaDeleteVector(*data.rc);
  CudaDeleteVector(*data.xc);

  cudaFree(data.d_f2cOperator);
  return;
}

#endif // MGDATA_HPP
