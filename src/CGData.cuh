#ifndef CGDATA_HPP
#define CGDATA_HPP

#include "SparseMatrix.cuh"
#include "Vector.cuh"

struct CGData_STRUCT {
  Vector r; //!< pointer to residual vector
  Vector z; //!< pointer to preconditioned residual vector
  Vector p; //!< pointer to direction vector
  Vector Ap; //!< pointer to Krylov vector
};
typedef struct CGData_STRUCT CGData;

/*!
 Constructor for the data structure of CG vectors.

 @param[in]  A    the data structure that describes the problem matrix and its structure
 @param[out] data the data structure for CG vectors that will be allocated to get it ready for use in CG iterations
 */
inline void InitializeSparseCGData(SparseMatrix & A, CGData & data) {
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;
  InitializeVector(data.r, nrow);
  InitializeVector(data.z, ncol);
  InitializeVector(data.p, ncol);
  InitializeVector(data.Ap, nrow);
  return;
}

inline void CudaInitializeSparseCGData(SparseMatrix& A, CGData& data)
{
    CudaInitializeVector(data.r, A.localNumberOfRows);
    CudaInitializeVector(data.z, A.localNumberOfColumns);
    CudaInitializeVector(data.p, A.localNumberOfColumns);
    CudaInitializeVector(data.Ap, A.localNumberOfRows);
}

/*!
 Destructor for the CG vectors data.

 @param[inout] data the CG vectors data structure whose storage is deallocated
 */
inline void DeleteCGData(CGData & data) {

  DeleteVector (data.r);
  DeleteVector (data.z);
  DeleteVector (data.p);
  DeleteVector (data.Ap);
  return;
}

inline void CudaDeleteCGData(CGData& data)
{
    CudaDeleteVector (data.r);
    CudaDeleteVector (data.z);
    CudaDeleteVector (data.p);
    CudaDeleteVector (data.Ap);
}


#endif // CGDATA_HPP

