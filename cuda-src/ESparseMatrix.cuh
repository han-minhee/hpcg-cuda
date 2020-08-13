#ifndef HPCG_NO_MPI
#include <mpi.h>
#include <numa.h>
#endif

#include <cassert>
#include <vector>

#include "../src/Geometry.hpp"
#include "../src/MGData.hpp"
#include "../src/SparseMatrix.hpp"

#include "EVector.cuh"
#include "Utils.cuh"

// E stands for Extended
struct ESparseMatrix_STRUCT : SparseMatrix {

#ifndef HPCG_NO_MPI
  // HIP related structures
  MPI_Request *recv_request;
  MPI_Request *send_request;

  local_int_t *d_elementsToSend;

  double *recv_buffer;
  double *send_buffer;
  double *d_send_buffer;

  // ELL matrix storage format arrays for halo part
  local_int_t halo_rows;
  local_int_t *halo_col_ind;
  local_int_t *halo_row_ind;
  double *halo_val;
#endif

  // HPCG matrix storage format arrays
  char *d_nonzerosInRow;
  global_int_t *d_mtxIndG;
  local_int_t *d_mtxIndL;
  double *d_matrixValues;
  local_int_t *d_matrixDiagonal;
  global_int_t *d_localToGlobalMap;
  local_int_t *d_rowHash;

  // ELL matrix storage format arrays
  local_int_t ell_width;    //!< Maximum nnz per row
  local_int_t *ell_col_ind; //!< ELL column indices
  double *ell_val;          //!< ELL values

  local_int_t *diag_idx; //!< Index to diagonal value in ell_val
  double *inv_diag;      //!< Inverse diagonal values

  // SymGS structures
  int nblocks;          //!< Number of independent sets
  int ublocks;          //!< Number of upper triangular sets
  local_int_t *sizes;   //!< Number of rows of independent sets
  local_int_t *offsets; //!< Pointer to the first row of each independent set
  local_int_t *perm;    //!< Permutation obtained by independent set
};

typedef struct ESparseMatrix_STRUCT ESparseMatrix;

inline void InitializeESparseMatrix(ESparseMatrix &A, Geometry *geom) {
  InitializeSparseMatrix(A, geom);

#ifndef HPCG_NO_MPI
  A.recv_request = NULL;
  A.send_request = NULL;
  A.d_elementsToSend = NULL;
  A.recv_buffer = NULL;
  A.send_buffer = NULL;
  A.d_send_buffer = NULL;

  A.halo_row_ind = NULL;
  A.halo_col_ind = NULL;
  A.halo_val = NULL;
#endif

  A.ell_width = 0;
  A.ell_col_ind = NULL;
  A.ell_val = NULL;
  A.diag_idx = NULL;
  A.inv_diag = NULL;

  A.nblocks = 0;
  A.ublocks = 0;
  A.sizes = NULL;
  A.offsets = NULL;
  A.perm = NULL;

  return;
}

void ConvertToELL(ESparseMatrix &A);
void ExtractDiagonal(ESparseMatrix &A);

void CUDACopyMatrixDiagonal(const ESparseMatrix &A, EVector &diagonal);

inline void ECopyMatrixDiagonal(ESparseMatrix &A, EVector &diagonal) {
  CopyMatrixDiagonal(A, diagonal);
  CUDACopyMatrixDiagonal(A, diagonal);
  return;
}

void CUDAReplaceMatrixDiagonal(ESparseMatrix &A, const EVector &diagonal) {
  ReplaceMatrixDiagonal(A, diagonal);
}