#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include <numa.h>
#endif

#include <cassert>
#include <vector>

#include "Geometry.hpp"
#include "MGData.hpp"

#if __cplusplus < 201103L
// for C++03
#include <map>
typedef std::map<global_int_t, local_int_t> GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map<global_int_t, local_int_t>;
#endif

struct SparseMatrix_STRUCT {
  char *title;    //!< name of the sparse matrix
  Geometry *geom; //!< geometry associated with this matrix
  global_int_t
      totalNumberOfRows; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros
                                      //!< across all processes
  local_int_t localNumberOfRows;      //!< number of rows local to this process
  local_int_t localNumberOfColumns; //!< number of columns local to this process
  local_int_t
      localNumberOfNonzeros; //!< number of nonzeros local to this process
  local_int_t numberOfNonzerosPerRow; //!< maximum number of nonzeros per row
  char *nonzerosInRow; //!< The number of nonzeros in a row will always be 27 or
                       //!< fewer
  global_int_t **mtxIndG;            //!< matrix indices as global values
  local_int_t **mtxIndL;             //!< matrix indices as local values
  double **matrixValues;             //!< values of matrix entries
  double **matrixDiagonal;           //!< values of matrix diagonal entries
  GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
  std::vector<global_int_t> localToGlobalMap; //!< local-to-global mapping
  mutable bool isDotProductOptimized;
  mutable bool isSpmvOptimized;
  mutable bool isMgOptimized;
  mutable bool isWaxpbyOptimized;
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable struct SparseMatrix_STRUCT *Ac; // Coarse grid matrix
  mutable MGData
      *mgData; // Pointer to the coarse level data for this fine matrix
  void *optimizationData; // pointer that can be used to store
                          // implementation-specific data

#ifndef HPCG_NO_MPI
  local_int_t numberOfExternalValues; //!< number of entries that are external
                                      //!< to this process
  int numberOfSendNeighbors;   //!< number of neighboring processes that will be
                               //!< send local data
  local_int_t totalToBeSent;   //!< total number of entries to be sent
  local_int_t *elementsToSend; //!< elements to send to neighboring processes
  int *neighbors;              //!< neighboring processes
  local_int_t *receiveLength; //!< lenghts of messages received from neighboring
                              //!< processes
  local_int_t
      *sendLength;    //!< lenghts of messages sent to neighboring processes
  double *sendBuffer; //!< send buffer for non-blocking sends

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
  double *halo_val;
#endif

  local_int_t *halo_row_ind;

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
typedef struct SparseMatrix_STRUCT SparseMatrix;

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
#endif // SPARSEMATRIX_HPP
