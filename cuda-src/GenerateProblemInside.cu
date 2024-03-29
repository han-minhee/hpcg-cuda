#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <cuda_runtime.h>

#include "GenerateProblemInside.cuh"
#include "Utils.cuh"

// sizeof(bool) * blockSizeY + sizeof(int) * blockSizeX * blockSizeY, 0,
#define LAUNCH_GENERATE_PROBLEM(blockSizeX, blockSizeY)                        \
  kernelGenerateProblem<blockSizeX, blockSizeY>                                \
      <<<dim3((localNumberOfRows - 1) / blockSizeY + 1),                       \
         dim3(blockSizeX, blockSizeY),                                         \
         sizeof(bool) * blockSizeY + sizeof(int) * blockSizeX * blockSizeY,    \
         0>>>(localNumberOfRows, nx, ny, nz, nx * ny, gnx, gny, gnz,           \
              gnx * gny, gix0, giy0, giz0, numberOfNonzerosPerRow,             \
              A.d_nonzerosInRow, A.d_mtxIndG, A.d_matrixValues,                \
              A.d_matrixDiagonal, A.d_localToGlobalMap, A.d_rowHash,           \
              (b != NULL) ? b->d_values : NULL)

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_set_one(local_int_t size, double *array) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  array[gid] = 1.0;
}

__device__ local_int_t get_hash(local_int_t ix, local_int_t iy,
                                local_int_t iz) {
  return ((ix & 1) << 2) | ((iy & 1) << 1) | ((iz & 1) << 0);
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__ void kernelGenerateProblem(
    local_int_t m, local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t nx_ny, global_int_t gnx, global_int_t gny, global_int_t gnz,
    global_int_t gnx_gny, global_int_t gix0, global_int_t giy0,
    global_int_t giz0, local_int_t numberOfNonzerosPerRow, char *nonzerosInRow,
    global_int_t *mtxIndG, double *matrixValues, local_int_t *matrixDiagonal,
    global_int_t *localToGlobalMap, local_int_t *rowHash, double *b) {
  // Current local row
  local_int_t currentLocalRow = blockIdx.x * BLOCKSIZEY + threadIdx.y;

  extern __shared__ char sdata[];

  // Offsets into shared arrays that hold
  // interior vertex marker, to determine if the current vertex is an interior
  // or boundary vertex
  bool *interior_vertex = reinterpret_cast<bool *>(sdata);
  // and column offset, that stores the column index array offset of the
  // current thread index in x direction
  int *column_offset =
      reinterpret_cast<int *>(sdata + sizeof(bool) * BLOCKSIZEY);

  // Offset into current local row
  column_offset += threadIdx.y * BLOCKSIZEX;

  // Initialize interior vertex marker
  if (threadIdx.x == 0) {
    interior_vertex[threadIdx.y] = true;
  }

  // Sync interior vertex initialization
  __syncthreads();

  // Do not exceed local number of rows
  if (currentLocalRow >= m) {
    return;
  }

  // Compute local vertex coordinates
  local_int_t iz = currentLocalRow / nx_ny;
  local_int_t iy = currentLocalRow / nx - ny * iz;
  local_int_t ix = currentLocalRow - iz * nx_ny - iy * nx;

  // Compute global vertex coordinates
  global_int_t giz = giz0 + iz;
  global_int_t giy = giy0 + iy;
  global_int_t gix = gix0 + ix;

  // Current global row
  global_int_t currentGlobalRow = giz * gnx_gny + giy * gnx + gix;

  // Obtain neighboring offsets in x, y and z direction relative to the
  // current vertex and compute the resulting neighboring coordinates
  global_int_t nb_giz = giz + threadIdx.x / 9 - 1;
  global_int_t nb_giy = giy + (threadIdx.x % 9) / 3 - 1;
  global_int_t nb_gix = gix + (threadIdx.x % 3) - 1;

  // Compute current global column for neighboring vertex
  global_int_t curcol = nb_giz * gnx_gny + nb_giy * gnx + nb_gix;

  // Check if current vertex is an interior or boundary vertex
  bool interior = (nb_giz > -1 && nb_giz < gnz && nb_giy > -1 && nb_giy < gny &&
                   nb_gix > -1 && nb_gix < gnx);

  // Number of non-zero entries in the current row
  char numberOfNonzerosInRow;

  // Each thread within the row checks if a neighbor exists for his
  // neighboring offset
  if (interior == false) {
    // If no neighbor exists for one of the offsets, we need to re-compute
    // the indexing for the column entry accesses
    interior_vertex[threadIdx.y] = false;
  }

  // Re-compute index into matrix, by marking if current offset is
  // a neighbor or not
  column_offset[threadIdx.x] = interior ? 1 : 0;

  // Wait for threads to finish
  __syncthreads();

  // Do we have an interior vertex?
  bool full_interior = interior_vertex[threadIdx.y];

  // Compute inclusive sum to obtain new matrix index offsets
  int tmp;
  if (threadIdx.x >= 1 && full_interior == false)
    tmp = column_offset[threadIdx.x - 1];
  __syncthreads();
  if (threadIdx.x >= 1 && full_interior == false)
    column_offset[threadIdx.x] += tmp;
  __syncthreads();
  if (threadIdx.x >= 2 && full_interior == false)
    tmp = column_offset[threadIdx.x - 2];
  __syncthreads();
  if (threadIdx.x >= 2 && full_interior == false)
    column_offset[threadIdx.x] += tmp;
  __syncthreads();
  if (threadIdx.x >= 4 && full_interior == false)
    tmp = column_offset[threadIdx.x - 4];
  __syncthreads();
  if (threadIdx.x >= 4 && full_interior == false)
    column_offset[threadIdx.x] += tmp;
  __syncthreads();
  if (threadIdx.x >= 8 && full_interior == false)
    tmp = column_offset[threadIdx.x - 8];
  __syncthreads();
  if (threadIdx.x >= 8 && full_interior == false)
    column_offset[threadIdx.x] += tmp;
  __syncthreads();
  if (threadIdx.x >= 16 && full_interior == false)
    tmp = column_offset[threadIdx.x - 16];
  __syncthreads();
  if (threadIdx.x >= 16 && full_interior == false)
    column_offset[threadIdx.x] += tmp;
  __syncthreads();

  // Do we have interior or boundary vertex, e.g. do we have a neighbor for each
  // direction?
  if (full_interior == true) {
    // Interior vertex

    // Index into matrix
    local_int_t idx = currentLocalRow * numberOfNonzerosPerRow + threadIdx.x;

    // Diagonal entry is threated differently
    if (curcol == currentGlobalRow) {
      // Store diagonal entry index
      matrixDiagonal[currentLocalRow] = threadIdx.x;
      // __builtin_nontemporal_store(threadIdx.x,
      //                             matrixDiagonal + currentLocalRow);

      // Diagonal matrix values are 26
      matrixValues[idx] = 26.0;
      // __builtin_nontemporal_store(26.0, matrixValues + idx);
    } else {
      // Off-diagonal matrix values are -1
      matrixValues[idx] = -1.0;
      // __builtin_nontemporal_store(-1.0, matrixValues + idx);
    }

    // Store current global column
    mtxIndG[idx] = curcol;
    // __builtin_nontemporal_store(curcol, mtxIndG + idx);

    // Interior vertices have 27 neighboring vertices
    numberOfNonzerosInRow = numberOfNonzerosPerRow;
  } else {
    // Boundary vertex, e.g. at least one neighboring offset is not a neighbor
    // (this happens e.g. on the global domains boundary) We do only process
    // "real" neighbors
    if (interior == true) {
      // Obtain current threads index into matrix from above inclusive scan
      // (convert from 1-based to 0-based indexing)
      int offset = column_offset[threadIdx.x] - 1;

      // Index into matrix
      local_int_t idx = currentLocalRow * numberOfNonzerosPerRow + offset;

      // Diagonal entry is threated differently
      if (curcol == currentGlobalRow) {
        // Store diagonal entry index
        matrixDiagonal[currentLocalRow] = offset;
        // __builtin_nontemporal_store(offset, matrixDiagonal +
        // currentLocalRow);

        // Diagonal matrix values are 26
        matrixValues[idx] = 26.0;
        // __builtin_nontemporal_store(26.0, matrixValues + idx);
      } else {
        // Off-diagonal matrix values are -1
        matrixValues[idx] = -1.0;
        // __builtin_nontemporal_store(-1.0, matrixValues + idx);
      }

      // Store current global column
      mtxIndG[idx] = curcol;
      // __builtin_nontemporal_store(curcol, mtxIndG + idx);
    }

    if (threadIdx.x == 0) {
      numberOfNonzerosInRow = column_offset[BLOCKSIZEX - 1];
    }
  }

  // For each row, initialize vector arrays and number of vertices
  if (threadIdx.x == 0) {
    nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
    localToGlobalMap[currentLocalRow] = currentGlobalRow;

    local_int_t crd = iz * nx * ny + iy * (nx << 1) + (ix << 2);
    local_int_t hash = get_hash(ix, iy, iz) * nx * ny * nz + crd;
    rowHash[currentLocalRow] = hash;

    if (b != NULL) {
      b[currentLocalRow] = (26.0 - (numberOfNonzerosInRow - 1.0));
    }
  }
}

template <unsigned int BLOCKSIZE>
__device__ void kernelDeviceReduceSum(local_int_t tid, local_int_t *data) {
  __syncthreads();

  if (BLOCKSIZE > 512) {
    if (tid < 512 && tid + 512 < BLOCKSIZE) {
      data[tid] += data[tid + 512];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 256) {
    if (tid < 256 && tid + 256 < BLOCKSIZE) {
      data[tid] += data[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 128) {
    if (tid < 128 && tid + 128 < BLOCKSIZE) {
      data[tid] += data[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 64) {
    if (tid < 64 && tid + 64 < BLOCKSIZE) {
      data[tid] += data[tid + 64];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 32) {
    if (tid < 32 && tid + 32 < BLOCKSIZE) {
      data[tid] += data[tid + 32];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 16) {
    if (tid < 16 && tid + 16 < BLOCKSIZE) {
      data[tid] += data[tid + 16];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 8) {
    if (tid < 8 && tid + 8 < BLOCKSIZE) {
      data[tid] += data[tid + 8];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 4) {
    if (tid < 4 && tid + 4 < BLOCKSIZE) {
      data[tid] += data[tid + 4];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 2) {
    if (tid < 2 && tid + 2 < BLOCKSIZE) {
      data[tid] += data[tid + 2];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 1) {
    if (tid < 1 && tid + 1 < BLOCKSIZE) {
      data[tid] += data[tid + 1];
    }
    __syncthreads();
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernelComputeLocalNNZPart1(local_int_t size, const char *nonzerosInRow,
                                    local_int_t *workspace) {
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
  local_int_t inc = gridDim.x * BLOCKSIZE;

  __shared__ local_int_t sdata[BLOCKSIZE];
  sdata[tid] = 0;

  for (local_int_t idx = gid; idx < size; idx += inc) {
    sdata[tid] += nonzerosInRow[idx];
  }

  kernelDeviceReduceSum<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_local_nnz_part2(local_int_t size, local_int_t *workspace) {
  local_int_t tid = threadIdx.x;

  __shared__ local_int_t sdata[BLOCKSIZE];
  sdata[tid] = 0;

  for (local_int_t idx = tid; idx < size; idx += BLOCKSIZE) {
    sdata[tid] += workspace[idx];
  }

  kernelDeviceReduceSum<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[0] = sdata[0];
  }
}

void GenerateProblemInside(SparseMatrix &A, Vector *b, Vector *x,
                           Vector *xexact) {
  // Local dimension in x, y and z direction
  local_int_t nx = A.geom->nx;
  local_int_t ny = A.geom->ny;
  local_int_t nz = A.geom->nz;

  // Global dimension in x, y and z direction
  global_int_t gnx = A.geom->gnx;
  global_int_t gny = A.geom->gny;
  global_int_t gnz = A.geom->gnz;

  // Base global index for current rank in the processor grid
  global_int_t gix0 = A.geom->gix0;
  global_int_t giy0 = A.geom->giy0;
  global_int_t giz0 = A.geom->giz0;

  // Local number of rows
  local_int_t localNumberOfRows = nx * ny * nz;
  assert(localNumberOfRows > 0);

  // Maximum number of entries per row in 27pt stencil
  local_int_t numberOfNonzerosPerRow = 27;

  // Global number of rows
  global_int_t totalNumberOfRows = gnx * gny * gnz;
  assert(totalNumberOfRows > 0);

  // Allocate vectors
  if (b != NULL)
    CudaInitializeVector(*b, localNumberOfRows);
  if (x != NULL)
    CudaInitializeVector(*x, localNumberOfRows);
  if (xexact != NULL)
    CudaInitializeVector(*xexact, localNumberOfRows);

  // Allocate structures
  CUDA_RETURN_VOID_IF_ERROR(cudaMalloc(
      (void **)&A.d_mtxIndG, std::max(sizeof(double), sizeof(global_int_t)) *
                                 localNumberOfRows * numberOfNonzerosPerRow));
  CUDA_RETURN_VOID_IF_ERROR(
      cudaMalloc((void **)&A.d_matrixValues,
                 sizeof(double) * localNumberOfRows * numberOfNonzerosPerRow));
  CUDA_RETURN_VOID_IF_ERROR(cudaMalloc((void **)&A.d_mtxIndL,
                                       sizeof(local_int_t) * localNumberOfRows *
                                           numberOfNonzerosPerRow));
  CUDA_RETURN_VOID_IF_ERROR(cudaMalloc((void **)&A.d_nonzerosInRow,
                                       sizeof(char) * localNumberOfRows));
  CUDA_RETURN_VOID_IF_ERROR(cudaMalloc(
      (void **)&A.d_matrixDiagonal, sizeof(local_int_t) * localNumberOfRows));
  CUDA_RETURN_VOID_IF_ERROR(cudaMalloc(
      (void **)&A.d_rowHash, sizeof(local_int_t) * localNumberOfRows));
  CUDA_RETURN_VOID_IF_ERROR(
      cudaMalloc((void **)&A.d_localToGlobalMap,
                 sizeof(global_int_t) * localNumberOfRows));

  // Determine blocksize
  unsigned int blocksize = 16;
  LAUNCH_GENERATE_PROBLEM(27, 16);

  // Initialize x vector, if not NULL
  if (x != NULL) {
    CUDA_RETURN_VOID_IF_ERROR(
        cudaMemset(x->d_values, 0, sizeof(double) * localNumberOfRows));
  }

  // Initialize exact solution, if not NULL
  if (xexact != NULL) {

    kernel_set_one<1024>
        <<<dim3((localNumberOfRows - 1) / 1024 + 1), dim3(1024)>>>(
            localNumberOfRows, xexact->d_values);
  }

  // printf("entering kernel\n");

  local_int_t *tmp = reinterpret_cast<local_int_t *>(workspace);

  // Compute number of local non-zero entries using two step reduction
  kernelComputeLocalNNZPart1<256>
      <<<dim3(256), dim3(256)>>>(localNumberOfRows, A.d_nonzerosInRow, tmp);

  kernel_local_nnz_part2<256><<<dim3(1), dim3(256)>>>(256, tmp);

  // Copy number of local non-zero entries to host
  local_int_t localNumberOfNonzeros;
  CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(&localNumberOfNonzeros, tmp,
                                       sizeof(local_int_t),
                                       cudaMemcpyDeviceToHost));

  global_int_t totalNumberOfNonzeros = 0;

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
  MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
#else
  long long lnnz = localNumberOfNonzeros, gnnz = 0;
  MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
  totalNumberOfNonzeros = gnnz;
#endif
#else
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif

  assert(totalNumberOfNonzeros > 0);

  // Initialize matrix parameters
  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.ell_width = numberOfNonzerosPerRow;
  A.numberOfNonzerosPerRow = numberOfNonzerosPerRow;
}

void CopyProblemToHostInside(SparseMatrix &A, Vector *b, Vector *x,
                             Vector *xexact) {
  // Allocate host structures
  A.nonzerosInRow = new char[A.localNumberOfRows];
  A.mtxIndG = new global_int_t *[A.localNumberOfRows];
  A.mtxIndL = new local_int_t *[A.localNumberOfRows];
  A.matrixValues = new double *[A.localNumberOfRows];
  A.matrixDiagonal = new double *[A.localNumberOfRows];
  local_int_t *mtxDiag = new local_int_t[A.localNumberOfRows];
  A.localToGlobalMap.resize(A.localNumberOfRows);

  // Now allocate the arrays pointed to
  A.mtxIndL[0] =
      new local_int_t[A.localNumberOfRows * A.numberOfNonzerosPerRow];
  A.matrixValues[0] =
      new double[A.localNumberOfRows * A.numberOfNonzerosPerRow];
  A.mtxIndG[0] =
      new global_int_t[A.localNumberOfRows * A.numberOfNonzerosPerRow];

  // Copy GPU data to host
  CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(A.nonzerosInRow, A.d_nonzerosInRow,
                                       sizeof(char) * A.localNumberOfRows,
                                       cudaMemcpyDeviceToHost));
  CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(
      A.mtxIndG[0], A.d_mtxIndG,
      sizeof(global_int_t) * A.localNumberOfRows * A.numberOfNonzerosPerRow,
      cudaMemcpyDeviceToHost));
  CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(A.matrixValues[0], A.d_matrixValues,
                                       sizeof(double) * A.localNumberOfRows *
                                           A.numberOfNonzerosPerRow,
                                       cudaMemcpyDeviceToHost));
  CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(
      mtxDiag, A.d_matrixDiagonal, sizeof(local_int_t) * A.localNumberOfRows,
      cudaMemcpyDeviceToHost));
  CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(
      A.localToGlobalMap.data(), A.d_localToGlobalMap,
      sizeof(global_int_t) * A.localNumberOfRows, cudaMemcpyDeviceToHost));

  // DONE : 08/26/2020 03:54 they are identical

  CUDA_RETURN_VOID_IF_ERROR(cudaFree(A.d_nonzerosInRow));
  CUDA_RETURN_VOID_IF_ERROR(cudaFree(A.d_matrixDiagonal));

  // Initialize pointers
  A.matrixDiagonal[0] = A.matrixValues[0] + mtxDiag[0];
  for (local_int_t i = 1; i < A.localNumberOfRows; ++i) {
    A.mtxIndL[i] = A.mtxIndL[0] + i * A.numberOfNonzerosPerRow;
    A.matrixValues[i] = A.matrixValues[0] + i * A.numberOfNonzerosPerRow;
    A.mtxIndG[i] = A.mtxIndG[0] + i * A.numberOfNonzerosPerRow;
    A.matrixDiagonal[i] = A.matrixValues[i] + mtxDiag[i];
  }

  delete[] mtxDiag;

  // Create global to local map
  for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
    A.globalToLocalMap[A.localToGlobalMap[i]] = i;
  }

  // Allocate and copy vectors, if available
  if (b != NULL) {
    InitializeVector(*b, A.localNumberOfRows);
    CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(b->values, b->d_values,
                                         sizeof(double) * b->localLength,
                                         cudaMemcpyDeviceToHost));
  }

  if (x != NULL) {
    InitializeVector(*x, A.localNumberOfRows);
    CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(x->values, x->d_values,
                                         sizeof(double) * x->localLength,
                                         cudaMemcpyDeviceToHost));
  }

  if (xexact != NULL) {
    InitializeVector(*xexact, A.localNumberOfRows);
    CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(xexact->values, xexact->d_values,
                                         sizeof(double) * xexact->localLength,
                                         cudaMemcpyDeviceToHost));
  }

  // DONE: identical 08/26/2020 04:03
}
