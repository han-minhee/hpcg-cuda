#ifndef HPCG_NO_MPI
#include <mpi.h>
#include <numa.h>
#endif

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define LAUNCH_COPY_INDICES(blockSizeX, blockSizeY)                            \
  kernelCopyIndices<blockSizeX, blockSizeY>                                    \
      <<<dim3((A.localNumberOfRows - 1) / blockSizeY + 1),                     \
         dim3(blockSizeX, blockSizeY)>>>(                                      \
          A.localNumberOfRows, A.d_nonzerosInRow, A.d_mtxIndG, A.d_mtxIndL)

#define LAUNCH_SETUP_HALO(blockSizeX, blockSizeY)                              \
  kernelSetupHalo<blockSizeX, blockSizeY>                                      \
      <<<dim3((A.localNumberOfRows - 1) / blockSizeY + 1),                     \
         dim3(blockSizeX, blockSizeY)>>>(                                      \
          A.localNumberOfRows, max_boundary, max_sending, max_neighbors, nx,   \
          ny, nz, (nx & (nx - 1)), (ny & (ny - 1)), (nz & (nz - 1)),           \
          A.geom->npx, A.geom->npy, A.geom->npz, A.geom->gnx,                  \
          A.geom->gnx * A.geom->gny, A.geom->gix0 / nx, A.geom->giy0 / ny,     \
          A.geom->giz0 / nz, A.d_nonzerosInRow, A.d_mtxIndG, A.d_mtxIndL,      \
          d_nsend_per_rank, d_nrecv_per_rank, d_neighbors, d_send_indices,     \
          d_recv_indices, d_halo_indices)

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__
    void kernelCopyIndices(int size, const char *nonzerosInRow,
                           const int *mtxIndG, int *mtxIndL) {
  int row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

  if (row >= size) {
    return;
  }

  int idx = row * BLOCKSIZEX + threadIdx.x;

  if (threadIdx.x < nonzerosInRow[row]) {
    mtxIndL[idx] = mtxIndG[idx];
  } else {
    mtxIndL[idx] = -1;
  }
}

 struct AddOp {
     template <typename T>
     //  CUB_RUNTIME_FUNCTION __forceinline__
     __device__ __forceinline__
     T operator()(const T &a, const T &b) const {
       return a + b;
     }
   };

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__
    void kernelSetupHalo(int m, int max_boundary,
                         int max_sending, int max_neighbors,
                         int nx, int ny, int nz,
                         bool xp2, bool yp2, bool zp2, int npx,
                         int npy, int npz, int gnx,
                         int gnxgny, int ipx0,
                         int ipy0, int ipz0,
                         const char *nonzerosInRow, const int *mtxIndG,
                         int *mtxIndL, int *nsend_per_rank,
                         int *nrecv_per_rank, int *neighbors,
                         int *send_indices, int *recv_indices,
                         int *halo_indices) {

  // Each block processes blockDim.y rows
  int currentLocalRow = blockIdx.x * BLOCKSIZEY + threadIdx.y;

  // Some shared memory to mark rows that need to be sent to neighboring
  // processes
  __shared__ bool sdata[BLOCKSIZEX * BLOCKSIZEY];
  sdata[threadIdx.x + threadIdx.y * max_neighbors] = false;

  __syncthreads();

  // Do not exceed number of rows
  if (currentLocalRow >= m) {
    return;
  }

  // Global ID for 1D grid of 2D blocks
  int gid = currentLocalRow * BLOCKSIZEX + threadIdx.x;

  // Process only non-zeros of current row ; each thread index in x direction
  // processes one column entry
  if (threadIdx.x < nonzerosInRow[currentLocalRow]) {
    // Obtain the corresponding global column index (generated in
    // GenerateProblem.cpp)
    int currentGlobalColumn = mtxIndG[gid];

    // Determine neighboring process of current global column
    int iz = currentGlobalColumn / gnxgny;
    int iy = (currentGlobalColumn - iz * gnxgny) / gnx;
    int ix = currentGlobalColumn % gnx;

    int ipz = iz / nz;
    int ipy = iy / ny;
    int ipx = ix / nx;

    // Compute neighboring process id depending on the global column.
    // Each domain has at most 26 neighboring domains.
    // Since the numbering is following a fixed order, we can compute the
    // neighbor process id by the actual x,y,z coordinate of the entry, using
    // the domains offsets into the global numbering.
    int neighborRankId =
        (ipz - ipz0) * 9 + (ipy - ipy0) * 3 + (ipx - ipx0);

    // This will give us the neighboring process id between [-13, 13] where 0
    // is the local domain. We shift the resulting id by 13 to avoid negative
    // indices.
    neighborRankId += 13;

    // Check whether we are in the local domain or not
    if (neighborRankId != 13) {
      // Mark current row for sending, to avoid multiple entries with the same
      // row index
      sdata[neighborRankId + threadIdx.y * max_neighbors] = true;

      // Also store the "real" process id this global column index belongs to
      neighbors[neighborRankId] = ipx + ipy * npx + ipz * npy * npx;

      // Count up the global column that we have to receive by a neighbor using
      // atomics
      int idx = atomicAdd(&nrecv_per_rank[neighborRankId], 1);

      // Halo indices array stores the global id, so we can easily access the
      // matrix column array at the halo position
      halo_indices[neighborRankId * max_boundary + idx] = gid;

      // Store the global column id that we have to receive from a neighbor
      recv_indices[neighborRankId * max_boundary + idx] = currentGlobalColumn;
    } else {
      // Determine local column index
      //            int lz = iz % nz;
      //            int ly = currentGlobalColumn / gnx % ny;
      //            int lx = currentGlobalColumn % nx;
      int lz = (zp2) ? iz % nz : iz & (nz - 1);
      int ly = (yp2) ? currentGlobalColumn / gnx % ny
                             : currentGlobalColumn / gnx & (ny - 1);
      int lx =
          (xp2) ? currentGlobalColumn % nx : currentGlobalColumn & (nx - 1);

      // Store the local column index in the local matrix column array
      mtxIndL[gid] = lz * ny * nx + ly * nx + lx;
    }
  } else {
    // This is a zero entry
    mtxIndL[gid] = -1;
  }

  __syncthreads();

  // Check if current row has been marked for sending its entry
  if (sdata[threadIdx.x + threadIdx.y * BLOCKSIZEX] == true) {
    // If current row has been marked for sending, store its index
    int idx = atomicAdd(&nsend_per_rank[threadIdx.x], 1);
    send_indices[threadIdx.x * max_sending + idx] = currentLocalRow;
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_halo_columns(int size, int m,
                             int rank_offset,
                             const int *halo_indices,
                             const int *offsets,
                             int *mtxIndL) {
  // 1D thread indexing
  int gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  // Do not run out of bounds
  if (gid >= size) {
    return;
  }

  // Loop over all halo entries of the current row
  for (int i = offsets[gid]; i < offsets[gid + 1]; ++i) {
    // Get the index value to access the halo entry in the local matrix column
    // array
    int idx = halo_indices[i];

    // Numbering of halo entries are consecutive with number of local rows as
    // offset
    mtxIndL[idx] = m + gid + rank_offset;
  }
}

void SetupHaloInside(SparseMatrix &A) {
    // Determine blocksize for 2D kernel launch
    unsigned int blocksize = 512 / 27;

    // Compute next power of two
    blocksize |= blocksize >> 1;
    blocksize |= blocksize >> 2;
    blocksize |= blocksize >> 4;
    blocksize |= blocksize >> 8;
    blocksize |= blocksize >> 16;
    ++blocksize;

    // Shift right until we obtain a valid blocksize
    while(blocksize * 27 > 512)
    {
        blocksize >>= 1;
    }

#ifdef HPCG_NO_MPI
    if     (blocksize == 32) LAUNCH_COPY_INDICES(27, 32);
    else if(blocksize == 16) LAUNCH_COPY_INDICES(27, 16);
    else if(blocksize ==  8) LAUNCH_COPY_INDICES(27,  8);
    else                     LAUNCH_COPY_INDICES(27,  4);
#else
    if(A.geom->size == 1)
    {
        if     (blocksize == 32) LAUNCH_COPY_INDICES(27, 32);
        else if(blocksize == 16) LAUNCH_COPY_INDICES(27, 16);
        else if(blocksize ==  8) LAUNCH_COPY_INDICES(27,  8);
        else                     LAUNCH_COPY_INDICES(27,  4);

        return;
    }

  // Local dimensions in x, y and z direction
  int nx = A.geom->nx;
  int ny = A.geom->ny;
  int nz = A.geom->nz;

  // Number of partitions with varying nz values have to be 1 in the current
  // implementation
  assert(A.geom->npartz == 1);

  // Array of partition ids of processor in z direction where new value of nz
  // starts have to be equal to the number of processors in z direction the in
  // the current implementation
  assert(A.geom->partz_ids[0] == A.geom->npz);

  // Array of length npartz containing the nz values for each partition have to
  // be equal to the local dimension in z direction in the current
  // implementation
  assert(A.geom->partz_nz[0] == nz);

  // Determine two largest dimensions
  int max_dim_1 = std::max(nx, std::max(ny, nz));
  int max_dim_2 =
      ((nx >= ny && nx <= nz) || (nx >= nz && nx <= ny))
          ? nx
          : ((ny >= nz && ny <= nx) || (ny >= nx && ny <= nz)) ? ny : nz;

  // Maximum of entries that can be sent to a single neighboring rank
  int max_sending = max_dim_1 * max_dim_2;

  // 27 pt stencil has a maximum of 9 boundary entries per boundary plane
  // and thus, the maximum number of boundary elements can be computed to be
  // 9 * max_dim_1 * max_dim_2
  int max_boundary = 9 * max_dim_1 * max_dim_2;

  int max_neighbors = 27;

  // Arrays to hold send and receive element offsets per rank
  int *d_nsend_per_rank;
  int *d_nrecv_per_rank;

  // Number of elements is stored for each neighboring rank
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&d_nsend_per_rank,
                                sizeof(int) * max_neighbors));
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&d_nrecv_per_rank,
                                sizeof(int) * max_neighbors));

  // Since we use increments, we have to initialize with 0
  CUDA_CHECK_COMMAND(
      cudaMemset(d_nsend_per_rank, 0, sizeof(int) * max_neighbors));
  CUDA_CHECK_COMMAND(
      cudaMemset(d_nrecv_per_rank, 0, sizeof(int) * max_neighbors));

  // Array to store the neighboring process ids
  int *d_neighbors;
  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&d_neighbors, sizeof(int) * max_neighbors));

  // Array to hold send indices
  int *d_send_indices;

  // d_send_indices holds max_sending elements per neighboring rank, at max
  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&d_send_indices,
                 sizeof(int) * max_sending * max_neighbors));

  // Array to hold receive and halo indices
  int *d_recv_indices;
  int *d_halo_indices;

  // Both arrays hold max_boundary elements per neighboring rank, at max
  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&d_recv_indices,
                 sizeof(int) * max_boundary * max_neighbors));
  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&d_halo_indices,
                 sizeof(int) * max_boundary * max_neighbors));

  // SetupHalo kernel
  if (blocksize == 32)
    LAUNCH_SETUP_HALO(27, 32);
  else if (blocksize == 16)
    LAUNCH_SETUP_HALO(27, 16);
  else if (blocksize == 8)
    LAUNCH_SETUP_HALO(27, 8);
  else
    LAUNCH_SETUP_HALO(27, 4);

  // Prefix sum to obtain send index offsets
  std::vector<int> nsend_per_rank(max_neighbors + 1);
  CUDA_CHECK_COMMAND(cudaMemcpy(nsend_per_rank.data() + 1, d_nsend_per_rank,
                                sizeof(int) * max_neighbors,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaFree(d_nsend_per_rank));

  nsend_per_rank[0] = 0;
  for (int i = 0; i < max_neighbors; ++i) {
    nsend_per_rank[i + 1] += nsend_per_rank[i];
  }

  // Total elements to be sent
  A.totalToBeSent = nsend_per_rank[max_neighbors];

  // Array to hold number of entries that have to be sent to each process
  A.sendLength = new int[A.geom->size - 1];

  // Allocate receive and send buffers on GPU and CPU
  size_t buffer_size = ((A.totalToBeSent - 1) / (1 << 21) + 1) * (1 << 21);
  A.recv_buffer = (double *)numa_alloc_local(sizeof(double) * buffer_size);
  A.send_buffer = (double *)numa_alloc_local(sizeof(double) * buffer_size);

  NULL_CHECK(A.recv_buffer);
  NULL_CHECK(A.send_buffer);

  CUDA_CHECK_COMMAND(cudaHostRegister(A.recv_buffer,
                                      sizeof(double) * A.totalToBeSent,
                                      cudaHostRegisterDefault));
  CUDA_CHECK_COMMAND(cudaHostRegister(A.send_buffer,
                                      sizeof(double) * A.totalToBeSent,
                                      cudaHostRegisterDefault));

  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&A.d_send_buffer, sizeof(double) * A.totalToBeSent));

  // Sort send indices to obtain elementsToSend array
  // elementsToSend array has to be in increasing order, so other processes know
  // where to place the elements.
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&A.d_elementsToSend,
                                sizeof(int) * A.totalToBeSent));

  // TODO segmented sort might be faster
  A.numberOfSendNeighbors = 0;

  // Loop over all possible neighboring processes
  for (int i = 0; i < max_neighbors; ++i) {
    // Compute number of entries to be sent to i-th rank
    int entriesToSend = nsend_per_rank[i + 1] - nsend_per_rank[i];

    // Check if this is actually a neighbor that receives some data
    if (entriesToSend == 0) {
      // Nothing to be sent / sorted, skip
      continue;
    }

    size_t rocprim_size;
    void *rocprim_buffer = NULL;

    // Obtain buffer size
    // TOdo :: radix_sort_keys
    // radix_sort_keys
    CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortKeys(
        rocprim_buffer, rocprim_size, d_send_indices + i * max_sending,
        A.d_elementsToSend + nsend_per_rank[i], entriesToSend));
    CUDA_CHECK_COMMAND(cudaMalloc(&rocprim_buffer, rocprim_size));

    // Sort send indices to obtain increasing order
    CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortKeys(
        rocprim_buffer, rocprim_size, d_send_indices + i * max_sending,
        A.d_elementsToSend + nsend_per_rank[i], entriesToSend));
    CUDA_CHECK_COMMAND(cudaFree(rocprim_buffer));
    rocprim_buffer = NULL;

    // Store number of elements that have to be sent to i-th process
    A.sendLength[A.numberOfSendNeighbors++] = entriesToSend;
  }

  // Free up memory
  CUDA_CHECK_COMMAND(cudaFree(d_send_indices));

  // Prefix sum to obtain receive indices offsets (with duplicates)
  std::vector<int> nrecv_per_rank(max_neighbors + 1);
  CUDA_CHECK_COMMAND(cudaMemcpy(nrecv_per_rank.data() + 1, d_nrecv_per_rank,
                                sizeof(int) * max_neighbors,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaFree(d_nrecv_per_rank));

  nrecv_per_rank[0] = 0;
  for (int i = 0; i < max_neighbors; ++i) {
    // Verify boundary size does not exceed maximum boundary elements
    assert(nrecv_per_rank[i + 1] < max_boundary);

    nrecv_per_rank[i + 1] += nrecv_per_rank[i];
  }

  // Initialize number of external values
  A.numberOfExternalValues = 0;

  // Array to hold number of elements that have to be received from each
  // neighboring process
  A.receiveLength = new int[A.geom->size - 1];

  // Counter for number of neighbors we are actually receiving data from
  int neighborCount = 0;

  // Create rank indexing array for send, recv and halo lists
  std::vector<int *> d_recvList(max_neighbors);
  std::vector<int *> d_haloList(max_neighbors);

  for (int i = 0; i < max_neighbors; ++i) {
    d_recvList[i] = d_recv_indices + i * max_boundary;
    d_haloList[i] = d_halo_indices + i * max_boundary;
  }

  // Own rank can be buffer, nothing should be sent/received by ourselves
  int *d_recvBuffer = d_recvList[13];
  int *d_haloBuffer = d_haloList[13];

  // Array to hold the process ids of all neighbors that we receive data from
  A.neighbors = new int[A.geom->size - 1];

  // Buffer to process the GPU data
  std::vector<int> neighbors(max_neighbors);
  CUDA_CHECK_COMMAND(cudaMemcpy(neighbors.data(), d_neighbors,
                                sizeof(int) * max_neighbors,
                                cudaMemcpyDeviceToHost));
  CUDA_CHECK_COMMAND(cudaFree(d_neighbors));

  // Loop over all possible neighbors
  for (int i = 0; i < max_neighbors; ++i) {
    // Number of entries that have to be received from i-th rank
    int entriesToRecv = nrecv_per_rank[i + 1] - nrecv_per_rank[i];

    // Check if we actually receive data
    if (entriesToRecv == 0) {
      // Nothing to receive, skip
      continue;
    }

    size_t rocprim_size;
    void *rocprim_buffer = NULL;

    // Obtain buffer size
    CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortPairs(
        rocprim_buffer, rocprim_size, d_recvList[i], d_recvBuffer,
        d_haloList[i], d_haloBuffer, entriesToRecv));
    CUDA_CHECK_COMMAND(cudaMalloc(&rocprim_buffer, rocprim_size));

    // // Sort receive index array and halo index array
    CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortPairs(
        rocprim_buffer, rocprim_size, d_recvList[i], d_recvBuffer,
        d_haloList[i], d_haloBuffer, entriesToRecv));
    CUDA_CHECK_COMMAND(cudaFree(rocprim_buffer));
    rocprim_buffer = NULL;

    // Swap receive buffer pointers
    int *gptr = d_recvBuffer;
    d_recvBuffer = d_recvList[i];
    d_recvList[i] = gptr;

    // Swap halo buffer pointers
    int *lptr = d_haloBuffer;
    d_haloBuffer = d_haloList[i];
    d_haloList[i] = lptr;

    // No need to allocate new memory, we can use existing buffers
    int *d_num_runs =
        reinterpret_cast<int *>(A.d_send_buffer);
    int *d_offsets = reinterpret_cast<int *>(d_recvBuffer);
    int *d_unique_out = reinterpret_cast<int *>(d_haloBuffer);
    ;

    // Obtain rocprim buffer size
    // TODO: implement run-length encoding for cuda
    cub::DeviceRunLengthEncode::Encode(rocprim_buffer, rocprim_size, d_recvList[i], d_unique_out, d_offsets +1, d_num_runs, entriesToRecv);
    
    // CUDA_CHECK_COMMAND(rocprim::run_length_encode(
    //     rocprim_buffer, rocprim_size, d_recvList[i], entriesToRecv,
    //     d_unique_out, d_offsets + 1, d_num_runs));
    CUDA_CHECK_COMMAND(cudaMalloc(&rocprim_buffer, rocprim_size));

    // Perform a run length encode over the receive indices to obtain the number
    // of halo entries in each row
    // CUDA_CHECK_COMMAND(rocprim::run_length_encode(
    //     rocprim_buffer, rocprim_size, d_recvList[i], entriesToRecv,
    //     d_unique_out, d_offsets + 1, d_num_runs));

    cub::DeviceRunLengthEncode::Encode(rocprim_buffer, rocprim_size, d_recvList[i], d_unique_out, d_offsets +1, d_num_runs, entriesToRecv);
    
    CUDA_CHECK_COMMAND(cudaFree(rocprim_buffer));
    rocprim_buffer = NULL;

    // Copy the number of halo entries with respect to the i-th neighbor
    int currentRankHaloEntries;
    CUDA_CHECK_COMMAND(cudaMemcpy(&currentRankHaloEntries, d_num_runs,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost));

    // Store the number of halo entries we need to get from i-th neighbor
    A.receiveLength[neighborCount] = currentRankHaloEntries;

    // d_offsets[0] = 0
    CUDA_CHECK_COMMAND(cudaMemset(d_offsets, 0, sizeof(int)));

    // Obtain rocprim buffer size
    AddOp add_op;
    // CUDA_CHECK_COMMAND(cub::DeviceScan::InclusiveScan(rocprim_buffer, rocprim_size, d_offsets +1, d_offsets+1, add_op, currentRankHaloEntries));

    CUDA_CHECK_COMMAND(cub::DeviceScan::InclusiveSum(rocprim_buffer, rocprim_size, d_offsets +1, d_offsets+1, currentRankHaloEntries));
    // CUDA_CHECK_COMMAND(rocprim::inclusive_scan(
    //     rocprim_buffer, rocprim_size, d_offsets + 1, d_offsets + 1,
    //     currentRankHaloEntries, rocprim::plus<int>()));
    CUDA_CHECK_COMMAND(cudaMalloc(&rocprim_buffer, rocprim_size));

    // Perform inclusive sum to obtain the offsets to the first halo entry of
    // each row
    CUDA_CHECK_COMMAND(cub::DeviceScan::InclusiveSum(rocprim_buffer, rocprim_size, d_offsets +1, d_offsets+1, currentRankHaloEntries));

    // CUDA_CHECK_COMMAND(cub::DeviceScan::InclusiveScan(rocprim_buffer, rocprim_size, d_offsets +1, d_offsets+1, add_op, currentRankHaloEntries));

    // CUDA_CHECK_COMMAND(rocprim::inclusive_scan(
    //   rocprim_buffer, rocprim_size, d_offsets + 1, d_offsets + 1,
    //   currentRankHaloEntries, rocprim::plus<int>()));
    // 1/4
    // inclusive_scan(InputIterator first, InputIterator last, OutputIterator
    // result) -> OutputIterator

    CUDA_CHECK_COMMAND(cudaFree(rocprim_buffer));
    rocprim_buffer = NULL;
    // // hipError_t inclusive_scan(void * temporary_storage,
    //   size_t& storage_size,
    //   InputIterator input,
    //   OutputIterator output,
    //   const size_t size,
    //   BinaryFunction scan_op = BinaryFunction(),
    //   const hipStream_t stream = 0,
    //   bool debug_synchronous = false)
    // {
    // Launch kernel to fill all halo columns in the local matrix column index
    // array for the i-th neighbor
    kernel_halo_columns<128>
        <<<dim3((currentRankHaloEntries - 1) / 128 + 1), dim3(128)>>>(
            currentRankHaloEntries, A.localNumberOfRows,
            A.numberOfExternalValues, d_haloList[i], d_offsets, A.d_mtxIndL);

    // Increase the number of external values by i-th neighbors halo entry
    // contributions
    A.numberOfExternalValues += currentRankHaloEntries;

    // Store the "real" neighbor id for i-th neighbor
    A.neighbors[neighborCount++] = neighbors[i];
  }

  // Free up data
  CUDA_CHECK_COMMAND(cudaFree(d_recv_indices));
  CUDA_CHECK_COMMAND(cudaFree(d_halo_indices));

  // Allocate MPI communication structures
  A.recv_request = new MPI_Request[A.numberOfSendNeighbors];
  A.send_request = new MPI_Request[A.numberOfSendNeighbors];

  // Store contents in our matrix struct
  A.localNumberOfColumns = A.localNumberOfRows + A.numberOfExternalValues;
#endif
}

void CopyHaloToHostInside(SparseMatrix &A) {
#ifndef HPCG_NO_MPI
  // Allocate host structures
  A.elementsToSend = new int[A.totalToBeSent];
  A.sendBuffer = new double[A.totalToBeSent];

  // Copy GPU data to host
  CUDA_CHECK_COMMAND(cudaMemcpy(A.elementsToSend, A.d_elementsToSend,
                                sizeof(int) * A.totalToBeSent,
                                cudaMemcpyDeviceToHost));
#endif
  CUDA_CHECK_COMMAND(cudaMemcpy(A.mtxIndL[0], A.d_mtxIndL,
                                sizeof(int) * A.localNumberOfRows *
                                    27,
                                cudaMemcpyDeviceToHost));
}
