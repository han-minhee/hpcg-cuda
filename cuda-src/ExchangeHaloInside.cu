#ifndef HPCG_NO_MPI

#include "../src/Geometry.hpp"
#include "ExchangeHaloInside.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>

/*!
  Communicates data that is at the border of the part of the domain assigned to
  this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be
  communicated; on exit: the vector with non-local entries updated by other
  processors
 */
void ExchangeHaloInside(const SparseMatrix &A, Vector &x) {

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t *receiveLength = A.receiveLength;
  local_int_t *sendLength = A.sendLength;
  int *neighbors = A.neighbors;
  double *sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t *elementsToSend = A.elementsToSend;

  double *const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request *request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  double *x_external = (double *)xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
              MPI_COMM_WORLD, request + i);
    x_external += n_recv;
  }

  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i = 0; i < totalToBeSent; i++)
    sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
             MPI_COMM_WORLD);
    sendBuffer += n_send;
  }

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if (MPI_Wait(request + i, &status)) {
      std::exit(-1); // TODO: have better error exit
    }
  }

  delete[] request;

  return;
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_gather(local_int_t size, const double * in,
                       const local_int_t * map,
                       const local_int_t * perm,
                       double * out) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  out[gid] = in[perm[map[gid]]];
}

void PrepareSendBufferInside(const SparseMatrix &A, const Vector &x) {
  // Prepare send buffer
  kernel_gather<128><<<dim3((A.totalToBeSent - 1) / 128 + 1), dim3(128)>>>(
      A.totalToBeSent, x.d_values, A.d_elementsToSend, A.perm, A.d_send_buffer);

  // Copy send buffer to host
  CUDA_CHECK_COMMAND(cudaMemcpyAsync(A.send_buffer, A.d_send_buffer,
                                     sizeof(double) * A.totalToBeSent,
                                     cudaMemcpyDeviceToHost, stream_halo));
}

void ExchangeHaloAsyncInside(const SparseMatrix &A) {
  int num_neighbors = A.numberOfSendNeighbors;
  int MPI_MY_TAG = 99;

  // Post async boundary receives
  local_int_t offset = 0;

  for (int n = 0; n < num_neighbors; ++n) {
    local_int_t nrecv = A.receiveLength[n];

    MPI_Irecv(A.recv_buffer + offset, nrecv, MPI_DOUBLE, A.neighbors[n],
              MPI_MY_TAG, MPI_COMM_WORLD, A.recv_request + n);

    offset += nrecv;
  }

  // Synchronize stream to make sure that send buffer is available
  CUDA_CHECK_COMMAND(cudaStreamSynchronize(stream_halo));

  // Post async boundary sends
  offset = 0;

  for (int n = 0; n < num_neighbors; ++n) {
    local_int_t nsend = A.sendLength[n];

    MPI_Isend(A.send_buffer + offset, nsend, MPI_DOUBLE, A.neighbors[n],
              MPI_MY_TAG, MPI_COMM_WORLD, A.send_request + n);

    offset += nsend;
  }
}

void ObtainRecvBufferInside(const SparseMatrix &A, Vector &x) {
  int num_neighbors = A.numberOfSendNeighbors;

  // Synchronize boundary transfers
  EXIT_IF_HPCG_ERROR(
      MPI_Waitall(num_neighbors, A.recv_request, MPI_STATUSES_IGNORE));
  EXIT_IF_HPCG_ERROR(
      MPI_Waitall(num_neighbors, A.send_request, MPI_STATUSES_IGNORE));

  // Update boundary values
  CUDA_CHECK_COMMAND(cudaMemcpyAsync(
      x.d_values + A.localNumberOfRows, A.recv_buffer,
      sizeof(double) * A.totalToBeSent, cudaMemcpyHostToDevice, stream_halo));
}
#endif
// ifndef HPCG_NO_MPI
