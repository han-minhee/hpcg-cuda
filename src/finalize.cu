
#include <cuda_runtime.h>
#include <fstream>

#include "Utils.cuh"
#include "hpcg.cuh"

/*!
  Closes the I/O stream used for logging information throughout the HPCG run.

  @return returns 0 upon success and non-zero otherwise

  @see HPCG_Init
*/
int HPCG_Finalize(void) {
  HPCG_fout.close();

  // Destroy streams
  CUDA_CHECK_COMMAND(cudaStreamDestroy(stream_interior));
  CUDA_CHECK_COMMAND(cudaStreamDestroy(stream_halo));

  // Free workspace
  CUDA_CHECK_COMMAND(cudaFree(workspace));

#ifdef HPCG_MEMMGMT
  // Clear allocator
  CUDA_CHECK_COMMAND(allocator.Clear());
#endif

  // Reset HIP device
  cudaDeviceReset();

  return 0;
}
