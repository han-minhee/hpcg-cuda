#include "../cuda-src/finalizeInside.cuh"

/*!
  Closes the I/O stream used for logging information throughout the HPCG run.

  @return returns 0 upon success and non-zero otherwise

  @see HPCG_Init
*/
int HPCG_Finalize(void) { return HPCG_FinalizeInside(); }
