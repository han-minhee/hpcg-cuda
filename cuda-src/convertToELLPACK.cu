void ConvertToELL(SparseMatrix &A) {
  // We can re-use mtxIndL array for ELL values
  A.ell_val = reinterpret_cast<double *>(A.d_mtxIndG);
  A.d_mtxIndG = NULL;

  // Resize
  HIP_CHECK(deviceRealloc((void *)A.ell_val,
                          sizeof(double) * A.ell_width * A.localNumberOfRows));

  // Determine blocksize
  unsigned int blocksize = 1024 / A.ell_width;

  // Compute next power of two
  blocksize |= blocksize >> 1;
  blocksize |= blocksize >> 2;
  blocksize |= blocksize >> 4;
  blocksize |= blocksize >> 8;
  blocksize |= blocksize >> 16;
  ++blocksize;

  // Shift right until we obtain a valid blocksize
  while (blocksize * A.ell_width > 1024) {
    blocksize >>= 1;
  }

  if (blocksize == 32)
    LAUNCH_TO_ELL_VAL(27, 32);
  else if (blocksize == 16)
    LAUNCH_TO_ELL_VAL(27, 16);
  else if (blocksize == 8)
    LAUNCH_TO_ELL_VAL(27, 8);
  else
    LAUNCH_TO_ELL_VAL(27, 4);

  // We can re-use mtxIndG array for the ELL column indices
  A.ell_col_ind = reinterpret_cast<local_int_t *>(A.d_matrixValues);
  A.d_matrixValues = NULL;

  // Resize the array
  HIP_CHECK(
      deviceRealloc((void *)A.ell_col_ind,
                    sizeof(local_int_t) * A.ell_width * A.localNumberOfRows));

  // Convert mtxIndL into ELL column indices
  local_int_t *d_halo_rows = reinterpret_cast<local_int_t *>(workspace);

#ifndef HPCG_NO_MPI
  HIP_CHECK(deviceMalloc((void **)&A.halo_row_ind,
                         sizeof(local_int_t) * A.totalToBeSent));

  HIP_CHECK(hipMemset(d_halo_rows, 0, sizeof(local_int_t)));
#endif

  if (blocksize == 32)
    LAUNCH_TO_ELL_COL(27, 32);
  else if (blocksize == 16)
    LAUNCH_TO_ELL_COL(27, 16);
  else if (blocksize == 8)
    LAUNCH_TO_ELL_COL(27, 8);
  else
    LAUNCH_TO_ELL_COL(27, 4);

  // Free old matrix indices
  HIP_CHECK(deviceFree(A.d_mtxIndL));

#ifndef HPCG_NO_MPI
  HIP_CHECK(hipMemcpy(&A.halo_rows, d_halo_rows, sizeof(local_int_t),
                      hipMemcpyDeviceToHost));
  assert(A.halo_rows <= A.totalToBeSent);

  HIP_CHECK(deviceMalloc((void **)&A.halo_col_ind,
                         sizeof(local_int_t) * A.ell_width * A.halo_rows));
  HIP_CHECK(deviceMalloc((void **)&A.halo_val,
                         sizeof(double) * A.ell_width * A.halo_rows));

  size_t rocprim_size;
  void *rocprim_buffer = NULL;
  HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer, rocprim_size,
                                     A.halo_row_ind, A.halo_row_ind,
                                     A.halo_rows));
  HIP_CHECK(deviceMalloc(&rocprim_buffer, rocprim_size));
  HIP_CHECK(rocprim::radix_sort_keys(rocprim_buffer, rocprim_size,
                                     A.halo_row_ind,
                                     A.halo_row_ind, // TODO inplace!
                                     A.halo_rows));
  HIP_CHECK(deviceFree(rocprim_buffer));

  hipLaunchKernelGGL((kernel_to_halo<128>), dim3((A.halo_rows - 1) / 128 + 1),
                     dim3(128), 0, 0, A.halo_rows, A.localNumberOfRows,
                     A.localNumberOfColumns, A.ell_width, A.ell_col_ind,
                     A.ell_val, A.halo_row_ind, A.halo_col_ind, A.halo_val);
#endif
}