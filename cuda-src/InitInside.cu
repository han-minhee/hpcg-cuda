#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
const char *NULLDEVICE = "nul";
#else
const char *NULLDEVICE = "/dev/null";
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "../src/hpcg.hpp"
#include "Utils.cuh"

#include "../src/ReadHpcgDat.hpp"
#include "InitInside.cuh"

cudaStream_t stream_interior;
cudaStream_t stream_halo;
void *workspace;

std::ofstream
    HPCG_fout; //!< output file stream for logging activities during HPCG run

static int startsWith(const char *s, const char *prefix) {
  size_t n = strlen(prefix);
  if (strncmp(s, prefix, n))
    return 0;
  return 1;
}

__global__ void kernelWarmUp() {}

int HPCG_InitInside(int *argc_p, char ***argv_p, HPCG_Params &params) {
  int argc = *argc_p;
  char **argv = *argv_p;
  char fname[80];
  int i, j, *iparams;
  char cparams[][8] = {"--nx=", "--ny=",  "--nz=",  "--rt=",  "--pz=", "--zl=",
                       "--zu=", "--npx=", "--npy=", "--npz=", "--dev="};
  time_t rawtime;
  tm *ptm;
  const int nparams = (sizeof cparams) / (sizeof cparams[0]);
  bool broadcastParams = false; // Make true if parameters read from file.

  iparams = (int *)malloc(sizeof(int) * nparams);

  // Initialize iparams
  for (i = 0; i < nparams; ++i)
    iparams[i] = 0;

  /* for sequential and some MPI implementations it's OK to read first three
   * args */
  for (i = 0; i < nparams; ++i)
    if (argc <= i + 1 || sscanf(argv[i + 1], "%d", iparams + i) != 1 ||
        iparams[i] < 10)
      iparams[i] = 0;

  /* for some MPI environments, command line arguments may get complicated so we
   * need a prefix */
  for (i = 1; i <= argc && argv[i]; ++i)
    for (j = 0; j < nparams; ++j)
      if (startsWith(argv[i], cparams[j]))
        if (sscanf(argv[i] + strlen(cparams[j]), "%d", iparams + j) != 1)
          iparams[j] = 0;

  // Check if --rt was specified on the command line
  int *rt = iparams + 3; // Assume runtime was not specified and will be read
                         // from the hpcg.dat file
  if (iparams[3])
    rt = 0; // If --rt was specified, we already have the runtime, so don't read
            // it from file
  if (!iparams[0] && !iparams[1] &&
      !iparams[2]) { /* no geometry arguments on the command line */
    ReadHpcgDat(iparams, rt, iparams + 7);
    broadcastParams = true;
  }

  // Check for small or unspecified nx, ny, nz values
  // If any dimension is less than 16, make it the max over the other two
  // dimensions, or 16, whichever is largest
  for (i = 0; i < 3; ++i) {
    if (iparams[i] < 16)
      for (j = 1; j <= 2; ++j)
        if (iparams[(i + j) % 3] > iparams[i])
          iparams[i] = iparams[(i + j) % 3];
    if (iparams[i] < 16)
      iparams[i] = 16;
  }

// Broadcast values of iparams to all MPI processes
#ifndef HPCG_NO_MPI
  if (broadcastParams) {
    MPI_Bcast(iparams, nparams, MPI_INT, 0, MPI_COMM_WORLD);
  }
#endif

  params.nx = iparams[0];
  params.ny = iparams[1];
  params.nz = iparams[2];

  params.runningTime = iparams[3];
  params.pz = iparams[4];
  params.zl = iparams[5];
  params.zu = iparams[6];

  params.npx = iparams[7];
  params.npy = iparams[8];
  params.npz = iparams[9];

  params.device = iparams[10];

#ifndef HPCG_NO_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &params.comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &params.comm_size);
  printf("MPI enabled\n");
#else
  printf("MPI disabled\n");
  params.comm_rank = 0;
  params.comm_size = 1;
#endif

  printf("mpi comm rank, %d %d size \n", params.comm_rank, params.comm_size);
  // Simple device management
  int ndevs = 0;
  CUDA_CHECK_COMMAND(cudaGetDeviceCount(&ndevs));
  printf("devices: %d \n", ndevs);

  // Single GPU device can be selected via cli
  // Multi GPU devices are selected automatically
  if (params.comm_size == 1) {
    if (ndevs <= params.device) {
      fprintf(stderr, "Error: invalid device ID\n");
      cudaDeviceReset();
      exit(1);
    }
  } else {
    params.device = params.comm_rank % ndevs;
  }

  // Set device
  CUDA_CHECK_COMMAND(cudaSetDevice(params.device));
  

  // Warm up
  kernelWarmUp<<<1, 1, 0, 0>>>();

  // Create streams
  CUDA_CHECK_COMMAND(cudaStreamCreate(&stream_interior));
  CUDA_CHECK_COMMAND(cudaStreamCreate(&stream_halo));

  // Allocate device workspace
  CUDA_CHECK_COMMAND(
      cudaMalloc((void **)&workspace, sizeof(local_int_t) * 1024));

#ifdef HPCG_NO_OPENMP
  params.numThreads = 1;
#else
#pragma omp parallel
  params.numThreads = omp_get_num_threads();
#endif

  time(&rawtime);
  ptm = localtime(&rawtime);
  sprintf(fname, "hpcg%04d%02d%02dT%02d%02d%02d.txt", 1900 + ptm->tm_year,
          ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min,
          ptm->tm_sec);

  if (0 == params.comm_rank) {
    HPCG_fout.open(fname);
  } else {
#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
    sprintf(fname, "hpcg%04d%02d%02dT%02d%02d%02d_%d.txt", 1900 + ptm->tm_year,
            ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min,
            ptm->tm_sec, params.comm_rank);
    HPCG_fout.open(fname);
#else
    HPCG_fout.open(NULLDEVICE);
#endif
  }

  free(iparams);

  return 0;
}
