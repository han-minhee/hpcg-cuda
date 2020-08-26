#include "MultiColoring.cuh"
#include "Utils.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

// ,
#define LAUNCH_JPL(blockSizeX, blockSizeY)                                     \
  kernelJPL<blockSizeX, blockSizeY>                                            \
      <<<dim3((m - 1) / blockSizeY + 1), dim3(blockSizeX, blockSizeY),         \
         2 * sizeof(bool) * blockSizeY, 0>>>(m, A.d_rowHash, color1, color2,   \
                                             A.d_nonzerosInRow, A.d_mtxIndL,   \
                                             A.perm)

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_identity(local_int_t size, local_int_t *data) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  data[gid] = gid;
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernel_create_perm(local_int_t size, const local_int_t *in,
                            local_int_t *out) {
  local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (gid >= size) {
    return;
  }

  out[in[gid]] = gid;
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
    void kernelCountColorsPart1(local_int_t size, local_int_t color,
                                const local_int_t *colors,
                                local_int_t *workspace) {
  local_int_t tid = threadIdx.x;
  local_int_t gid = blockIdx.x * BLOCKSIZE + tid;
  local_int_t inc = gridDim.x * BLOCKSIZE;

  __shared__ local_int_t sdata[BLOCKSIZE];

  local_int_t sum = 0;
  for (local_int_t idx = gid; idx < size; idx += inc) {
    if (colors[idx] == color) {
      ++sum;
    }
  }

  sdata[tid] = sum;

  kernelDeviceReduceSum<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[blockIdx.x] = sdata[0];
  }
}

template <unsigned int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void kernelCountColorsPart2(local_int_t size, local_int_t *workspace) {
  local_int_t tid = threadIdx.x;

  __shared__ local_int_t sdata[BLOCKSIZE];

  local_int_t sum = 0;
  for (local_int_t idx = tid; idx < size; idx += BLOCKSIZE) {
    sum += workspace[idx];
  }

  sdata[tid] = sum;

  kernelDeviceReduceSum<BLOCKSIZE>(tid, sdata);

  if (tid == 0) {
    workspace[0] = sdata[0];
  }
}

template <unsigned int BLOCKSIZEX, unsigned int BLOCKSIZEY>
__launch_bounds__(BLOCKSIZEX *BLOCKSIZEY) __global__
    void kernelJPL(local_int_t m, const local_int_t *hash, int color1,
                   int color2, const char *nonzerosInRow,
                   const local_int_t *mtxIndL, local_int_t *colors) {
  local_int_t row = blockIdx.x * BLOCKSIZEY + threadIdx.y;

  extern __shared__ bool sdata[];
  bool *min = &sdata[0];
  bool *max = &sdata[BLOCKSIZEY];

  // Assume current vertex is maximum
  if (threadIdx.x == 0) {
    min[threadIdx.y] = true;
    max[threadIdx.y] = true;
  }

  __syncthreads();

  if (row >= m) {
    return;
  }

  // Do not process already colored vertices
  if (colors[row] != -1) {
    return;
  }

  // Get row hash value
  local_int_t row_hash = hash[row];

  local_int_t idx = row * BLOCKSIZEX + threadIdx.x;
  local_int_t col = mtxIndL[idx];

  if (col >= 0 && col < m) {
    // Skip diagonal
    if (col != row) {
      // Get neighbors color
      int color_nb = colors[col];

      // Compare only with uncolored neighbors
      if (color_nb == -1 || color_nb == color1 || color_nb == color2) {
        // Get column hash value
        local_int_t col_hash = hash[col];

        // If neighbor has larger weight, vertex is not a maximum
        if (col_hash >= row_hash) {
          max[threadIdx.y] = false;
        }

        // If neighbor has lesser weight, vertex is not a minimum
        if (col_hash <= row_hash) {
          min[threadIdx.y] = false;
        }
      }
    }
  }

  __syncthreads();

  // If vertex is a maximum, color it
  if (threadIdx.x == 0) {
    if (max[threadIdx.y] == true) {
      colors[row] = color1;
    } else if (min[threadIdx.y] == true) {
      colors[row] = color2;
    }
  }
}

void JPLColoring(SparseMatrix &A) {
  local_int_t m = A.localNumberOfRows;

  CUDA_RETURN_VOID_IF_ERROR(
      cudaMalloc((void **)&A.perm, sizeof(local_int_t) * m));
  CUDA_RETURN_VOID_IF_ERROR(cudaMemset(A.perm, -1, sizeof(local_int_t) * m));

  A.nblocks = 0;

  // Color seed
  srand(RNG_SEED);

  // Temporary workspace
  local_int_t *tmp = reinterpret_cast<local_int_t *>(workspace);

  // Counter for uncolored vertices
  local_int_t colored = 0;

  // Number of vertices of each block
  A.sizes = new local_int_t[MAX_COLORS];

  // Offset into blocks
  A.offsets = new local_int_t[MAX_COLORS];
  A.offsets[0] = 0;

  // Determine blocksize
  unsigned int blocksize = 16;

  // DONE: hash identical
  // local_int_t* hash_temp = new local_int_t[A.localNumberOfRows];
  // cudaMemcpy(hash_temp, A.d_rowHash, sizeof(local_int_t) * A.localNumberOfRows, cudaMemcpyDeviceToHost);

  // for(int i =0; i<A.localNumberOfRows; i++){
  //     printf("%d th hash: %d\n", i, hash_temp[i]);
  // }

  // Run Jones-Plassmann Luby algorithm until all vertices have been colored
  while (colored != m) {
    // The first 8 colors are selected by RNG, afterwards we just count upwards
    int color1 = (A.nblocks < 8) ? rand() % 8 : A.nblocks;
    int color2 = (A.nblocks < 8) ? rand() % 8 : A.nblocks + 1;
    
    
    // printf("color1: %d\n", color1);
    // printf("color2: %d\n", color2);

    

    LAUNCH_JPL(27, 16);

    // printf("entering coloring 1\n");
    // int* temp_perm = new int[100];
    // cudaMemcpy(temp_perm, A.perm, sizeof(local_int_t) * 100, cudaMemcpyDeviceToHost);
    // for (int i= 0; i<10;i++){
    //   printf("%d perm : %d\n", i, temp_perm[i]);
    // }
    // free(temp_perm);

    //NOW COloring works : 08/27/2020 01:11
    // not? 08/27/2020 01:43

    // int totalSum = 0;

    // for (int i = 0; i<m; i++){
    //   if(temp_perm[i] == color1){
    //     ++totalSum;
    //   }
      
    // }

    // Count colored vertices
    kernelCountColorsPart1<256>
        <<<dim3(256), dim3(256)>>>(m, color1, A.perm, tmp);

    printf("entering coloring 2\n");
     kernelCountColorsPart2<256><<<dim3(1), dim3(256)>>>(256, tmp);

    //      int* temp_result = new int[1];
    // cudaMemcpy(temp_result, tmp, sizeof(local_int_t), cudaMemcpyDeviceToHost);
    // printf("coloring part2 : %d\n", temp_result[0]);
    //   free(temp_result);

    // Copy colored max vertices for current iteration to host
    CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(
        &A.sizes[A.nblocks], tmp, sizeof(local_int_t), cudaMemcpyDeviceToHost));

        printf("colored: %d, sizes: %d\n", colored, A.sizes[A.nblocks]);

    kernelCountColorsPart1<256>
        <<<dim3(256), dim3(256)>>>(m, color2, A.perm, tmp);

    kernelCountColorsPart2<256><<<dim3(1), dim3(256)>>>(256, tmp);

    // Copy colored min vertices for current iteration to host
    CUDA_RETURN_VOID_IF_ERROR(cudaMemcpy(&A.sizes[A.nblocks + 1], tmp,
                                         sizeof(local_int_t),
                                         cudaMemcpyDeviceToHost));

    // Total number of colored vertices after max
    colored += A.sizes[A.nblocks];
    A.offsets[A.nblocks + 1] = colored;
    ++A.nblocks;

    // Total number of colored vertices after min
    colored += A.sizes[A.nblocks];
    A.offsets[A.nblocks + 1] = colored;
    ++A.nblocks;
  }

  A.ublocks = A.nblocks - 1;

  CUDA_CHECK_COMMAND(cudaFree(A.d_rowHash));

  local_int_t *tmp_color;
  local_int_t *tmp_perm;
  local_int_t *perm;

  CUDA_CHECK_COMMAND(cudaMalloc((void **)&tmp_color, sizeof(local_int_t) * m));
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&tmp_perm, sizeof(local_int_t) * m));
  CUDA_CHECK_COMMAND(cudaMalloc((void **)&perm, sizeof(local_int_t) * m));

  kernel_identity<1024><<<dim3((m - 1) / 1024 + 1), dim3(1024)>>>(m, perm);

  // rocprim::double_buffer<local_int_t> keys(A.perm, tmp_color);
  // rocprim::double_buffer<local_int_t> vals(perm, tmp_perm);

  size_t size = 0;
  void *buf = NULL;

  int startbit = 0;
  int endbit = 32 - __builtin_clz(A.nblocks);
  printf("nblocks : %d\n", A.nblocks);

  // SortPairs (void *d_temp_storage, size_t &temp_storage_bytes, const KeyT
  // *d_keys_in, KeyT *d_keys_out, const ValueT *d_values_in, ValueT
  // *d_values_out, int num_items, int begin_bit=0, int end_bit=sizeof(KeyT)*8,
  // cudaStream_t stream=0, bool debug_synchronous=false)
  // TODO: radix_sort_pairs

  CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortPairs(
      buf, size, A.perm, tmp_color, perm, tmp_perm, m, startbit, endbit));
  CUDA_CHECK_COMMAND(cudaMalloc(&buf, size));
  CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortPairs(
    buf, size, A.perm, tmp_color, perm, tmp_perm, m, startbit, endbit));
  CUDA_CHECK_COMMAND(cudaFree(buf));

//   CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortPairs(
//     buf, size, tmp_color, A.perm, tmp_perm, perm, m, startbit, endbit));
// CUDA_CHECK_COMMAND(cudaMalloc(&buf, size));
// CUDA_CHECK_COMMAND(cub::DeviceRadixSort::SortPairs(
//   buf, size, tmp_color, A.perm, tmp_perm, perm, m, startbit, endbit));
//   CUDA_CHECK_COMMAND(cudaFree(buf));

  local_int_t * keys_temp = new local_int_t[10];
  local_int_t * vals_temp = new local_int_t[10];
  cudaMemcpy(keys_temp, A.perm
  , sizeof(local_int_t) * 10, cudaMemcpyDeviceToHost);
  cudaMemcpy(vals_temp, perm, sizeof(local_int_t) * 10, cudaMemcpyDeviceToHost);
  for (int i = 0; i< 10; i++){
    printf("keys[%d] : %d\n", i, keys_temp[i]);
    printf("vals[%d] : %d\n", i, vals_temp[i]);
}

  cudaMemcpy(keys_temp, tmp_color
  , sizeof(local_int_t) * 10, cudaMemcpyDeviceToHost);
  cudaMemcpy(vals_temp, tmp_perm, sizeof(local_int_t) * 10, cudaMemcpyDeviceToHost);

  for (int i = 0; i< 10; i++){
      printf("keys_temp[%d] : %d\n", i, keys_temp[i]);
      printf("vals_temp[%d] : %d\n", i, vals_temp[i]);
  }




  kernel_create_perm<1024>
      <<<dim3((m - 1) / 1024 + 1), dim3(1024)>>>(m, tmp_perm, A.perm);

      cudaMemcpy(keys_temp, A.perm, sizeof(local_int_t) * 10, cudaMemcpyDeviceToHost);

      for (int i = 0; i< 10; i++){
        printf("final perm[%d] : %d\n", i, keys_temp[i]);
    }
  
    //DONE: 08/27/2020 02:24

  CUDA_CHECK_COMMAND(cudaFree(tmp_color));
  CUDA_CHECK_COMMAND(cudaFree(tmp_perm));
  CUDA_CHECK_COMMAND(cudaFree(perm));
  free(keys_temp);
  free(vals_temp);
#ifndef HPCG_REFERENCE
  --A.ublocks;
#endif
}
