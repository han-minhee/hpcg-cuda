 #include <cassert>
 #include "ComputeDotProduct.cuh"
 
 // notice: originally, for the use of MPI, there should be result, time_allreduce variable,
 // but currently omitted.
 // instead of changing a value, it returns a double var.
 __global__ double kernelDotProduct(
  int n,
  double *  xv,
  double *  yv,
  double * result;
)               
{
  int localIndex;
  int elemsPerThreads = warpSize;

  int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
  globalIndex *= elemsPerThreads;
  if(globalIndex > n) return;

  double localResult = 0.0;

for(localIndex = globalIndex; localIndex < globalIndex + elemsPerThreads; localIndex++){
      localResult += xv[localIndex] * yv[localIndex];
}
  result = localResult;
}

 int ComputeDotProduct_cuda(const local_int_t n, const Vector & x, const Vector & y,
     double & result, double & time_allreduce) {
   assert(x.localLength>=n); // Test vector lengths
   assert(y.localLength>=n);
 
   double local_result = 0.0;
   double * xv = x.values;
   double * yv = y.values;

   cudaMalloc(xv_d, n_size);
   cudaMalloc(yv_d, n_size);

   cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);
   cudaMemcpy(xv_d, xv, n_size, cudaMemcpyHostToDevice);

   int numBlocks = ( n + warpSize -1 ) / warpSize;
   kernelWAXPBY<<<numBlocks, warpSize>>>(n, xv_d, yv_d, wv_d);
   
   if(gpuAssert( cudaPeekAtLastError()) == -1  ){
    return -1; 
   }
   if (gpuAssert( cudaDeviceSynchronize()) == -1){
     return -1;
   }
 }
 