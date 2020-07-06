 #include <cassert>
 #include "ComputeDotProduct.cuh"
 
 // notice: originally, for the use of MPI, there should be result, time_allreduce variable,
 // but currently omitted.
 // instead of changing a value, it returns a double var.
 __global__ double kernelDotProduct(
  int n,
  int itemsPerThreads,
  double *  xv,
  double *  yv,
  double * result;
)               
{
  int globalIndex = blockDim.x * blockIdx.x + threadIdx.x;
  globalIndex *= elemsPerThreads;

  if(globalIndex > n) return;

  int localIndex;
  double localResult = 0.0;

for(localIndex = globalIndex; localIndex < globalIndex + elemsPerThreads; localIndex++){
      localResult = xv[localIndex] * yv[localIndex];
}
  result += localResult;
}

 int ComputeDotProduct_cuda(const local_int_t n, const Vector & x, const Vector & y,
     double & result, double & time_allreduce) {
   assert(x.localLength>=n); // Test vector lengths
   assert(y.localLength>=n);
 
   double local_result = 0.0;
   double * xv = x.values;
   double * yv = y.values;

   int elemsPerThreads = 0;

   // cudaOccupancyMaxPotentialBlockSize
   // FIXME: dimension
   dim3 globalDim();
   dim3 blockDim();

   result = kernelDotProduct<<<globalDim, blockDim>>>(n, itemsPerThreads, xv, yv);
 
   return 0;
 }
 