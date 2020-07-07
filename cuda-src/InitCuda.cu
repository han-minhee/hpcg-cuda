#include "InitCuda.cuh"
#include <cassert>
#include <cudart>

// SMs contain SPs
extern int numSM;
extern int numSP;

extern int threadsPerBlock;
extern int warpSize;
extern int regsPerBlock;
extern int multiProcessorCount;


// macro ref: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline int gpuAssert(cudaError_t code)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
      return -1;
   }
   else{
       return 0;
   }
}

int getDeviceProp(cudaDeviceProp deviceProp){
    if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
        return 0;
    }
    else{
        return 1;
    }
}

int main(void) {
    cudaDeviceProp deviceProp;
    
    int getDevicePropSuccess = getDeviceProp(deviceProp);
    if(!getDevicePropSuccess){
        return -1;
    }

    warpSize = deviceProp.warpSize;
    regsPerBlock = deviceProp.regsPerBlock;
    multiProcessorCount = deviceProp.multiProcessorCount;
}