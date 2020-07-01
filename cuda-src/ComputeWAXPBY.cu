
 #include "ComputeWAXPBY_cuda.hpp"

 #include <cassert>
 /*!
   Routine to compute the update of a vector with the sum of two
   scaled vectors where: w = alpha*x + beta*y
 
   This is the reference WAXPBY impmentation.  It CANNOT be modified for the
   purposes of this benchmark.
 
   @param[in] n the number of vector elements (on this processor)
   @param[in] alpha, beta the scalars applied to x and y respectively.
   @param[in] x, y the input vectors
   @param[out] w the output vector.
 
   @return returns 0 upon success and non-zero otherwise
 
   @see ComputeWAXPBY
 */


// cublasDcopy
/*

cublasStatus_t cublasDcopy(cublasHandle_t handle, int n,
                           const double          *x, int incx,
                           double                *y, int incy)

cublasStatus_t  cublasDscal(cublasHandle_t handle, int n,
                            const double          *alpha,
                            double          *x, int incx)
                            
cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n,
                           const double          *alpha,
                           const double          *x, int incx,
                           double                *y, int incy)

*/
// cublasDaxpy

__global__ void kernelWAXPBY(int n
		  double alpha,
		  double beta,
		  double *  xv,
		  double *  yv,
		  double *  wv
		)               
{
    int global_index;
    int local_index;
	for(local_index = global_index; local_index < /* FIXME : limits*/; local_index++){
			if (alpha==1.0) {
			    wv[x] = xv[local_index] + beta * yv[local_index];
			} else if (beta==1.0) {
			    wv[x] = alpha * xv[local_index] + yv[local_index];
			} else  {
			    wv[x] = alpha * xv[local_index] + beta * yv[local_index];
			}
			
	}
}

 int ComputeWAXPBY_cuda_0(const local_int_t n, const double alpha, const Vector & x,
     const double beta, const Vector & y, Vector & w) {
 
        // Test vector lengths
        // should also be in the kernel?
   assert(x.localLength>=n); 
   assert(y.localLength>=n);
 
   const double * const xv = x.values;
   const double * const yv = y.values;
   double * const wv = w.values;

   //global dim : n
   //local dim : for now, just 1?

   /*
   if (alpha==1.0) {
     for (local_int_t i=0; i<n; i++) wv[i] = xv[i] + beta * yv[i];
   } else if (beta==1.0) {
     for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + yv[i];
   } else  {
     for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
   }
*/ 
   return 0;
 }
 