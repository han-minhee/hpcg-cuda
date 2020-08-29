#include <cmath>
#include <fstream>

#include "hpcg.hpp"

#include "CG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeMG.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeWAXPBY.hpp"
#include "mytimer.hpp"

#include <cuda_runtime.h>

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()                                                                 \
  cudaDeviceSynchronize();                                                     \
  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t)                                                                \
  cudaDeviceSynchronize();                                                     \
  t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors
  preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate
  solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if
  tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm
  of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last
  iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first
  iteration.
  @param[out]   times     The 7-element vector of the timing information
  accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the
  preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int CG(const SparseMatrix &A, CGData &data, const Vector &b, Vector &x,
       const int max_iter, const double tolerance, int &niters, double &normr,
       double &normr0, double *times, bool doPreconditioning) {
  printf("====entering CG====\n");
  double t_begin = mytimer(); // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;

  //#ifndef HPCG_NO_MPI
  //  double t6 = 0.0;
  //#endif

  local_int_t nrow = A.localNumberOfRows;
  Vector &r = data.r; // Residual vector
  Vector &z = data.z; // Preconditioned residual vector
  Vector &p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector &Ap = data.Ap;


  // double * ellVal = new double[100];
  // double * rVal = new double[100];
  // double * zVal = new double[100];
  // double * pVal = new double[100];
  // double * ApVal = new double[100];

  if (!doPreconditioning && A.geom->rank == 0)
    HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq > 50)
    print_freq = 50;
  if (print_freq < 1)
    print_freq = 1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  //     // printf("localLength %d\n", x.localLength);

    // cudaMemcpy(ApVal, x.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);
    // // printf("localLength %d\n", 100);

    //  for(int i = 0; i < 100; i++)
    // {
    //     printf("before copy, xVal [%d] %f \n", i, ApVal[i]);
    // }

// in the test, zeroVector

  CudaCopyVector(x, p);
  //   cudaMemcpy(pVal, p.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);
  //   cudaMemcpy(ApVal, Ap.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);

  //   for(int i = 0; i<100; i++){
  //     printf("before SPMV pVal, ApVal [%d] %f %f \n", i, pVal[i], ApVal[i]);
  //   }

  TICK();
  // in the test, should be zero
  ComputeSPMV(A, p, Ap);
  TOCK(t3); // Ap = A*p


    // cudaMemcpy(pVal, p.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);
    // cudaMemcpy(ApVal, Ap.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<100; i++){
    //   printf("after SPMV pVal, ApVal [%d] %f %f \n", i, pVal[i], ApVal[i]);
    // }

  TICK();
  ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, r, A.isWaxpbyOptimized);
  TOCK(t2); // r = b - Ax (x stored in p)

  // cudaMemcpy(pVal, b.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);
  // cudaMemcpy(ApVal, r.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);

  //   for(int i = 0; i<100; i++){
  //     printf("after copy bVal, rVal [%d] %f %f \n", i, pVal[i], ApVal[i]);
  //   }

  TICK();
  ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized);
  TOCK(t1);
  normr = sqrt(normr);
      printf("current normr : %f\n", normr);

#ifdef HPCG_DEBUG
  if (A.geom->rank == 0)
    HPCG_fout << "Initial Residual = " << normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations
  printf("=== entering CG iterations\n");



  for (int k = 1; k <= max_iter && normr / normr0 > tolerance; k++) {

    // printf("beginning ell, r, z\n");

    // cudaMemcpy(ellVal, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(rVal, r.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    // cudaMemcpy(zVal, z.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<10; i++){
    //   printf("ell r z [%d] %f %f %f\n", i, ellVal[i], rVal[i], zVal[i]);
    // }



    TICK();
    if (doPreconditioning)
      ComputeMG(A, r, z); // Apply preconditioner
    else
      CudaCopyVector(r, z); // copy r to z (no preconditioning)
    TOCK(t5);               // Preconditioner apply time
    
    // printf("after MG, copy\n");
    // cudaMemcpy(ellVal, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(rVal, r.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    // cudaMemcpy(zVal, z.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<10; i++){
    //   printf("ell r z [%d] %f %f %f\n", i, ellVal[i], rVal[i], zVal[i]);
    // }

    if (k == 1) {
      TICK();
      ComputeWAXPBY(nrow, 1.0, z, 0.0, z, p, A.isWaxpbyOptimized);
      TOCK(t2); // Copy Mr to p
      TICK();
      ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized);
      TOCK(t1); // rtz = r'*z
    } else {
      oldrtz = rtz;
      TICK();
      ComputeDotProduct(nrow, r, z, rtz, t4, A.isDotProductOptimized);
      TOCK(t1); // rtz = r'*z
      beta = rtz / oldrtz;
      TICK();
      ComputeWAXPBY(nrow, 1.0, z, beta, p, p, A.isWaxpbyOptimized);
      TOCK(t2); // p = beta*p + z
    }

    //     printf("dotprod, waxpby\n");
    // cudaMemcpy(ellVal, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(rVal, r.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    // cudaMemcpy(zVal, z.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<10; i++){
    //   printf("ell r z [%d] %f %f %f\n", i, ellVal[i], rVal[i], zVal[i]);
    // }

    TICK();
    ComputeSPMV(A, p, Ap);
    TOCK(t3); // Ap = A*p

    //     printf("after spmv-final, copy\n");
    // cudaMemcpy(ellVal, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(rVal, r.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    // cudaMemcpy(zVal, z.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<10; i++){
    //   printf("ell r z [%d] %f %f %f\n", i, ellVal[i], rVal[i], zVal[i]);
    // }

    TICK();
    ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized);
    TOCK(t1); // alpha = p'*Ap

    //     printf("after dotProd-final, copy\n");
    // cudaMemcpy(ellVal, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(rVal, r.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    // cudaMemcpy(zVal, z.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<10; i++){
    //   printf("ell r z [%d] %f %f %f\n", i, ellVal[i], rVal[i], zVal[i]);
    // }

    alpha = rtz / pAp;
#ifndef HPCG_REFERENCE
    TICK();
    ComputeFusedWAXPBYDot(nrow, -alpha, Ap, r, normr, t4);
    // printf("fusedWaxpby parms at %d: %d, %f, %f\n", k, nrow, -alpha, normr);

    // cudaMemcpy(zVal, Ap.d_values, sizeof(double) * 100, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<100; i++){
    //   printf("Ap [%d] %f\n", i, zVal[i]);
    // }

    // printf("normr, normr0 at %d iter: %f, %f \n", k, normr, normr0);


    // printf("after fusedwaxpby, copy\n");
    // cudaMemcpy(ellVal, A.ell_val, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(rVal, r.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);
    // cudaMemcpy(zVal, z.d_values, sizeof(double) * 10, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<10; i++){
    //   printf("ell r z [%d] %f %f %f\n", i, ellVal[i], rVal[i], zVal[i]);
    // }

    ComputeWAXPBY(nrow, 1.0, x, alpha, p, x, A.isWaxpbyOptimized);
    TOCK(t2); // x = x + alpha*p
#else
    TICK();
    ComputeWAXPBY(nrow, 1.0, x, alpha, p, x,
                  A.isWaxpbyOptimized); // x = x + alpha*p
    ComputeWAXPBY(nrow, 1.0, r, -alpha, Ap, r, A.isWaxpbyOptimized);
    TOCK(t2); // r = r - alpha*Ap
    TICK();
    ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized);
    TOCK(t1);
#endif
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = " << k
                << "   Scaled Residual = " << normr / normr0 << std::endl;
#endif
    niters = k;
  }

  // free(ellVal);
  // free(rVal);
  // free(zVal);
  // free(pVal);
  // free(ApVal);

  // Store times
  times[1] += t1;                  // dot-product time
  times[2] += t2;                  // WAXPBY time
  times[3] += t3;                  // SPMV time
  times[4] += t4;                  // AllReduce time
  times[5] += t5;                  // preconditioner apply time
                                   //#ifndef HPCG_NO_MPI
                                   //  times[6] += t6; // exchange halo time
                                   //#endif
  times[0] += mytimer() - t_begin; // Total time. All done...
  return 0;
}
