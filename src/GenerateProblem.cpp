#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "GenerateProblem.hpp"
#include "GenerateProblem_ref.hpp"

void GenerateProblem(SparseMatrix &A, Vector *b, Vector *x, Vector *xexact) {

  // The call to this reference version of GenerateProblem can be replaced with
  // custom code. However, the data structures must remain unchanged such that
  // the CheckProblem function is satisfied. Furthermore, any code must work for
  // general unstructured sparse matrices.  Special knowledge about the specific
  // nature of the sparsity pattern may not be explicitly used.

  return (GenerateProblem_ref(A, b, x, xexact));
}
