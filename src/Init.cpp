#include "../cuda-src/InitInside.cuh"

int HPCG_Init(int *argc_p, char ***argv_p, HPCG_Params &params) {
  return HPCG_InitInside(argc_p, argv_p, params);
}
