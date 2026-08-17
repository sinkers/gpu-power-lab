#ifndef GPL_GEMM_LT_H
#define GPL_GEMM_LT_H

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#include "args.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A cuBLASLt GEMM held open across many enqueues. Descriptors, layouts and
 * the chosen algorithm are set up once; the steady loop only calls
 * cublasLtMatmul, so per-call overhead stays out of the measurement. */
typedef struct {
    cublasLtHandle_t            lt;
    cublasLtMatmulDesc_t        op_desc;
    cublasLtMatrixLayout_t      layout_a, layout_b, layout_c, layout_d;
    cublasLtMatmulPreference_t  pref;
    cublasLtMatmulHeuristicResult_t heuristic;

    void   *dA, *dB, *dC, *dD;
    void   *dScaleA, *dScaleB, *dScaleD;
    void   *workspace;
    size_t  workspace_bytes;

    cudaStream_t stream;
    int          n;
    gpl_prec_t   prec;
    bool         ready;
} gpl_lt_gemm_t;

/* True for the formats that only exist behind cublasLtMatmul. */
bool gpl_lt_precision_supported(gpl_prec_t p);

/* Returns 0 on success. A non-zero return with a logged "no algorithm
 * available" is the normal outcome on a part that lacks the format — the
 * caller should treat it as "not supported here", not as a crash. */
int  gpl_lt_init(gpl_lt_gemm_t *g, gpl_prec_t prec, int n, cudaStream_t stream);
int  gpl_lt_enqueue(gpl_lt_gemm_t *g, int count);
void gpl_lt_free(gpl_lt_gemm_t *g);

#ifdef __cplusplus
}
#endif

#endif
