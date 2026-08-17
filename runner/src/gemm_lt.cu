/*
 * cuBLASLt GEMM — the FP8 and FP4 path.
 *
 * `cublasGemmEx` cannot express the narrow formats at all: FP8 needs
 * per-tensor scale factors and FP4 needs block scaling, neither of which the
 * legacy entry point has arguments for. Both live behind cublasLtMatmul.
 *
 * Why this matters here rather than being a completeness exercise: the B300
 * ladder topped out at 90.4% of 1100 W with bf16, never throttled, so the
 * missing ~105 W is workload shortfall. The narrow formats move the most
 * arithmetic per unit of operand traffic of anything on the chip, and are the
 * most plausible way to close it. They are also the interesting case for the
 * watts-per-flop question: the fastest format is not necessarily the
 * hungriest, because a format that finishes its operands sooner spends the
 * difference waiting on memory.
 *
 * Layout constraint worth knowing before debugging this: FP8 matmul requires
 * the A operand transposed (`CUBLAS_OP_T`) with both operands in the same
 * leading-dimension convention. Getting it wrong returns
 * CUBLAS_STATUS_NOT_SUPPORTED rather than a wrong answer, which at least
 * fails loudly.
 */

#include "gemm_lt.h"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include "util.h"

#define LT_OK(x, what) do {                                            \
    cublasStatus_t _s = (x);                                           \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
        gpl_errf("cublasLt %s failed: status %d", (what), (int)_s);    \
        return -1;                                                     \
    }                                                                  \
} while (0)

#define CU_OK(x, what) do {                                            \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                           \
        gpl_errf("cuda %s: %s", (what), cudaGetErrorString(_e));       \
        return -1;                                                     \
    }                                                                  \
} while (0)

bool gpl_lt_precision_supported(gpl_prec_t p) {
#if defined(CUBLAS_VERSION) && (CUBLAS_VERSION >= 120800)
    return p == GPL_PREC_FP8 || p == GPL_PREC_FP4;
#else
    return p == GPL_PREC_FP8;
#endif
}

int gpl_lt_init(gpl_lt_gemm_t *g, gpl_prec_t prec, int n, cudaStream_t stream) {
    memset(g, 0, sizeof(*g));
    g->n = n;
    g->prec = prec;
    g->stream = stream;

    cudaDataType_t in_type, out_type, scale_type;
    cublasComputeType_t compute_type;

    switch (prec) {
        case GPL_PREC_FP8:
            /* e4m3 for both operands: the format used for weights and
             * activations in practice. e5m2 exists for gradients and is not
             * interesting for a power ceiling. */
            in_type = CUDA_R_8F_E4M3;
            out_type = CUDA_R_16BF;    /* accumulate out to bf16 */
            scale_type = CUDA_R_32F;
            compute_type = CUBLAS_COMPUTE_32F;
            break;
#if defined(CUBLAS_VERSION) && (CUBLAS_VERSION >= 120800)
        case GPL_PREC_FP4:
            in_type = CUDA_R_4F_E2M1;
            out_type = CUDA_R_16BF;
            scale_type = CUDA_R_32F;
            compute_type = CUBLAS_COMPUTE_32F;
            break;
#endif
        default:
            gpl_errf("gemm_lt: precision %s is not an Lt format", gpl_prec_name(prec));
            return -1;
    }

    LT_OK(cublasLtCreate(&g->lt), "create");

    size_t a_elem = (prec == GPL_PREC_FP8) ? 1 : 1;   /* FP4 packs 2/byte; see below */
    size_t a_bytes = (size_t)n * n * a_elem;
    if (prec != GPL_PREC_FP8) a_bytes = ((size_t)n * n + 1) / 2;
    size_t c_bytes = (size_t)n * n * 2;               /* bf16 out */

    CU_OK(cudaMalloc(&g->dA, a_bytes), "malloc A");
    CU_OK(cudaMalloc(&g->dB, a_bytes), "malloc B");
    CU_OK(cudaMalloc(&g->dC, c_bytes), "malloc C");
    CU_OK(cudaMalloc(&g->dD, c_bytes), "malloc D");
    /* 0x38 in e4m3 is a small positive value; the arithmetic just has to be
     * unremarkable, not correct. */
    CU_OK(cudaMemset(g->dA, 0x38, a_bytes), "memset A");
    CU_OK(cudaMemset(g->dB, 0x38, a_bytes), "memset B");
    CU_OK(cudaMemset(g->dC, 0, c_bytes), "memset C");

    /* Per-tensor scales. Required for FP8 — omitting them is one of the ways
     * to get NOT_SUPPORTED back. */
    float one = 1.0f;
    CU_OK(cudaMalloc(&g->dScaleA, sizeof(float)), "malloc scaleA");
    CU_OK(cudaMalloc(&g->dScaleB, sizeof(float)), "malloc scaleB");
    CU_OK(cudaMalloc(&g->dScaleD, sizeof(float)), "malloc scaleD");
    CU_OK(cudaMemcpy(g->dScaleA, &one, sizeof(float), cudaMemcpyHostToDevice), "scaleA");
    CU_OK(cudaMemcpy(g->dScaleB, &one, sizeof(float), cudaMemcpyHostToDevice), "scaleB");
    CU_OK(cudaMemcpy(g->dScaleD, &one, sizeof(float), cudaMemcpyHostToDevice), "scaleD");

    LT_OK(cublasLtMatmulDescCreate(&g->op_desc, compute_type, scale_type), "descCreate");

    /* FP8 requires A transposed. */
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT_OK(cublasLtMatmulDescSetAttribute(g->op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &ta, sizeof(ta)), "transa");
    LT_OK(cublasLtMatmulDescSetAttribute(g->op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &tb, sizeof(tb)), "transb");
    LT_OK(cublasLtMatmulDescSetAttribute(g->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                         &g->dScaleA, sizeof(g->dScaleA)), "scaleA ptr");
    LT_OK(cublasLtMatmulDescSetAttribute(g->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                         &g->dScaleB, sizeof(g->dScaleB)), "scaleB ptr");

    LT_OK(cublasLtMatrixLayoutCreate(&g->layout_a, in_type, n, n, n), "layout A");
    LT_OK(cublasLtMatrixLayoutCreate(&g->layout_b, in_type, n, n, n), "layout B");
    LT_OK(cublasLtMatrixLayoutCreate(&g->layout_c, out_type, n, n, n), "layout C");
    LT_OK(cublasLtMatrixLayoutCreate(&g->layout_d, out_type, n, n, n), "layout D");

    /* Workspace: 32 MiB is the size NVIDIA's own samples use and is ample
     * for square GEMMs at these sizes. */
    g->workspace_bytes = 32u << 20;
    CU_OK(cudaMalloc(&g->workspace, g->workspace_bytes), "malloc workspace");

    LT_OK(cublasLtMatmulPreferenceCreate(&g->pref), "prefCreate");
    LT_OK(cublasLtMatmulPreferenceSetAttribute(
              g->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &g->workspace_bytes, sizeof(g->workspace_bytes)), "pref workspace");

    int returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        g->lt, g->op_desc, g->layout_a, g->layout_b, g->layout_c, g->layout_d,
        g->pref, 1, &g->heuristic, &returned);
    if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
        /* This is the expected failure when the format is not supported on
         * the part, or when the layout convention is wrong. Say which, since
         * the two need very different fixes. */
        gpl_errf("gemm_lt: no %s algorithm available on this device "
                 "(status %d, returned %d). Either the part lacks the format "
                 "or the layout is wrong.", gpl_prec_name(prec), (int)hs, returned);
        return -1;
    }

    g->ready = true;
    gpl_logf("gemm_lt: %s GEMM n=%d ready (workspace %zu MiB)",
             gpl_prec_name(prec), n, g->workspace_bytes >> 20);
    return 0;
}

int gpl_lt_enqueue(gpl_lt_gemm_t *g, int count) {
    if (!g->ready) return -1;
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < count; i++) {
        cublasStatus_t s = cublasLtMatmul(
            g->lt, g->op_desc,
            &alpha, g->dA, g->layout_a, g->dB, g->layout_b,
            &beta,  g->dC, g->layout_c, g->dD, g->layout_d,
            &g->heuristic.algo, g->workspace, g->workspace_bytes, g->stream);
        if (s != CUBLAS_STATUS_SUCCESS) {
            gpl_errf("cublasLtMatmul failed: status %d", (int)s);
            return -1;
        }
    }
    return 0;
}

void gpl_lt_free(gpl_lt_gemm_t *g) {
    if (!g->ready) return;
    cublasLtMatmulPreferenceDestroy(g->pref);
    cublasLtMatrixLayoutDestroy(g->layout_a);
    cublasLtMatrixLayoutDestroy(g->layout_b);
    cublasLtMatrixLayoutDestroy(g->layout_c);
    cublasLtMatrixLayoutDestroy(g->layout_d);
    cublasLtMatmulDescDestroy(g->op_desc);
    cublasLtDestroy(g->lt);
    cudaFree(g->dA); cudaFree(g->dB); cudaFree(g->dC); cudaFree(g->dD);
    cudaFree(g->dScaleA); cudaFree(g->dScaleB); cudaFree(g->dScaleD);
    cudaFree(g->workspace);
    g->ready = false;
}
