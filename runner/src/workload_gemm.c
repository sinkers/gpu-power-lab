#include "workload.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#define CUDA_OK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        ctx->error = cudaGetErrorString(_e); \
        gpl_errf("CUDA: %s (at %s:%d)", ctx->error, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

#define CUBLAS_OK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        ctx->error = "cublas failure"; \
        gpl_errf("cuBLAS status %d at %s:%d", (int)_s, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

static size_t elem_bytes(gpl_prec_t p) {
    switch (p) {
        case GPL_PREC_FP32:
        case GPL_PREC_TF32: return 4;
        case GPL_PREC_FP16:
        case GPL_PREC_BF16: return 2;
        case GPL_PREC_FP8:  return 1;
        default:            return 4;
    }
}

/* Map our precision to cuBLAS types for cublasGemmEx. */
static void prec_to_cublas(gpl_prec_t p,
                           cudaDataType *inT, cudaDataType *outT,
                           cublasComputeType_t *cT) {
    switch (p) {
        case GPL_PREC_FP32:
            *inT = CUDA_R_32F; *outT = CUDA_R_32F; *cT = CUBLAS_COMPUTE_32F;
            break;
        case GPL_PREC_TF32:
            *inT = CUDA_R_32F; *outT = CUDA_R_32F; *cT = CUBLAS_COMPUTE_32F_FAST_TF32;
            break;
        case GPL_PREC_FP16:
            *inT = CUDA_R_16F; *outT = CUDA_R_16F; *cT = CUBLAS_COMPUTE_16F;
            break;
        case GPL_PREC_BF16:
            *inT = CUDA_R_16BF; *outT = CUDA_R_16BF; *cT = CUBLAS_COMPUTE_32F;
            break;
        case GPL_PREC_FP8:
            /* FP8 requires cublasLtMatmul; not supported in this v0.1 path. */
        default:
            *inT = CUDA_R_32F; *outT = CUDA_R_32F; *cT = CUBLAS_COMPUTE_32F;
            break;
    }
}

int gpl_workload_gemm_run(gpl_workload_ctx_t *ctx) {
    const gpl_args_t *a = ctx->args;

    if (a->precision == GPL_PREC_FP8) {
        ctx->error = "fp8 not implemented in v0.1 (requires cublasLtMatmul)";
        gpl_errf("%s", ctx->error);
        return -1;
    }

    const int N = a->size;
    const int nstreams = a->streams;
    const size_t eb = elem_bytes(a->precision);
    const size_t bytes = (size_t)N * (size_t)N * eb;

    cudaDataType inT, outT;
    cublasComputeType_t cT;
    prec_to_cublas(a->precision, &inT, &outT, &cT);

    /* Allocate A, B, and one C per stream so parallel streams don't clobber. */
    void *dA = NULL, *dB = NULL;
    void **dC = (void **)calloc(nstreams, sizeof(void *));
    if (!dC) { ctx->error = "oom (dC)"; return -1; }

    CUDA_OK(cudaMalloc(&dA, bytes));
    CUDA_OK(cudaMalloc(&dB, bytes));
    CUDA_OK(cudaMemset(dA, 0x3c, bytes));   /* fp16 ~1.0-ish; harmless bit pattern */
    CUDA_OK(cudaMemset(dB, 0x3c, bytes));
    for (int i = 0; i < nstreams; i++) {
        CUDA_OK(cudaMalloc(&dC[i], bytes));
        CUDA_OK(cudaMemsetAsync(dC[i], 0, bytes, 0));
    }

    cudaStream_t *streams = (cudaStream_t *)calloc(nstreams, sizeof(cudaStream_t));
    if (!streams) { ctx->error = "oom (streams)"; return -1; }
    for (int i = 0; i < nstreams; i++) {
        CUDA_OK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    cublasHandle_t *h = (cublasHandle_t *)calloc(nstreams, sizeof(cublasHandle_t));
    if (!h) { ctx->error = "oom (handles)"; return -1; }
    for (int i = 0; i < nstreams; i++) {
        CUBLAS_OK(cublasCreate(&h[i]));
        CUBLAS_OK(cublasSetStream(h[i], streams[i]));
        if (a->precision == GPL_PREC_TF32) {
            cublasSetMathMode(h[i], CUBLAS_TF32_TENSOR_OP_MATH);
        } else {
            cublasSetMathMode(h[i], CUBLAS_DEFAULT_MATH);
        }
    }

    /* Alpha / beta are always float32 host scalars for these compute types. */
    float alpha_f = 1.0f, beta_f = 0.0f;
    /* CUBLAS_COMPUTE_16F wants half-precision host scalars. `__half` and
     * `__float2half` are C++-only (cuda_fp16.h guards them), and this file
     * is compiled as C11, so use the IEEE-754 binary16 bit patterns
     * directly: 0x3C00 is 1.0, 0x0000 is 0.0. cuBLAS only reads the 16
     * bits, and this keeps the file out of nvcc. */
    unsigned short alpha_h = 0x3C00, beta_h = 0x0000;
    void *alpha_p, *beta_p;
    if (cT == CUBLAS_COMPUTE_16F) { alpha_p = &alpha_h; beta_p = &beta_h; }
    else                          { alpha_p = &alpha_f; beta_p = &beta_f; }

    /* Prime the pipes so first-call overhead isn't billed to warmup. */
    for (int i = 0; i < nstreams; i++) {
        CUBLAS_OK(cublasGemmEx(h[i], CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N, N,
                               alpha_p,
                               dA, inT, N,
                               dB, inT, N,
                               beta_p,
                               dC[i], outT, N,
                               cT,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_OK(cudaDeviceSynchronize());

    /* ---------- WARMUP ---------- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_WARMUP);
    uint64_t warm_start = gpl_mono_ns();
    uint64_t warm_deadline = warm_start + (uint64_t)(a->warmup_sec * 1e9);
    uint64_t warm_iters = 0;
    while (gpl_mono_ns() < warm_deadline) {
        for (int i = 0; i < nstreams; i++) {
            CUBLAS_OK(cublasGemmEx(h[i], CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, N, N,
                                   alpha_p, dA, inT, N,
                                            dB, inT, N,
                                   beta_p,  dC[i], outT, N,
                                   cT, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            warm_iters++;
        }
        /* Don't sync every batch — let the GPU actually saturate. Sync
         * periodically to bound in-flight work. */
        if ((warm_iters % (nstreams * 8)) == 0) {
            CUDA_OK(cudaDeviceSynchronize());
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    ctx->iterations_warmup = warm_iters;

    /* ---------- STEADY ---------- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_STEADY);
    uint64_t steady_start = gpl_mono_ns();
    uint64_t steady_deadline = steady_start + (uint64_t)(a->steady_sec * 1e9);
    uint64_t steady_iters = 0;
    bool aborted = false;
    while (gpl_mono_ns() < steady_deadline) {
        for (int i = 0; i < nstreams; i++) {
            CUBLAS_OK(cublasGemmEx(h[i], CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, N, N,
                                   alpha_p, dA, inT, N,
                                            dB, inT, N,
                                   beta_p,  dC[i], outT, N,
                                   cT, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            steady_iters++;
        }
        if ((steady_iters % (nstreams * 8)) == 0) {
            CUDA_OK(cudaDeviceSynchronize());
            if (a->stop_on_throttle && gpl_telemetry_throttled_now(ctx->tele)) {
                aborted = true;
                break;
            }
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    uint64_t steady_end = gpl_mono_ns();
    ctx->iterations_steady = steady_iters;
    ctx->compute_seconds_steady = (double)(steady_end - steady_start) / 1e9;
    ctx->aborted_on_throttle = aborted;

    /* Cleanup */
    for (int i = 0; i < nstreams; i++) {
        cublasDestroy(h[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(dC[i]);
    }
    cudaFree(dA); cudaFree(dB);
    free(dC); free(streams); free(h);

    return 0;
}
