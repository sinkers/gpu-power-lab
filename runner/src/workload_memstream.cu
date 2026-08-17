/*
 * workload_memstream.cu — STREAM-triad memory-bandwidth workload.
 *
 * Kernel: C[i] = A[i] + scalar * B[i]  (FP32, always — precision arg ignored)
 *
 * Working set: treat args.size as a square-matrix side length for consistency
 * with GEMM.  Each array (A, B, C) holds size×size float32 elements.
 *   bytes_per_array = size * size * sizeof(float)
 *
 * A and B are shared read-only across all streams; each stream has its own
 * C output buffer so concurrent writes do not alias.
 *
 * Iterations counted as individual kernel dispatches (one per stream per
 * outer loop pass).  GB/s is computed in summary.c via
 * gpl_workload_bytes_per_iter() × iterations / seconds.
 */

#include "workload.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

/* ------------------------------------------------------------------ */
/* Error-handling macros — identical style to workload_gemm.c         */
/* ------------------------------------------------------------------ */

#define CUDA_OK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        ctx->error = cudaGetErrorString(_e); \
        gpl_errf("CUDA: %s (at %s:%d)", ctx->error, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/* Device kernel                                                       */
/* ------------------------------------------------------------------ */

static __global__ void stream_triad_kernel(
        float * __restrict__       C,
        const float * __restrict__ A,
        const float * __restrict__ B,
        float  scalar,
        size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + scalar * B[idx];
    }
}

/* ------------------------------------------------------------------ */
/* Public interface — extern "C" so main.c (plain C) can call us      */
/* ------------------------------------------------------------------ */

int gpl_workload_memstream_run(gpl_workload_ctx_t *ctx)
{
    const gpl_args_t *a      = ctx->args;
    const int         ns     = a->streams;

    /* Working set sizes. */
    const size_t n_elems = (size_t)a->size * (size_t)a->size;
    const size_t n_bytes = n_elems * sizeof(float);

    /* Allocate shared A/B (read-only pattern) and per-stream C. */
    float  *dA   = NULL;
    float  *dB   = NULL;
    float **dC   = (float **)calloc(ns, sizeof(float *));
    if (!dC) { ctx->error = "oom (dC array)"; return -1; }

    CUDA_OK(cudaMalloc((void **)&dA, n_bytes));
    CUDA_OK(cudaMalloc((void **)&dB, n_bytes));
    /* Match the 0x3c memset pattern used in workload_gemm.c. */
    CUDA_OK(cudaMemset(dA, 0x3c, n_bytes));
    CUDA_OK(cudaMemset(dB, 0x3c, n_bytes));

    for (int i = 0; i < ns; i++) {
        CUDA_OK(cudaMalloc((void **)&dC[i], n_bytes));
        CUDA_OK(cudaMemset(dC[i], 0, n_bytes));
    }

    /* One CUDA stream per channel. */
    cudaStream_t *streams = (cudaStream_t *)calloc(ns, sizeof(cudaStream_t));
    if (!streams) { ctx->error = "oom (streams)"; return -1; }
    for (int i = 0; i < ns; i++) {
        CUDA_OK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    /* Launch geometry. */
    const int          threads = 256;
    const unsigned int blocks  =
        (unsigned int)((n_elems + (size_t)threads - 1) / (size_t)threads);
    const float        scalar  = 0.5f;

    /* One priming dispatch per stream to flush CUDA JIT / pipeline setup
     * before the warmup clock starts — same pattern as workload_gemm.c. */
    for (int i = 0; i < ns; i++) {
        stream_triad_kernel<<<blocks, threads, 0, streams[i]>>>(
            dC[i], dA, dB, scalar, n_elems);
    }
    CUDA_OK(cudaDeviceSynchronize());

    /* ----------------------------------------------------------------
     * WARMUP
     * ---------------------------------------------------------------- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_WARMUP);
    uint64_t warm_start    = gpl_mono_ns();
    uint64_t warm_deadline = warm_start + (uint64_t)(a->warmup_sec * 1e9);
    uint64_t warm_iters    = 0;

    while (gpl_mono_ns() < warm_deadline) {
        for (int i = 0; i < ns; i++) {
            stream_triad_kernel<<<blocks, threads, 0, streams[i]>>>(
                dC[i], dA, dB, scalar, n_elems);
            warm_iters++;
        }
        /* Periodic sync to bound in-flight work — mirrors GEMM approach. */
        if ((warm_iters % (uint64_t)(ns * 8)) == 0) {
            CUDA_OK(cudaDeviceSynchronize());
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    ctx->iterations_warmup = warm_iters;

    /* ----------------------------------------------------------------
     * STEADY
     * ---------------------------------------------------------------- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_STEADY);
    uint64_t steady_start    = gpl_mono_ns();
    uint64_t steady_deadline = steady_start + (uint64_t)(a->steady_sec * 1e9);
    uint64_t steady_iters    = 0;
    bool     aborted         = false;

    while (gpl_mono_ns() < steady_deadline) {
        for (int i = 0; i < ns; i++) {
            stream_triad_kernel<<<blocks, threads, 0, streams[i]>>>(
                dC[i], dA, dB, scalar, n_elems);
            steady_iters++;
        }
        if ((steady_iters % (uint64_t)(ns * 8)) == 0) {
            CUDA_OK(cudaDeviceSynchronize());
            if (a->stop_on_throttle && gpl_telemetry_throttled_now(ctx->tele)) {
                aborted = true;
                break;
            }
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    uint64_t steady_end = gpl_mono_ns();

    ctx->iterations_steady      = steady_iters;
    ctx->compute_seconds_steady = (double)(steady_end - steady_start) / 1e9;
    ctx->aborted_on_throttle    = aborted;

    /* Cleanup */
    for (int i = 0; i < ns; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(dC[i]);
    }
    cudaFree(dA);
    cudaFree(dB);
    free(dC);
    free(streams);

    return 0;
}
