/*
 * workload_fft.c — batched 1-D cuFFT C2C (complex-to-complex) workload.
 *
 * Plan: forward 1D FFT of length args.size, batched as
 *   batch = max(1, (1 << 20) / args.size)
 * so that the total working-set per stream stays roughly constant
 * (~8 MB input + ~8 MB output) across sizes from 64 to ~1 M.
 *
 * One cufftHandle per stream (cuFFT plans are not safely shared across
 * concurrent streams).  All plans are CUFFT_C2C (single-precision complex).
 * The precision arg is ignored for v0.1 (FP32 always).
 *
 * Iterations are counted as individual cufftExecC2C calls (one per stream
 * per outer loop pass).  Throughput in GFFT/s is computed in summary.c.
 */

#include "workload.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

/* ------------------------------------------------------------------ */
/* Error-handling macros — mirror style of workload_gemm.c            */
/* ------------------------------------------------------------------ */

#define CUDA_OK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        ctx->error = cudaGetErrorString(_e); \
        gpl_errf("CUDA: %s (at %s:%d)", ctx->error, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

#define CUFFT_OK(x) do { \
    cufftResult _r = (x); \
    if (_r != CUFFT_SUCCESS) { \
        ctx->error = "cufft failure"; \
        gpl_errf("cuFFT result %d at %s:%d", (int)_r, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/* Helper                                                              */
/* ------------------------------------------------------------------ */

static int fft_batch(int fft_len) {
    if (fft_len <= 0) return 1;
    int b = (1 << 20) / fft_len;
    return (b < 1) ? 1 : b;
}

/* ------------------------------------------------------------------ */
/* Workload implementation                                             */
/* ------------------------------------------------------------------ */

int gpl_workload_fft_run(gpl_workload_ctx_t *ctx)
{
    const gpl_args_t *a   = ctx->args;
    const int         ns  = a->streams;
    const int         fft_len = a->size;
    const int         batch   = fft_batch(fft_len);
    const size_t      n_elem  = (size_t)fft_len * (size_t)batch;
    const size_t      n_bytes = n_elem * sizeof(cufftComplex);

    gpl_logf("fft: size=%d batch=%d elem_per_stream=%zu bytes_per_stream=%zu streams=%d",
             fft_len, batch, n_elem, n_bytes, ns);

    /* Per-stream allocations. */
    cufftComplex **dIn   = (cufftComplex **)calloc(ns, sizeof(cufftComplex *));
    cufftComplex **dOut  = (cufftComplex **)calloc(ns, sizeof(cufftComplex *));
    cufftHandle   *plans = (cufftHandle *)  calloc(ns, sizeof(cufftHandle));
    cudaStream_t  *streams = (cudaStream_t *)calloc(ns, sizeof(cudaStream_t));

    if (!dIn || !dOut || !plans || !streams) {
        ctx->error = "oom (fft bookkeeping arrays)";
        free(dIn); free(dOut); free(plans); free(streams);
        return -1;
    }

    /* Zero-initialise so the cleanup path can safely skip uninitialised
     * entries if an error fires mid-loop. */
    memset(plans,   0, (size_t)ns * sizeof(cufftHandle));
    memset(streams, 0, (size_t)ns * sizeof(cudaStream_t));

    for (int i = 0; i < ns; i++) {
        CUDA_OK(cudaMalloc((void **)&dIn[i],  n_bytes));
        CUDA_OK(cudaMalloc((void **)&dOut[i], n_bytes));
        CUDA_OK(cudaMemset(dIn[i], 0, n_bytes));

        CUDA_OK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        /* cufftPlan1d(plan, nx, type, batch) — one plan per stream so
         * concurrent dispatches don't race on internal state. */
        CUFFT_OK(cufftPlan1d(&plans[i], fft_len, CUFFT_C2C, batch));
        CUFFT_OK(cufftSetStream(plans[i], streams[i]));
    }

    /* Prime: one execution per stream to trigger cuFFT's internal
     * JIT / kernel-selection before the warmup clock starts. */
    for (int i = 0; i < ns; i++) {
        CUFFT_OK(cufftExecC2C(plans[i], dIn[i], dOut[i], CUFFT_FORWARD));
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
            CUFFT_OK(cufftExecC2C(plans[i], dIn[i], dOut[i], CUFFT_FORWARD));
            warm_iters++;
        }
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
            CUFFT_OK(cufftExecC2C(plans[i], dIn[i], dOut[i], CUFFT_FORWARD));
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
        if (plans[i])   cufftDestroy(plans[i]);
        if (streams[i]) cudaStreamDestroy(streams[i]);
        if (dIn[i])     cudaFree(dIn[i]);
        if (dOut[i])    cudaFree(dOut[i]);
    }
    free(dIn); free(dOut); free(plans); free(streams);

    return 0;
}
