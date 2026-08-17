#ifndef GPL_WORKLOAD_H
#define GPL_WORKLOAD_H

#include <stdbool.h>
#include <stdint.h>

#include "args.h"
#include "telemetry.h"

/* Workload entry points are defined in a mix of .c and .cu files and are
 * called from main.c, which is C. Force C linkage on both sides. */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Set by caller. */
    const gpl_args_t *args;
    gpl_telemetry_t  *tele;

    /* Populated by workload. */
    uint64_t iterations_warmup;
    uint64_t iterations_steady;
    double   compute_seconds_steady;  /* wall time inside steady loop, seconds */

    /* Optional: aborted early because --stop-on-throttle tripped */
    bool     aborted_on_throttle;

    /* Error info. */
    const char *error;
} gpl_workload_ctx_t;

/* Run a GEMM workload driven by ctx->args, sampling via ctx->tele.
 * Returns 0 on success, non-zero on CUDA/cuBLAS error. */
int gpl_workload_gemm_run(gpl_workload_ctx_t *ctx);

/* STREAM-triad memory-bandwidth workload (FP32, always). */
int gpl_workload_memstream_run(gpl_workload_ctx_t *ctx);

/* Batched 1-D cuFFT C2C workload (FP32, always). */
int gpl_workload_fft_run(gpl_workload_ctx_t *ctx);

/* Sample-only: drives no GPU work, just runs the phases for the requested
 * duration. For measuring an external process (training, inference) or the
 * idle floor. */
int gpl_workload_observe_run(gpl_workload_ctx_t *ctx);

/* Mixed-unit maximum-power workload (O1). Warp-specialized: tensor cores,
 * FFMA chains and DRAM streaming run concurrently on the same SM, in the
 * ratio given by the --mix-* weights. Honours --duty-* and --iters. */
int gpl_workload_powervirus_run(gpl_workload_ctx_t *ctx);

/* FLOPs per iteration for this rung.
 * Returns 2*M*N*K for SGEMM; 0.0 for bandwidth/FFT ops (use
 * gpl_workload_bytes_per_iter for those). */
double gpl_workload_flops_per_iter(const gpl_args_t *a);

/* Bytes moved per iteration for bandwidth-bound and FFT ops.
 * Returns 0.0 for SGEMM (TFLOPS is the primary metric there).
 * See workload_util.c for the exact formulae. */
double gpl_workload_bytes_per_iter(const gpl_args_t *a);

#ifdef __cplusplus
}
#endif

#endif
