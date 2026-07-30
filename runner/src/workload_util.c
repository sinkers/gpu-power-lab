/*
 * workload_util.c — per-op metric helpers shared across workloads.
 *
 * gpl_workload_flops_per_iter was originally defined in workload_gemm.c;
 * it now lives here so it can handle all three op types and so that
 * gpl_workload_bytes_per_iter (for bandwidth-bound and FFT workloads)
 * lives next to it.
 *
 * IMPORTANT: both functions must stay in sync with the working-set
 * sizes chosen in the individual workload files.
 */

#include "workload.h"

/* ------------------------------------------------------------------ */
/* FLOPs per iteration                                                 */
/*   SGEMM  : 2 * N^3   (M=N=K square GEMM, one iteration = one GEMM) */
/*   Others : 0.0       (bandwidth- or latency-bound; use bytes below) */
/* ------------------------------------------------------------------ */
double gpl_workload_flops_per_iter(const gpl_args_t *a) {
    switch (a->op) {
        case GPL_OP_SGEMM: {
            double n = (double)a->size;
            return 2.0 * n * n * n;
        }
        case GPL_OP_MEMSTREAM:
        case GPL_OP_FFT:
        default:
            return 0.0;
    }
}

/* ------------------------------------------------------------------ */
/* Bytes moved per iteration                                           */
/*   MEMSTREAM : 3 arrays × size² × 4 B  (read A, read B, write C)   */
/*   FFT       : 2 × size × batch × 8 B  (cufftComplex in + out)     */
/*   SGEMM     : 0.0  (TFLOPS is the primary metric for GEMM)         */
/*                                                                     */
/* Batch for FFT: max(1, (1 << 20) / size) — must match workload_fft.c*/
/* ------------------------------------------------------------------ */
double gpl_workload_bytes_per_iter(const gpl_args_t *a) {
    switch (a->op) {
        case GPL_OP_MEMSTREAM: {
            /* A[size²] + B[size²] + C[size²], each float32 (4 bytes). */
            double n = (double)a->size;
            return 3.0 * n * n * 4.0;
        }
        case GPL_OP_FFT: {
            /* cufftComplex = 8 bytes; in-array + out-array each fft_len×batch. */
            int batch = (1 << 20) / a->size;
            if (batch < 1) batch = 1;
            return 2.0 * (double)a->size * (double)batch * 8.0;
        }
        case GPL_OP_SGEMM:
        default:
            return 0.0;
    }
}
