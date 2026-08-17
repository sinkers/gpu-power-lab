#ifndef GPL_ARGS_H
#define GPL_ARGS_H

#include <stdbool.h>

/* Included from .cu files, which nvcc compiles as C++, but defined in
 * args.c as C11. */
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GPL_OP_SGEMM = 1,
    GPL_OP_FFT,
    GPL_OP_MEMSTREAM,
    GPL_OP_POWERVIRUS,
} gpl_op_t;

typedef enum {
    GPL_PREC_FP32 = 1,
    GPL_PREC_TF32,
    GPL_PREC_FP16,
    GPL_PREC_BF16,
    GPL_PREC_FP8,
} gpl_prec_t;

typedef struct {
    gpl_op_t   op;
    gpl_prec_t precision;
    int        size;
    int        streams;
    double     warmup_sec;
    double     steady_sec;
    int        sample_hz;
    int        device;
    const char *out_metrics;   /* NDJSON path, or NULL to skip */
    const char *out_summary;   /* summary JSON path, or NULL for stdout */
    const char *rung_id;       /* optional rung id, generated if NULL */
    bool       stop_on_throttle;

    /* Tensor role implementation for powervirus.
     *   wmma   — warp-synchronous mma.sync in our own kernel. Occupies
     *            warp slots, so on any architecture it competes with the
     *            FMA and DRAM roles for the same budget.
     *   cublas — a concurrent cuBLAS GEMM stream. On Blackwell this is how
     *            we reach tcgen05, which is async and single-thread-issued,
     *            and therefore the only way to test whether tensor work can
     *            overlap the other roles instead of displacing them.
     * This distinction IS the O1 experiment on Blackwell. */
    bool       tensor_cublas;

    /* --- O1: powervirus mix weights (relative, 0 disables that unit) --- */
    int        mix_tensor;
    int        mix_fma;
    int        mix_dram;

    /* --- O2: duty-cycle modulation. on_ms <= 0 disables. --- */
    double     duty_on_ms;
    double     duty_off_ms;

    /* --- Fixed-work mode: >0 means run exactly this many steady
     * iterations and measure elapsed time, instead of running to the
     * steady deadline. Required for meaningful EDP / EDPp. --- */
    long long  iters;

    /* --- Platform control --- */
    bool       raise_power_limit;  /* set enforced limit to device max */
    bool       probe;              /* capability probe, then exit */
} gpl_args_t;

/* Fill args with defaults. */
void gpl_args_defaults(gpl_args_t *a);

/* Parse argv into args. Returns 0 on success, non-zero on error or --help.
 * On --help, prints usage and returns 1 (caller should exit 0). */
int gpl_args_parse(int argc, char **argv, gpl_args_t *a);

const char *gpl_op_name(gpl_op_t op);
const char *gpl_prec_name(gpl_prec_t p);

#ifdef __cplusplus
}
#endif

#endif
