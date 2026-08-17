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
    GPL_OP_OBSERVE,
    GPL_OP_NCCL,
} gpl_op_t;

/* Declared at file scope, not inside gpl_args_t: an unnamed enum nested in a
 * struct scopes its enumerators to that struct under C++, which would make
 * GPL_RAMP_NONE unreachable from the .cu translation units. */
typedef enum {
    GPL_RAMP_NONE = 0,
    GPL_RAMP_LINEAR,
    GPL_RAMP_EXP,
} gpl_ramp_t;

typedef enum {
    GPL_PREC_FP32 = 1,
    GPL_PREC_TF32,
    GPL_PREC_FP16,
    GPL_PREC_BF16,
    GPL_PREC_FP8,
    GPL_PREC_FP64,
    GPL_PREC_INT8,
    GPL_PREC_FP4,
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
    /* The rest of the attribution ladder: each isolates one more unit, so
     * its marginal watts can be read off and pairs can expose the
     * interaction term. */
    int        mix_sfu;      /* transcendentals */
    int        mix_int32;    /* integer pipe */
    int        mix_smem;     /* shared memory / L1 */
    int        mix_l2;       /* L2-resident streaming */
    int        mix_atomic;   /* atomics, resolved in L2 */

    /* --- O2: duty-cycle modulation. on_ms <= 0 disables. --- */
    double     duty_on_ms;
    double     duty_off_ms;
    /* Edge shape. A square wave is the worst case for upstream power
     * delivery; a ramped edge is what a considerate scheduler would do
     * instead. Measuring both says how much of the transient hazard is
     * inherent and how much is just an implementation choice. */
    gpl_ramp_t duty_ramp;
    double     duty_ramp_ms;   /* edge duration; 0 = a quarter of on-time */

    /* --- Fixed-work mode: >0 means run exactly this many steady
     * iterations and measure elapsed time, instead of running to the
     * steady deadline. Required for meaningful EDP / EDPp. --- */
    long long  iters;

    /* --- Platform control --- */
    bool       raise_power_limit;  /* set enforced limit to device max */
    /* Set the enforced limit to an explicit value in watts before measuring.
     * This is what makes a cap sweep possible: on a part whose default limit
     * is already its maximum there is nothing to raise into, but there is
     * plenty of room below - and the EDP-vs-cap curve, the throughput cost
     * of capping, and the chance of catching an over-limit excursion all
     * live down there. 0 = leave alone. */
    double     power_limit_w;
    bool       lock_clocks;        /* pin SM clock (boost-off profile) */
    bool       probe;              /* capability probe, then exit */

    /* --- multi-GPU: what the fabric costs, and what a collective does to
     * the shape of a node's power draw. A collective is a synchronised event
     * across every GPU at once, which is the in-phase swing the duty-cycle
     * work was trying to synthesise - real training already produces it. */
    const char *nccl_op;           /* allreduce|allgather|reducescatter|
                                      broadcast|alltoall|sendrecv */
    long long   nccl_bytes;        /* message size per rank */
    int         nccl_devices;      /* 0 = every visible GPU */
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
