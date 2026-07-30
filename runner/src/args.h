#ifndef GPL_ARGS_H
#define GPL_ARGS_H

#include <stdbool.h>

typedef enum {
    GPL_OP_SGEMM = 1,
    GPL_OP_FFT,
    GPL_OP_MEMSTREAM,
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
} gpl_args_t;

/* Fill args with defaults. */
void gpl_args_defaults(gpl_args_t *a);

/* Parse argv into args. Returns 0 on success, non-zero on error or --help.
 * On --help, prints usage and returns 1 (caller should exit 0). */
int gpl_args_parse(int argc, char **argv, gpl_args_t *a);

const char *gpl_op_name(gpl_op_t op);
const char *gpl_prec_name(gpl_prec_t p);

#endif
