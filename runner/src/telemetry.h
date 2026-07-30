#ifndef GPL_TELEMETRY_H
#define GPL_TELEMETRY_H

#include <nvml.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "util.h"

typedef enum {
    GPL_PHASE_IDLE = 0,
    GPL_PHASE_WARMUP,
    GPL_PHASE_STEADY,
    GPL_PHASE_STOP,
} gpl_phase_t;

typedef struct {
    /* Per-phase aggregates (STEADY only used in summary). */
    gpl_dvec  power_w;
    gpl_dvec  temp_c;
    gpl_dvec  sm_mhz;
    gpl_dvec  mem_mhz;
    gpl_dvec  sm_util;
    gpl_dvec  mem_util;
    uint64_t  throttle_mask_or;    /* bitwise OR of throttle reasons across steady window */
    double    throttled_sec;       /* total steady-phase seconds where any throttle bit was set */
    uint64_t  energy_start_mj;
    uint64_t  energy_end_mj;
    bool      energy_valid;
    uint64_t  samples;
} gpl_phase_stats_t;

typedef struct {
    /* Config */
    nvmlDevice_t device;
    int          sample_hz;
    FILE        *ndjson;           /* NULL = don't write per-sample */
    const char  *rung_id;

    /* Shared state */
    _Atomic gpl_phase_t phase;
    _Atomic bool        throttle_seen_steady;

    /* Thread */
    pthread_t thread;
    bool      thread_started;

    /* Stats — only touched by sampler thread until it joins. */
    gpl_phase_stats_t warmup;
    gpl_phase_stats_t steady;
} gpl_telemetry_t;

/* Start the sampler thread. Phase begins at IDLE — call gpl_telemetry_set_phase
 * to move to WARMUP / STEADY / STOP. */
int  gpl_telemetry_start(gpl_telemetry_t *t, nvmlDevice_t dev, int sample_hz,
                         FILE *ndjson_out, const char *rung_id);

void gpl_telemetry_set_phase(gpl_telemetry_t *t, gpl_phase_t p);

/* Join and free. Aggregates are usable after this returns. */
void gpl_telemetry_stop(gpl_telemetry_t *t);

/* Introspection — safe to call from workload thread. */
bool gpl_telemetry_throttled_now(gpl_telemetry_t *t);

/* Translate an NVML throttle bitmask to an array of stable string names.
 * Returns count written to `out_names`, up to max_names. */
int  gpl_throttle_reason_names(uint64_t mask, const char **out_names, int max_names);

#endif
