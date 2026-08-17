#ifndef GPL_TELEMETRY_H
#define GPL_TELEMETRY_H

#include <nvml.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "util.h"

/* This header is included from both C11 translation units and from .cu
 * files, which nvcc compiles as C++. C11 `_Atomic T` is not valid C++, so
 * spell the atomic members through a macro. std::atomic<T> and _Atomic T
 * are layout-compatible for these types on every ABI we target, which
 * matters because the .cu side sees the same struct. */
#ifdef __cplusplus
#include <atomic>
#define GPL_ATOMIC(T) std::atomic<T>
#else
#include <stdatomic.h>
#define GPL_ATOMIC(T) _Atomic T
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GPL_PHASE_IDLE = 0,
    GPL_PHASE_WARMUP,
    GPL_PHASE_STEADY,
    GPL_PHASE_STOP,
} gpl_phase_t;

typedef struct {
    /* Per-phase aggregates (STEADY only used in summary). */
    gpl_dvec  power_w;          /* nvmlDeviceGetPowerUsage — driver-averaged */
    gpl_dvec  power_instant_w;  /* NVML_FI_DEV_POWER_INSTANT, where available */
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

    /* O2 evidence. The energy counter is a monotonic hardware accumulator:
     * it cannot miss a spike the way a sampler can, so a gap between it and
     * the integral of the sampled curve means we are missing area. */
    double    power_integral_j;    /* sum(power_w * dt) over the phase */
    double    elapsed_s;           /* sum(dt) — gives the ACHIEVED sample rate */
    uint64_t  violation_us_start;
    uint64_t  violation_us_end;
    bool      violation_valid;
    double    peak_instant_w;
} gpl_phase_stats_t;

typedef struct {
    /* Config */
    nvmlDevice_t device;
    int          sample_hz;
    FILE        *ndjson;           /* NULL = don't write per-sample */
    const char  *rung_id;

    /* Capability flags, resolved once at start. */
    bool have_instant;
    bool have_violation;

    /* Shared state */
    GPL_ATOMIC(gpl_phase_t) phase;
    GPL_ATOMIC(bool)        throttle_seen_steady;

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

#ifdef __cplusplus
}
#endif

#endif
