#ifndef GPL_PROBE_H
#define GPL_PROBE_H

#include <nvml.h>
#include <stdbool.h>
#include <stdio.h>

#include "util.h"

/* What this platform actually permits. See probe.c for why this exists. */
typedef struct {
    bool   power_usage;
    bool   power_instant;
    bool   power_average;
    bool   power_samples;
    unsigned int power_samples_count;
    bool   energy_counter;
    bool   violation_status;

    bool   limit_constraints;
    double limit_current_w;
    double limit_default_w;
    double limit_min_w;
    double limit_max_w;
    bool   limit_write;

    bool   persistence_on;
    bool   persistence_write;
    bool   clock_read;
    bool   clock_lock;
    bool   sched_rr;
    bool   mig_enabled;

    bool   can_measure_o1;
    bool   can_measure_o2;
} gpl_probe_t;

/* Run the probe, write JSON to `out`, log a verdict to stderr.
 * Returns 0 if O1 is measurable here, 7 if not. */
int gpl_probe_run(nvmlDevice_t dev, FILE *out);

#endif
