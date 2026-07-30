#ifndef GPL_SUMMARY_H
#define GPL_SUMMARY_H

#include <nvml.h>
#include <stdio.h>

#include "args.h"
#include "telemetry.h"
#include "workload.h"

typedef struct {
    char        uuid[80];
    char        name[96];
    char        driver[64];
    char        cuda_runtime[32];
    char        vbios[64];
    char        pci_bus_id[32];
    double      power_limit_w;
} gpl_device_info_t;

int  gpl_device_info_fill(nvmlDevice_t dev, gpl_device_info_t *out);

/* Emit the summary JSON to `out`.
 * `start_utc` is the wall-clock start-of-warmup ISO-8601 string.
 * `wall_ns` is the total process wall time from before-warmup to after-steady. */
int  gpl_summary_emit(FILE *out,
                      const gpl_args_t         *args,
                      const gpl_device_info_t  *dev,
                      const gpl_workload_ctx_t *wctx,
                      gpl_telemetry_t          *tele,
                      const char               *start_utc,
                      uint64_t                  warmup_ns,
                      uint64_t                  steady_ns,
                      uint64_t                  wall_ns,
                      const char               *rung_id,
                      const char               *result_string,
                      const char               *error_message);

#endif
