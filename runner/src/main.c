/*
 * gpu-power-runner — main entry point.
 *
 * Runs a single rung end-to-end:
 *   parse args → CUDA/NVML init → start telemetry → warmup → steady →
 *   stop telemetry → emit summary → cleanup.
 *
 * Exit codes:
 *   0  ok
 *   1  --help
 *   2  bad args
 *   3  nvml init failure
 *   4  cuda init failure
 *   5  workload error (CUDA / cuBLAS)
 *   6  throttle hit under --stop-on-throttle
 */

#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args.h"
#include "probe.h"
#include "summary.h"
#include "telemetry.h"
#include "util.h"
#include "workload.h"

static const char *nvml_str(nvmlReturn_t r) {
    return nvmlErrorString(r);
}

int main(int argc, char **argv) {
    gpl_args_t args;
    int prc = gpl_args_parse(argc, argv, &args);
    if (prc == 1) return 0;
    if (prc != 0) return 2;

    /* NVML init */
    nvmlReturn_t nr = nvmlInit_v2();
    if (nr != NVML_SUCCESS) {
        gpl_errf("nvmlInit: %s", nvml_str(nr));
        return 3;
    }

    nvmlDevice_t nvdev;
    nr = nvmlDeviceGetHandleByIndex_v2(args.device, &nvdev);
    if (nr != NVML_SUCCESS) {
        gpl_errf("nvmlDeviceGetHandleByIndex(%d): %s", args.device, nvml_str(nr));
        nvmlShutdown();
        return 3;
    }

    /* CUDA runtime init on the same index */
    if (cudaSetDevice(args.device) != cudaSuccess) {
        gpl_errf("cudaSetDevice(%d) failed", args.device);
        nvmlShutdown();
        return 4;
    }

    /* Capability probe: what does this platform actually permit?
     * Runs before anything else so a container platform that silently
     * denies the power-limit write fails here, loudly, instead of
     * producing a sweep of unenforced rungs. */
    if (args.probe) {
        int prc2 = gpl_probe_run(nvdev, stdout);
        nvmlShutdown();
        return prc2;
    }

    /* Set an explicit enforced limit. This is the cap-sweep path: on a part
     * whose default limit already equals its maximum, raising does nothing,
     * but lowering opens the whole EDP-vs-cap curve and is where an
     * over-limit excursion is most likely to be observable. */
    if (args.power_limit_w > 0.0) {
        unsigned int mn = 0, mx = 0, want = (unsigned int)(args.power_limit_w * 1000.0);
        if (nvmlDeviceGetPowerManagementLimitConstraints(nvdev, &mn, &mx) == NVML_SUCCESS) {
            if (want < mn || want > mx) {
                gpl_errf("--power-limit %.0f W outside device range %.0f-%.0f W",
                         args.power_limit_w, mn / 1000.0, mx / 1000.0);
                nvmlShutdown();
                return 2;
            }
        }
        nvmlReturn_t sr = nvmlDeviceSetPowerManagementLimit(nvdev, want);
        if (sr == NVML_SUCCESS) {
            gpl_logf("enforced power limit set to %.0f W", args.power_limit_w);
        } else {
            gpl_errf("limit_set: denied (%s) — measuring at the current limit",
                     nvml_str(sr));
        }
    }

    /* Locked clocks: removes DVFS from the loop so rungs are comparable at
     * the cost of realism. Both profiles matter; this is `boost-off`. */
    if (args.lock_clocks) {
        unsigned int sm = 0;
        if (nvmlDeviceGetClockInfo(nvdev, NVML_CLOCK_SM, &sm) == NVML_SUCCESS) {
            if (nvmlDeviceSetGpuLockedClocks(nvdev, sm, sm) == NVML_SUCCESS) {
                gpl_logf("SM clock locked at %u MHz", sm);
            } else {
                gpl_errf("lock_clocks: denied");
            }
        }
    }

    /* Raise the enforced limit to the device maximum (O1). Needs root;
     * degrade cleanly and record it rather than aborting the rung. */
    if (args.raise_power_limit) {
        unsigned int mn = 0, mx = 0;
        if (nvmlDeviceGetPowerManagementLimitConstraints(nvdev, &mn, &mx) == NVML_SUCCESS) {
            nvmlReturn_t sr = nvmlDeviceSetPowerManagementLimit(nvdev, mx);
            if (sr == NVML_SUCCESS) {
                gpl_logf("power limit raised to device max: %.0f W", mx / 1000.0);
            } else {
                gpl_errf("limit_raise: denied (%s) — measuring at the current limit",
                         nvml_str(sr));
            }
        } else {
            gpl_errf("limit_raise: constraints unavailable");
        }
    }

    /* Device info */
    gpl_device_info_t dev;
    gpl_device_info_fill(nvdev, &dev);

    gpl_logf("rung: op=%s prec=%s size=%d streams=%d warmup=%.1fs steady=%.1fs sample=%dHz",
             gpl_op_name(args.op), gpl_prec_name(args.precision),
             args.size, args.streams, args.warmup_sec, args.steady_sec, args.sample_hz);
    gpl_logf("device[%d]: %s  driver=%s  power_limit=%.0fW  cuda_rt=%s",
             args.device, dev.name, dev.driver, dev.power_limit_w, dev.cuda_runtime);

    /* Open per-sample NDJSON if requested. */
    FILE *ndjson = NULL;
    if (args.out_metrics) {
        ndjson = fopen(args.out_metrics, "w");
        if (!ndjson) {
            gpl_errf("cannot open --out-metrics %s", args.out_metrics);
            nvmlShutdown();
            return 2;
        }
    }

    /* Start telemetry (phase = IDLE until workload flips it) */
    gpl_telemetry_t tele;
    if (gpl_telemetry_start(&tele, nvdev, args.sample_hz, ndjson, args.rung_id) != 0) {
        if (ndjson) fclose(ndjson);
        nvmlShutdown();
        return 3;
    }

    /* Wall-clock start-of-warmup timestamp for the summary. */
    char start_utc[40];
    gpl_utc_iso8601(start_utc, sizeof(start_utc));
    uint64_t t0 = gpl_mono_ns();

    /* Run workload. */
    gpl_workload_ctx_t wctx = {0};
    wctx.args = &args;
    wctx.tele = &tele;

    int wrc;
    switch (args.op) {
        case GPL_OP_SGEMM:
            wrc = gpl_workload_gemm_run(&wctx);
            break;
        case GPL_OP_MEMSTREAM:
            wrc = gpl_workload_memstream_run(&wctx);
            break;
        case GPL_OP_FFT:
            wrc = gpl_workload_fft_run(&wctx);
            break;
        case GPL_OP_POWERVIRUS:
            wrc = gpl_workload_powervirus_run(&wctx);
            break;
        case GPL_OP_OBSERVE:
            wrc = gpl_workload_observe_run(&wctx);
            break;
        default:
            wctx.error = "unknown op";
            wrc = -1;
            break;
    }

    /* Stop telemetry regardless of workload result. */
    gpl_telemetry_set_phase(&tele, GPL_PHASE_STOP);
    gpl_telemetry_stop(&tele);
    uint64_t t1 = gpl_mono_ns();

    uint64_t warmup_ns = (uint64_t)(args.warmup_sec * 1e9);
    uint64_t steady_ns = (uint64_t)(wctx.compute_seconds_steady * 1e9);
    uint64_t wall_ns = t1 - t0;

    const char *result_str;
    if (wrc != 0) {
        result_str = wctx.error && strstr(wctx.error, "out of memory") ? "oom" : "cuda_error";
    } else if (wctx.aborted_on_throttle) {
        result_str = "throttle_hit";
    } else {
        result_str = "ok";
    }

    /* Emit summary. */
    FILE *sf = stdout;
    if (args.out_summary) {
        sf = fopen(args.out_summary, "w");
        if (!sf) {
            gpl_errf("cannot open --out-summary %s", args.out_summary);
            sf = stdout;
        }
    }
    gpl_summary_emit(sf, &args, &dev, &wctx, &tele,
                     start_utc, warmup_ns, steady_ns, wall_ns,
                     args.rung_id, result_str, wctx.error);
    if (sf != stdout) fclose(sf);

    /* Leave the device as we found it: a locked clock or a lowered cap
     * left behind would silently contaminate every subsequent rung. */
    if (args.lock_clocks) nvmlDeviceResetGpuLockedClocks(nvdev);
    if (args.power_limit_w > 0.0) {
        unsigned int def = 0;
        if (nvmlDeviceGetPowerManagementDefaultLimit(nvdev, &def) == NVML_SUCCESS) {
            nvmlDeviceSetPowerManagementLimit(nvdev, def);
        }
    }

    if (ndjson) fclose(ndjson);
    nvmlShutdown();

    if (wrc != 0) return 5;
    if (wctx.aborted_on_throttle) return 6;
    return 0;
}
