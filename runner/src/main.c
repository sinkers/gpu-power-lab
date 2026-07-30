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

    if (ndjson) fclose(ndjson);
    nvmlShutdown();

    if (wrc != 0) return 5;
    if (wctx.aborted_on_throttle) return 6;
    return 0;
}
