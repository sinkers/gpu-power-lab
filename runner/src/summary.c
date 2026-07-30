#include "summary.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util.h"

int gpl_device_info_fill(nvmlDevice_t dev, gpl_device_info_t *out) {
    memset(out, 0, sizeof(*out));
    nvmlDeviceGetUUID(dev, out->uuid, sizeof(out->uuid));
    nvmlDeviceGetName(dev, out->name, sizeof(out->name));
    /* System-wide driver version, not per-device — but a device handle is fine. */
    nvmlSystemGetDriverVersion(out->driver, sizeof(out->driver));

    int cuda_rt = 0;
    cudaRuntimeGetVersion(&cuda_rt);
    snprintf(out->cuda_runtime, sizeof(out->cuda_runtime),
             "%d.%d", cuda_rt / 1000, (cuda_rt % 1000) / 10);

    nvmlDeviceGetVbiosVersion(dev, out->vbios, sizeof(out->vbios));

    nvmlPciInfo_t pci;
    if (nvmlDeviceGetPciInfo(dev, &pci) == NVML_SUCCESS) {
        snprintf(out->pci_bus_id, sizeof(out->pci_bus_id), "%s", pci.busId);
    }

    unsigned int limit_mw = 0;
    if (nvmlDeviceGetPowerManagementLimit(dev, &limit_mw) == NVML_SUCCESS) {
        out->power_limit_w = (double)limit_mw / 1000.0;
    }
    return 0;
}

/* Compute (min, max, sum) so we can derive avg and range. */
static void stats_min_max_sum(const gpl_dvec *v, double *mn, double *mx, double *sum) {
    if (v->len == 0) { *mn = *mx = *sum = 0.0; return; }
    double a = v->data[0], b = v->data[0], s = 0.0;
    for (size_t i = 0; i < v->len; i++) {
        double x = v->data[i];
        if (x < a) a = x;
        if (x > b) b = x;
        s += x;
    }
    *mn = a; *mx = b; *sum = s;
}

static double avg_of(const gpl_dvec *v) {
    if (v->len == 0) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < v->len; i++) s += v->data[i];
    return s / (double)v->len;
}

int gpl_summary_emit(FILE *out,
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
                     const char               *error_message) {
    gpl_phase_stats_t *s = &tele->steady;

    /* Power percentiles — computed by copying into scratch vec so we don't
     * destroy the original ordering (we don't strictly need it later but
     * keep the invariant clean). */
    gpl_dvec pw; gpl_dvec_init(&pw, s->power_w.len + 1);
    for (size_t i = 0; i < s->power_w.len; i++) gpl_dvec_push(&pw, s->power_w.data[i]);

    double p_mn, p_mx, p_sum;
    stats_min_max_sum(&s->power_w, &p_mn, &p_mx, &p_sum);
    double avg_w  = s->power_w.len ? p_sum / (double)s->power_w.len : 0.0;
    double p50_w  = gpl_dvec_percentile(&pw, 50.0);
    double p95_w  = gpl_dvec_percentile(&pw, 95.0);
    double p99_w  = gpl_dvec_percentile(&pw, 99.0);
    gpl_dvec_free(&pw);

    /* Energy: prefer NVML monotonic counter delta; fall back to avg_w * seconds. */
    double energy_j = 0.0;
    if (s->energy_valid && s->energy_end_mj >= s->energy_start_mj) {
        energy_j = (double)(s->energy_end_mj - s->energy_start_mj) / 1000.0;
    } else {
        energy_j = avg_w * ((double)steady_ns / 1e9);
    }

    double t_mn, t_mx, t_sum;
    stats_min_max_sum(&s->temp_c, &t_mn, &t_mx, &t_sum);
    double t_avg = s->temp_c.len ? t_sum / (double)s->temp_c.len : 0.0;

    double sm_mn, sm_mx, sm_sum;
    stats_min_max_sum(&s->sm_mhz, &sm_mn, &sm_mx, &sm_sum);
    double sm_avg = s->sm_mhz.len ? sm_sum / (double)s->sm_mhz.len : 0.0;
    double mem_avg = avg_of(&s->mem_mhz);
    double sm_util_avg  = avg_of(&s->sm_util);
    double mem_util_avg = avg_of(&s->mem_util);

    /* TFLOPS (zero for non-GEMM ops; throughput section below handles those). */
    double flops_per_iter = gpl_workload_flops_per_iter(args);
    double tflops = 0.0;
    if (flops_per_iter > 0.0
            && wctx->compute_seconds_steady > 0.0
            && wctx->iterations_steady > 0) {
        tflops = (flops_per_iter * (double)wctx->iterations_steady)
               / wctx->compute_seconds_steady / 1e12;
    }

    /* Throttle name list. */
    const char *names[16];
    int nnames = gpl_throttle_reason_names(s->throttle_mask_or, names, 16);
    bool any_throttle = (s->throttled_sec > 0.0);

    /* SM boost dropped heuristic: min SM clock < 90% of max SM clock during
     * steady window (indicates a downward step). */
    bool sm_boost_dropped = (s->sm_mhz.len > 4 && sm_mn < 0.9 * sm_mx);

    /* Fabricate a rung_id if none provided. */
    char rid[128];
    if (rung_id && *rung_id) {
        snprintf(rid, sizeof(rid), "%s", rung_id);
    } else {
        snprintf(rid, sizeof(rid), "%s-%s-%d-s%d-%s",
                 gpl_op_name(args->op), gpl_prec_name(args->precision),
                 args->size, args->streams, start_utc);
    }

    fprintf(out,
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"rung_id\": \"%s\",\n"
        "  \"invocation\": {\n"
        "    \"op\": \"%s\", \"precision\": \"%s\", \"size\": %d, \"streams\": %d,\n"
        "    \"warmup_sec\": %.3f, \"steady_sec\": %.3f, \"sample_hz\": %d, \"device\": %d\n"
        "  },\n"
        "  \"device\": {\n"
        "    \"uuid\": \"%s\", \"name\": \"%s\", \"driver\": \"%s\",\n"
        "    \"cuda_runtime\": \"%s\", \"vbios\": \"%s\", \"pci_bus_id\": \"%s\",\n"
        "    \"power_limit_w\": %.2f\n"
        "  },\n"
        "  \"timing\": {\n"
        "    \"start_utc\": \"%s\", \"warmup_ms\": %.2f, \"steady_ms\": %.2f, \"wall_ms\": %.2f\n"
        "  },\n"
        "  \"compute\": {\n"
        "    \"iterations\": %llu, \"tflops_measured\": %.4f,\n"
        "    \"tflops_theoretical\": null, \"efficiency\": null\n"
        "  },\n"
        "  \"power\": {\n"
        "    \"avg_w\": %.3f, \"peak_w\": %.3f, \"p50_w\": %.3f, \"p95_w\": %.3f, \"p99_w\": %.3f,\n"
        "    \"energy_j\": %.3f\n"
        "  },\n"
        "  \"thermal\": {\n"
        "    \"avg_c\": %.2f, \"peak_c\": %.2f, \"throttled_sec\": %.3f\n"
        "  },\n"
        "  \"throttle\": {\n"
        "    \"mask\": %llu, \"any_throttled\": %s, \"reasons\": [",
        rid,
        gpl_op_name(args->op), gpl_prec_name(args->precision),
        args->size, args->streams,
        args->warmup_sec, args->steady_sec, args->sample_hz, args->device,
        dev->uuid, dev->name, dev->driver, dev->cuda_runtime, dev->vbios, dev->pci_bus_id,
        dev->power_limit_w,
        start_utc,
        (double)warmup_ns / 1e6, (double)steady_ns / 1e6, (double)wall_ns / 1e6,
        (unsigned long long)wctx->iterations_steady, tflops,
        avg_w, p_mx, p50_w, p95_w, p99_w, energy_j,
        t_avg, t_mx, s->throttled_sec,
        (unsigned long long)s->throttle_mask_or,
        any_throttle ? "true" : "false"
    );

    for (int i = 0; i < nnames; i++) {
        fprintf(out, "%s\"%s\"", i ? ", " : "", names[i]);
    }

    fprintf(out,
        "]\n"
        "  },\n"
        "  \"utilization\": {\n"
        "    \"sm_avg_pct\": %.2f, \"mem_avg_pct\": %.2f\n"
        "  },\n"
        "  \"clocks\": {\n"
        "    \"sm_avg_mhz\": %.1f, \"sm_min_mhz\": %.1f, \"sm_max_mhz\": %.1f,\n"
        "    \"mem_avg_mhz\": %.1f, \"sm_boost_dropped\": %s\n"
        "  },\n"
        "  \"samples_written\": %llu,\n"
        "  \"result\": \"%s\"",
        sm_util_avg, mem_util_avg,
        sm_avg, sm_mn, sm_mx, mem_avg, sm_boost_dropped ? "true" : "false",
        (unsigned long long)s->samples,
        result_string
    );

    if (error_message && *error_message) {
        fprintf(out, ",\n  \"error_message\": \"%s\"", error_message);
    }

    /* Throughput section: emitted for bandwidth-bound and FFT workloads.
     * MEMSTREAM reports GB/s (bytes moved / time).
     * FFT reports GFFT/s (giga-transforms per second, i.e. batch × iters / s / 1e9). */
    double bytes_per_iter = gpl_workload_bytes_per_iter(args);
    if (bytes_per_iter > 0.0
            && wctx->compute_seconds_steady > 0.0
            && wctx->iterations_steady > 0) {
        double      tp_value;
        const char *tp_unit;
        if (args->op == GPL_OP_MEMSTREAM) {
            tp_value = (bytes_per_iter * (double)wctx->iterations_steady)
                     / wctx->compute_seconds_steady / 1e9;
            tp_unit  = "GB/s";
        } else { /* GPL_OP_FFT */
            /* Derive batch from the same formula used in workload_fft.c. */
            int fft_batch = (1 << 20) / args->size;
            if (fft_batch < 1) fft_batch = 1;
            tp_value = ((double)fft_batch * (double)wctx->iterations_steady)
                     / wctx->compute_seconds_steady / 1e9;
            tp_unit  = "GFFT/s";
        }
        fprintf(out, ",\n  \"throughput\": { \"op_specific_unit\": \"%s\", \"value\": %.4f }",
                tp_unit, tp_value);
    }

    fprintf(out, "\n}\n");

    return 0;
}
