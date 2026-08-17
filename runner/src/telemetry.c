#include "telemetry.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Stable name table for NVML throttle reasons. The NVML header spells these
 * as `nvmlClocksEventReason*` in recent drivers; we hard-code the bits to keep
 * the mapping independent of header vintage. */
static const struct { uint64_t bit; const char *name; } k_throttle_names[] = {
    { 0x0000000000000001ULL, "gpu_idle" },
    { 0x0000000000000002ULL, "applications_clocks_setting" },
    { 0x0000000000000004ULL, "sw_power_cap" },
    { 0x0000000000000008ULL, "hw_slowdown" },
    { 0x0000000000000010ULL, "sync_boost" },
    { 0x0000000000000020ULL, "sw_thermal_slowdown" },
    { 0x0000000000000040ULL, "hw_thermal_slowdown" },
    { 0x0000000000000080ULL, "hw_power_brake_slowdown" },
    { 0x0000000000000100ULL, "display_clocks_setting" },
};

int gpl_throttle_reason_names(uint64_t mask, const char **out, int max) {
    int n = 0;
    for (size_t i = 0; i < sizeof(k_throttle_names) / sizeof(k_throttle_names[0]) && n < max; i++) {
        if (mask & k_throttle_names[i].bit) {
            out[n++] = k_throttle_names[i].name;
        }
    }
    return n;
}

/* Any bit indicating meaningful throttling. We exclude gpu_idle (0x1) and
 * applications_clocks_setting (0x2) because they are neutral. */
static const uint64_t k_meaningful_throttle_mask =
    0x0000000000000004ULL |   /* sw_power_cap */
    0x0000000000000008ULL |   /* hw_slowdown */
    0x0000000000000020ULL |   /* sw_thermal_slowdown */
    0x0000000000000040ULL |   /* hw_thermal_slowdown */
    0x0000000000000080ULL ;   /* hw_power_brake_slowdown */

static void phase_stats_init(gpl_phase_stats_t *s) {
    gpl_dvec_init(&s->power_w, 4096);
    gpl_dvec_init(&s->power_instant_w, 4096);
    gpl_dvec_init(&s->temp_c, 4096);
    gpl_dvec_init(&s->sm_mhz, 4096);
    gpl_dvec_init(&s->mem_mhz, 4096);
    gpl_dvec_init(&s->sm_util, 4096);
    gpl_dvec_init(&s->mem_util, 4096);
    s->throttle_mask_or = 0;
    s->throttled_sec = 0.0;
    s->energy_start_mj = 0;
    s->energy_end_mj = 0;
    s->energy_valid = false;
    s->samples = 0;
    s->power_integral_j = 0.0;
    s->elapsed_s = 0.0;
    s->violation_us_start = 0;
    s->violation_us_end = 0;
    s->violation_valid = false;
    s->peak_instant_w = 0.0;
}

bool gpl_telemetry_throttled_now(gpl_telemetry_t *t) {
    return atomic_load(&t->throttle_seen_steady);
}

void gpl_telemetry_set_phase(gpl_telemetry_t *t, gpl_phase_t p) {
    atomic_store(&t->phase, p);
}

static const char *phase_name(gpl_phase_t p) {
    switch (p) {
        case GPL_PHASE_IDLE:   return "idle";
        case GPL_PHASE_WARMUP: return "warmup";
        case GPL_PHASE_STEADY: return "steady";
        case GPL_PHASE_STOP:   return "stop";
        default:               return "unknown";
    }
}

static void *sampler_thread(void *arg) {
    gpl_telemetry_t *t = (gpl_telemetry_t *)arg;
    const long period_ns = 1000000000L / t->sample_hz;

    /* Best-effort real-time policy — ignore failure (unprivileged is fine). */
    struct sched_param sp = { .sched_priority = 10 };
    pthread_setschedparam(pthread_self(), SCHED_RR, &sp);

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    gpl_phase_t prev_phase = GPL_PHASE_IDLE;
    uint64_t prev_ns = gpl_mono_ns();

    while (1) {
        gpl_phase_t phase = atomic_load(&t->phase);
        if (phase == GPL_PHASE_STOP) break;

        /* On phase transition into STEADY, capture starting energy. */
        gpl_phase_stats_t *stats =
            phase == GPL_PHASE_WARMUP ? &t->warmup :
            phase == GPL_PHASE_STEADY ? &t->steady : NULL;

        /* Sample. */
        unsigned int power_mw = 0;
        unsigned int temp_gpu = 0, temp_mem = 0;
        unsigned int clock_sm = 0, clock_mem = 0;
        nvmlUtilization_t util = {0};
        unsigned long long energy_mj = 0;
        unsigned long long reasons = 0;
        unsigned int fan_pct = 0;
        double power_instant_w = -1.0;
        unsigned long long violation_us = 0;
        bool violation_ok = false;

        nvmlDeviceGetPowerUsage(t->device, &power_mw);

#ifdef NVML_FI_DEV_POWER_INSTANT
        if (t->have_instant) {
            nvmlFieldValue_t fv;
            memset(&fv, 0, sizeof(fv));
            fv.fieldId = NVML_FI_DEV_POWER_INSTANT;
            if (nvmlDeviceGetFieldValues(t->device, 1, &fv) == NVML_SUCCESS &&
                fv.nvmlReturn == NVML_SUCCESS) {
                /* Field is milliwatts, but the value type varies by driver. */
                double mw_i = (fv.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)
                                ? (double)fv.value.uiVal
                                : (double)fv.value.ullVal;
                power_instant_w = mw_i / 1000.0;
            }
        }
#endif
        if (t->have_violation) {
            nvmlViolationTime_t vio;
            if (nvmlDeviceGetViolationStatus(t->device, NVML_PERF_POLICY_POWER,
                                             &vio) == NVML_SUCCESS) {
                violation_us = vio.violationTime;
                violation_ok = true;
            }
        }
        nvmlDeviceGetTemperature(t->device, NVML_TEMPERATURE_GPU, &temp_gpu);
        /* HBM temperature, where the part exposes it. On stacked-memory parts
         * this is often the binding thermal limit rather than the core, so a
         * soak run that only watches the core can miss why power droops.
         * nvmlTemperatureSensors_t has no memory sensor - GPU is the only
         * member - so this comes through the field-value API instead. */
#ifdef NVML_FI_DEV_MEMORY_TEMP
        if (t->have_temp_mem) {
            nvmlFieldValue_t fvm;
            memset(&fvm, 0, sizeof(fvm));
            fvm.fieldId = NVML_FI_DEV_MEMORY_TEMP;
            if (nvmlDeviceGetFieldValues(t->device, 1, &fvm) == NVML_SUCCESS &&
                fvm.nvmlReturn == NVML_SUCCESS) {
                temp_mem = (fvm.valueType == NVML_VALUE_TYPE_UNSIGNED_INT)
                             ? fvm.value.uiVal : (unsigned int)fvm.value.ullVal;
            }
        }
#endif
        nvmlDeviceGetClockInfo(t->device, NVML_CLOCK_SM, &clock_sm);
        nvmlDeviceGetClockInfo(t->device, NVML_CLOCK_MEM, &clock_mem);
        nvmlDeviceGetUtilizationRates(t->device, &util);
        (void)nvmlDeviceGetTotalEnergyConsumption(t->device, &energy_mj);
        (void)nvmlDeviceGetCurrentClocksThrottleReasons(t->device, &reasons);
        (void)nvmlDeviceGetFanSpeed(t->device, &fan_pct);

        uint64_t now_ns = gpl_mono_ns();
        double dt_s = (double)(now_ns - prev_ns) / 1e9;
        prev_ns = now_ns;
        if (dt_s <= 0.0 || dt_s > 1.0) dt_s = 1.0 / (double)t->sample_hz;

        double power_w = (double)power_mw / 1000.0;
        double temp_c  = (double)temp_gpu;

        if (stats) {
            /* Phase transition: capture energy anchor. */
            if (phase != prev_phase && phase == GPL_PHASE_STEADY) {
                if (energy_mj > 0) {
                    stats->energy_start_mj = energy_mj;
                    stats->energy_valid = true;
                }
            }
            gpl_dvec_push(&stats->power_w, power_w);
            /* Integrate against measured dt, not the nominal period. The
             * sampler does not always achieve --sample-hz: each tick makes
             * several NVML calls and on some drivers they are slow enough
             * to halve the rate. Using 1/sample_hz here silently loses
             * area and shows up as a bogus energy gap. */
            stats->power_integral_j += power_w * dt_s;
            if (power_instant_w >= 0.0) {
                gpl_dvec_push(&stats->power_instant_w, power_instant_w);
                if (power_instant_w > stats->peak_instant_w)
                    stats->peak_instant_w = power_instant_w;
            }
            if (violation_ok) {
                if (!stats->violation_valid) {
                    stats->violation_us_start = violation_us;
                    stats->violation_valid = true;
                }
                stats->violation_us_end = violation_us;
            }
            gpl_dvec_push(&stats->temp_c, temp_c);
            gpl_dvec_push(&stats->sm_mhz, (double)clock_sm);
            gpl_dvec_push(&stats->mem_mhz, (double)clock_mem);
            gpl_dvec_push(&stats->sm_util, (double)util.gpu);
            gpl_dvec_push(&stats->mem_util, (double)util.memory);
            stats->throttle_mask_or |= (uint64_t)reasons;
            if (((uint64_t)reasons & k_meaningful_throttle_mask) != 0) {
                stats->throttled_sec += dt_s;
                if (phase == GPL_PHASE_STEADY) atomic_store(&t->throttle_seen_steady, true);
            }
            if (energy_mj > 0) stats->energy_end_mj = energy_mj;
            stats->samples++;
            stats->elapsed_s += dt_s;
        }

        /* Optional per-sample NDJSON. */
        if (t->ndjson) {
            char ts[40];
            gpl_utc_iso8601(ts, sizeof(ts));
            fprintf(t->ndjson,
                "{\"ts\":\"%s\",\"rung_id\":\"%s\",\"phase\":\"%s\","
                "\"power_w\":%.3f,\"temp_c\":%.1f,"
                "\"sm_mhz\":%u,\"mem_mhz\":%u,"
                "\"sm_util\":%u,\"mem_util\":%u,"
                "\"energy_mj\":%llu,\"throttle_mask\":%llu,\"fan_pct\":%u,"
                "\"temp_mem_c\":%u,"
                "\"power_instant_w\":%.3f,\"source\":\"%s\"}\n",
                ts, t->rung_id ? t->rung_id : "", phase_name(phase),
                power_w, temp_c, clock_sm, clock_mem,
                util.gpu, util.memory,
                (unsigned long long)energy_mj,
                (unsigned long long)reasons,
                fan_pct,
                temp_mem,
                power_instant_w,
                power_instant_w >= 0.0 ? "nvml_usage+instant" : "nvml_usage");
            /* Flush every 100 samples so tail -f is useful and a crash
             * loses at most ~1s of data. */
            if ((t->warmup.samples + t->steady.samples) % 100 == 0) {
                fflush(t->ndjson);
            }
        }

        prev_phase = phase;

        /* Absolute sleep to next tick — resistant to accumulated drift. */
        next.tv_nsec += period_ns;
        while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
        int rc;
        do {
            rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
        } while (rc == EINTR);
    }

    return NULL;
}

int gpl_telemetry_start(gpl_telemetry_t *t, nvmlDevice_t dev, int sample_hz,
                        FILE *ndjson_out, const char *rung_id) {
    memset(t, 0, sizeof(*t));
    t->device = dev;
    t->sample_hz = sample_hz;
    t->ndjson = ndjson_out;
    t->rung_id = rung_id;
    /* Resolve capabilities once — polling a field that is unsupported on
     * this part costs a syscall per sample for nothing. */
#ifdef NVML_FI_DEV_POWER_INSTANT
    {
        nvmlFieldValue_t fv;
        memset(&fv, 0, sizeof(fv));
        fv.fieldId = NVML_FI_DEV_POWER_INSTANT;
        t->have_instant = (nvmlDeviceGetFieldValues(dev, 1, &fv) == NVML_SUCCESS &&
                           fv.nvmlReturn == NVML_SUCCESS);
    }
#else
    t->have_instant = false;
#endif
    {
        /* Not every part exposes an HBM sensor; probe once rather than
         * paying a failing field query on every sample. */
#ifdef NVML_FI_DEV_MEMORY_TEMP
        nvmlFieldValue_t fvm;
        memset(&fvm, 0, sizeof(fvm));
        fvm.fieldId = NVML_FI_DEV_MEMORY_TEMP;
        t->have_temp_mem = (nvmlDeviceGetFieldValues(dev, 1, &fvm) == NVML_SUCCESS &&
                            fvm.nvmlReturn == NVML_SUCCESS);
#else
        t->have_temp_mem = false;
#endif
    }
    {
        nvmlViolationTime_t vio;
        t->have_violation =
            (nvmlDeviceGetViolationStatus(dev, NVML_PERF_POLICY_POWER, &vio) == NVML_SUCCESS);
    }

    atomic_store(&t->phase, GPL_PHASE_IDLE);
    atomic_store(&t->throttle_seen_steady, false);
    phase_stats_init(&t->warmup);
    phase_stats_init(&t->steady);

    if (pthread_create(&t->thread, NULL, sampler_thread, t) != 0) {
        gpl_errf("pthread_create failed: %s", strerror(errno));
        return -1;
    }
    t->thread_started = true;
    return 0;
}

void gpl_telemetry_stop(gpl_telemetry_t *t) {
    if (!t->thread_started) return;
    atomic_store(&t->phase, GPL_PHASE_STOP);
    pthread_join(t->thread, NULL);
    t->thread_started = false;
    if (t->ndjson) fflush(t->ndjson);
}
