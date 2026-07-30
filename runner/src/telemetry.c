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

    while (1) {
        gpl_phase_t phase = atomic_load(&t->phase);
        if (phase == GPL_PHASE_STOP) break;

        /* On phase transition into STEADY, capture starting energy. */
        gpl_phase_stats_t *stats =
            phase == GPL_PHASE_WARMUP ? &t->warmup :
            phase == GPL_PHASE_STEADY ? &t->steady : NULL;

        /* Sample. */
        unsigned int power_mw = 0;
        unsigned int temp_gpu = 0;
        unsigned int clock_sm = 0, clock_mem = 0;
        nvmlUtilization_t util = {0};
        unsigned long long energy_mj = 0;
        unsigned long long reasons = 0;
        unsigned int fan_pct = 0;

        nvmlDeviceGetPowerUsage(t->device, &power_mw);
        nvmlDeviceGetTemperature(t->device, NVML_TEMPERATURE_GPU, &temp_gpu);
        nvmlDeviceGetClockInfo(t->device, NVML_CLOCK_SM, &clock_sm);
        nvmlDeviceGetClockInfo(t->device, NVML_CLOCK_MEM, &clock_mem);
        nvmlDeviceGetUtilizationRates(t->device, &util);
        (void)nvmlDeviceGetTotalEnergyConsumption(t->device, &energy_mj);
        (void)nvmlDeviceGetCurrentClocksThrottleReasons(t->device, &reasons);
        (void)nvmlDeviceGetFanSpeed(t->device, &fan_pct);

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
            gpl_dvec_push(&stats->temp_c, temp_c);
            gpl_dvec_push(&stats->sm_mhz, (double)clock_sm);
            gpl_dvec_push(&stats->mem_mhz, (double)clock_mem);
            gpl_dvec_push(&stats->sm_util, (double)util.gpu);
            gpl_dvec_push(&stats->mem_util, (double)util.memory);
            stats->throttle_mask_or |= (uint64_t)reasons;
            if (((uint64_t)reasons & k_meaningful_throttle_mask) != 0) {
                stats->throttled_sec += 1.0 / (double)t->sample_hz;
                if (phase == GPL_PHASE_STEADY) atomic_store(&t->throttle_seen_steady, true);
            }
            if (energy_mj > 0) stats->energy_end_mj = energy_mj;
            stats->samples++;
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
                "\"energy_mj\":%llu,\"throttle_mask\":%llu,\"fan_pct\":%u}\n",
                ts, t->rung_id ? t->rung_id : "", phase_name(phase),
                power_w, temp_c, clock_sm, clock_mem,
                util.gpu, util.memory,
                (unsigned long long)energy_mj,
                (unsigned long long)reasons,
                fan_pct);
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
