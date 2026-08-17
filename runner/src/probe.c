/*
 * Capability probe.
 *
 * TESTPLAN.md: container and k8s GPU platforms deny most of what this
 * harness needs, and they deny it quietly — the campaign completes and the
 * numbers are wrong rather than absent. So every campaign starts by asking
 * the platform what it actually permits, records the answer in the
 * manifest, and fails fast with a readable message instead of producing a
 * sweep where half the rungs were never enforced.
 *
 * Read-only except for the power-limit check, which sets the limit to the
 * value it already has. That is the only way to find out whether the write
 * path is permitted without changing anything.
 */

#include "probe.h"

#include <nvml.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static const char *yn(bool b) { return b ? "yes" : "no"; }

static void jbool(FILE *f, const char *k, bool v, bool comma) {
    fprintf(f, "  \"%s\": %s%s\n", k, v ? "true" : "false", comma ? "," : "");
}

int gpl_probe_run(nvmlDevice_t dev, FILE *out) {
    gpl_probe_t p;
    memset(&p, 0, sizeof(p));

    char name[96] = "unknown";
    nvmlDeviceGetName(dev, name, sizeof(name));

    /* --- power read paths --- */
    unsigned int mw = 0;
    p.power_usage = (nvmlDeviceGetPowerUsage(dev, &mw) == NVML_SUCCESS);

    unsigned long long energy = 0;
    p.energy_counter =
        (nvmlDeviceGetTotalEnergyConsumption(dev, &energy) == NVML_SUCCESS);

    /* POWER_INSTANT vs POWER_AVERAGE: the divergence between these two is
     * our only in-band evidence of a transient, so knowing whether both
     * exist on this part matters more than either value. */
#ifdef NVML_FI_DEV_POWER_INSTANT
    nvmlFieldValue_t fv[2];
    memset(fv, 0, sizeof(fv));
    fv[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    fv[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;
    if (nvmlDeviceGetFieldValues(dev, 2, fv) == NVML_SUCCESS) {
        p.power_instant = (fv[0].nvmlReturn == NVML_SUCCESS);
        p.power_average = (fv[1].nvmlReturn == NVML_SUCCESS);
    }
#endif

    /* Driver-buffered samples: beats polling because the timestamps come
     * from the device, not from our thread's wakeup. */
    nvmlValueType_t vt;
    unsigned int scount = 0;
    nvmlReturn_t sr = nvmlDeviceGetSamples(dev, NVML_TOTAL_POWER_SAMPLES, 0,
                                           &vt, &scount, NULL);
    p.power_samples = (sr == NVML_SUCCESS || sr == NVML_ERROR_INSUFFICIENT_SIZE);
    p.power_samples_count = scount;

    nvmlViolationTime_t vio;
    p.violation_status =
        (nvmlDeviceGetViolationStatus(dev, NVML_PERF_POLICY_POWER, &vio) == NVML_SUCCESS);

    /* --- limits --- */
    unsigned int cur = 0, def = 0, minl = 0, maxl = 0;
    if (nvmlDeviceGetPowerManagementLimit(dev, &cur) == NVML_SUCCESS)
        p.limit_current_w = cur / 1000.0;
    if (nvmlDeviceGetPowerManagementDefaultLimit(dev, &def) == NVML_SUCCESS)
        p.limit_default_w = def / 1000.0;
    if (nvmlDeviceGetPowerManagementLimitConstraints(dev, &minl, &maxl) == NVML_SUCCESS) {
        p.limit_min_w = minl / 1000.0;
        p.limit_max_w = maxl / 1000.0;
        p.limit_constraints = true;
    }

    /* Write path: set the limit to what it already is. Succeeds only where
     * we have the privilege, changes nothing either way. */
    if (cur > 0) {
        p.limit_write = (nvmlDeviceSetPowerManagementLimit(dev, cur) == NVML_SUCCESS);
    }

    nvmlEnableState_t pm = NVML_FEATURE_DISABLED;
    if (nvmlDeviceGetPersistenceMode(dev, &pm) == NVML_SUCCESS) {
        p.persistence_on = (pm == NVML_FEATURE_ENABLED);
        p.persistence_write =
            (nvmlDeviceSetPersistenceMode(dev, pm) == NVML_SUCCESS);
    }

    unsigned int sm_clk = 0;
    p.clock_read = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_clk) == NVML_SUCCESS);
    /* Locking clocks to their current value is the same trick as above. */
    p.clock_lock = (nvmlDeviceSetGpuLockedClocks(dev, sm_clk, sm_clk) == NVML_SUCCESS);
    if (p.clock_lock) nvmlDeviceResetGpuLockedClocks(dev);

    /* --- sampler thread scheduling --- */
    struct sched_param sp = { .sched_priority = 10 };
    p.sched_rr = (pthread_setschedparam(pthread_self(), SCHED_RR, &sp) == 0);
    if (p.sched_rr) {
        struct sched_param back = { .sched_priority = 0 };
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &back);
    }

    /* --- MIG: a MIG instance reports device-wide power that isn't ours --- */
    nvmlEnableState_t mig_cur = NVML_FEATURE_DISABLED, mig_pend;
    if (nvmlDeviceGetMigMode(dev, &mig_cur, &mig_pend) == NVML_SUCCESS)
        p.mig_enabled = (mig_cur == NVML_FEATURE_ENABLED);

    /* --- verdict --- */
    p.can_measure_o1 = p.power_usage && p.limit_write && !p.mig_enabled;
    p.can_measure_o2 = p.can_measure_o1 && (p.power_samples || p.power_instant);

    fprintf(out, "{\n");
    fprintf(out, "  \"device_name\": \"%s\",\n", name);
    jbool(out, "power_usage", p.power_usage, true);
    jbool(out, "power_instant", p.power_instant, true);
    jbool(out, "power_average", p.power_average, true);
    jbool(out, "power_samples", p.power_samples, true);
    fprintf(out, "  \"power_samples_available\": %u,\n", p.power_samples_count);
    jbool(out, "energy_counter", p.energy_counter, true);
    jbool(out, "violation_status", p.violation_status, true);
    jbool(out, "limit_constraints", p.limit_constraints, true);
    fprintf(out, "  \"limit_current_w\": %.1f,\n", p.limit_current_w);
    fprintf(out, "  \"limit_default_w\": %.1f,\n", p.limit_default_w);
    fprintf(out, "  \"limit_min_w\": %.1f,\n", p.limit_min_w);
    fprintf(out, "  \"limit_max_w\": %.1f,\n", p.limit_max_w);
    jbool(out, "limit_write", p.limit_write, true);
    jbool(out, "persistence_on", p.persistence_on, true);
    jbool(out, "persistence_write", p.persistence_write, true);
    jbool(out, "clock_read", p.clock_read, true);
    jbool(out, "clock_lock", p.clock_lock, true);
    jbool(out, "sched_rr", p.sched_rr, true);
    jbool(out, "mig_enabled", p.mig_enabled, true);
    jbool(out, "can_measure_o1", p.can_measure_o1, true);
    jbool(out, "can_measure_o2", p.can_measure_o2, false);
    fprintf(out, "}\n");

    /* Human-readable verdict on stderr so it is visible in campaign logs
     * even when stdout is being captured as JSON. */
    gpl_logf("probe: %s", name);
    gpl_logf("  power read=%s instant=%s samples=%s(%u) energy=%s violation=%s",
             yn(p.power_usage), yn(p.power_instant), yn(p.power_samples),
             p.power_samples_count, yn(p.energy_counter), yn(p.violation_status));
    gpl_logf("  limit cur=%.0fW def=%.0fW max=%.0fW write=%s persistence=%s lock=%s",
             p.limit_current_w, p.limit_default_w, p.limit_max_w,
             yn(p.limit_write), yn(p.persistence_write), yn(p.clock_lock));
    gpl_logf("  sched_rr=%s mig=%s", yn(p.sched_rr), yn(p.mig_enabled));
    if (!p.can_measure_o1) {
        gpl_errf("  VERDICT: cannot measure O1 here.%s%s%s",
                 p.power_usage ? "" : " No power readings.",
                 p.limit_write ? "" : " Power-limit write denied (container platform?).",
                 p.mig_enabled ? " MIG is on — power is not this instance's." : "");
    } else if (!p.can_measure_o2) {
        gpl_logf("  VERDICT: O1 ok; O2 limited — no high-rate power path on this part.");
    } else {
        gpl_logf("  VERDICT: O1 and O2 both measurable here.");
    }

    return p.can_measure_o1 ? 0 : 7;
}
