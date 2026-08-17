/*
 * observe — sample only, drive nothing.
 *
 * Every other op has the runner owning the workload. That does not work for
 * the things we actually want to measure next: a PyTorch training step or a
 * vLLM server is a separate process, and the runner's job there is purely to
 * watch. This op runs the same telemetry loop, the same phases and the same
 * summary schema with no CUDA work of its own, so everything downstream —
 * the report generator, the energy cross-check, the transient maths — works
 * unchanged on a workload we did not write.
 *
 * It is also how the idle floor gets measured, which matters more than it
 * sounds: every swing number in this project is a distance from that floor,
 * and until now we had never actually recorded it.
 *
 * Iterations are counted in samples rather than kernel launches, so the
 * efficiency block is meaningless here and the summary reports
 * work_unit=iteration with the caller's own interpretation. Nothing to do
 * about that: there is no work to divide by.
 */

#include "workload.h"

#include <errno.h>
#include <time.h>

#include "util.h"

/* Sleep until an absolute monotonic deadline, tolerating signal wakeups. */
static void sleep_until_ns(uint64_t deadline_ns) {
    struct timespec ts;
    ts.tv_sec  = (time_t)(deadline_ns / 1000000000ull);
    ts.tv_nsec = (long)(deadline_ns % 1000000000ull);
    int rc;
    do {
        rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
    } while (rc == EINTR);
}

int gpl_workload_observe_run(gpl_workload_ctx_t *ctx) {
    const gpl_args_t *a = ctx->args;

    gpl_logf("observe: sampling only — this process drives no GPU work");

    uint64_t t0 = gpl_mono_ns();

    if (a->warmup_sec > 0.0) {
        gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_WARMUP);
        sleep_until_ns(t0 + (uint64_t)(a->warmup_sec * 1e9));
    }

    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_STEADY);
    uint64_t s0 = gpl_mono_ns();
    uint64_t deadline = s0 + (uint64_t)(a->steady_sec * 1e9);

    /* Wake once a second rather than sleeping the whole window in one call,
     * so --stop-on-throttle stays responsive and a long observe run can be
     * interrupted by the same mechanism as any other rung. */
    while (gpl_mono_ns() < deadline) {
        uint64_t next = gpl_mono_ns() + 1000000000ull;
        if (next > deadline) next = deadline;
        sleep_until_ns(next);
        ctx->iterations_steady++;
        if (a->stop_on_throttle && gpl_telemetry_throttled_now(ctx->tele)) {
            ctx->aborted_on_throttle = true;
            break;
        }
    }

    ctx->compute_seconds_steady = (double)(gpl_mono_ns() - s0) / 1e9;
    return 0;
}
