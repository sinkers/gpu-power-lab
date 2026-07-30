#ifndef GPL_UTIL_H
#define GPL_UTIL_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>

/* Monotonic nanoseconds since some fixed origin. */
static inline uint64_t gpl_mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Current wall-clock time as ISO-8601 UTC (e.g. "2026-07-28T10:37:04.123Z").
 * Buffer must be at least 32 bytes. */
void gpl_utc_iso8601(char *buf, size_t buflen);

/* Format a uint64 into a null-terminated hex string prefixed with "0x". */
void gpl_u64_to_hex(uint64_t v, char *buf, size_t buflen);

/* Simple growable buffer of doubles for online percentile at end. */
typedef struct {
    double *data;
    size_t len;
    size_t cap;
} gpl_dvec;

int  gpl_dvec_init(gpl_dvec *v, size_t initial_cap);
int  gpl_dvec_push(gpl_dvec *v, double x);
void gpl_dvec_free(gpl_dvec *v);

/* Nearest-rank percentile in [0,100]. Sorts v in-place. */
double gpl_dvec_percentile(gpl_dvec *v, double pct);

/* Log helpers. */
#define gpl_logf(fmt, ...) fprintf(stderr, "[runner] " fmt "\n", ##__VA_ARGS__)
#define gpl_errf(fmt, ...) fprintf(stderr, "[runner][ERROR] " fmt "\n", ##__VA_ARGS__)

#endif
