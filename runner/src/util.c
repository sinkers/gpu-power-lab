#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void gpl_utc_iso8601(char *buf, size_t buflen) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    int ms = (int)(ts.tv_nsec / 1000000);
    snprintf(buf, buflen, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
}

void gpl_u64_to_hex(uint64_t v, char *buf, size_t buflen) {
    snprintf(buf, buflen, "0x%llx", (unsigned long long)v);
}

int gpl_dvec_init(gpl_dvec *v, size_t initial_cap) {
    if (initial_cap == 0) initial_cap = 1024;
    v->data = (double *)malloc(sizeof(double) * initial_cap);
    if (!v->data) return -1;
    v->len = 0;
    v->cap = initial_cap;
    return 0;
}

int gpl_dvec_push(gpl_dvec *v, double x) {
    if (v->len == v->cap) {
        size_t nc = v->cap * 2;
        double *nd = (double *)realloc(v->data, sizeof(double) * nc);
        if (!nd) return -1;
        v->data = nd;
        v->cap = nc;
    }
    v->data[v->len++] = x;
    return 0;
}

void gpl_dvec_free(gpl_dvec *v) {
    free(v->data);
    v->data = NULL;
    v->len = v->cap = 0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

double gpl_dvec_percentile(gpl_dvec *v, double pct) {
    if (v->len == 0) return 0.0;
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    qsort(v->data, v->len, sizeof(double), cmp_double);
    /* Nearest rank */
    size_t idx = (size_t)((pct / 100.0) * (double)(v->len - 1) + 0.5);
    if (idx >= v->len) idx = v->len - 1;
    return v->data[idx];
}
