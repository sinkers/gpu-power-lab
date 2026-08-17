#include "args.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gpl_args_defaults(gpl_args_t *a) {
    a->op = GPL_OP_SGEMM;
    a->precision = GPL_PREC_FP32;
    a->size = 4096;
    a->streams = 1;
    a->warmup_sec = 5.0;
    a->steady_sec = 30.0;
    a->sample_hz = 100;
    a->device = 0;
    a->out_metrics = NULL;
    a->out_summary = NULL;
    a->rung_id = NULL;
    a->stop_on_throttle = false;
    a->mix_tensor = 1;
    a->mix_fma = 1;
    a->mix_dram = 1;
    a->duty_on_ms = 0.0;
    a->duty_off_ms = 0.0;
    a->iters = 0;
    a->raise_power_limit = false;
    a->probe = false;
}

const char *gpl_op_name(gpl_op_t op) {
    switch (op) {
        case GPL_OP_SGEMM:     return "sgemm";
        case GPL_OP_FFT:       return "fft";
        case GPL_OP_MEMSTREAM: return "memstream";
        case GPL_OP_POWERVIRUS: return "powervirus";
        default:               return "unknown";
    }
}

const char *gpl_prec_name(gpl_prec_t p) {
    switch (p) {
        case GPL_PREC_FP32: return "fp32";
        case GPL_PREC_TF32: return "tf32";
        case GPL_PREC_FP16: return "fp16";
        case GPL_PREC_BF16: return "bf16";
        case GPL_PREC_FP8:  return "fp8";
        default:            return "unknown";
    }
}

static int parse_op(const char *s, gpl_op_t *out) {
    if      (!strcmp(s, "sgemm"))     *out = GPL_OP_SGEMM;
    else if (!strcmp(s, "fft"))       *out = GPL_OP_FFT;
    else if (!strcmp(s, "memstream")) *out = GPL_OP_MEMSTREAM;
    else if (!strcmp(s, "powervirus")) *out = GPL_OP_POWERVIRUS;
    else return -1;
    return 0;
}

static int parse_prec(const char *s, gpl_prec_t *out) {
    if      (!strcmp(s, "fp32")) *out = GPL_PREC_FP32;
    else if (!strcmp(s, "tf32")) *out = GPL_PREC_TF32;
    else if (!strcmp(s, "fp16")) *out = GPL_PREC_FP16;
    else if (!strcmp(s, "bf16")) *out = GPL_PREC_BF16;
    else if (!strcmp(s, "fp8"))  *out = GPL_PREC_FP8;
    else return -1;
    return 0;
}

static void usage(FILE *f, const char *argv0) {
    fprintf(f,
        "Usage: %s [options]\n"
        "\n"
        "Workload:\n"
        "  --op OP                sgemm | fft | memstream | powervirus\n"
        "                                                           (default: sgemm)\n"
        "  --precision PREC       fp32 | tf32 | fp16 | bf16 | fp8   (default: fp32)\n"
        "  --size N               square matrix dimension           (default: 4096)\n"
        "  --streams N            number of CUDA streams            (default: 1)\n"
        "\n"
        "Powervirus mix (relative weights, 0 disables the unit):\n"
        "  --mix-tensor N         tensor-core warps                 (default: 1)\n"
        "  --mix-fma N            FP32 FFMA-chain warps             (default: 1)\n"
        "  --mix-dram N           DRAM-streaming warps              (default: 1)\n"
        "\n"
        "Timing:\n"
        "  --warmup-sec S         warm-up phase seconds             (default: 5)\n"
        "  --steady-sec S         steady-state (measured) seconds   (default: 30)\n"
        "  --sample-hz N          NVML sampling rate (Hz)           (default: 100)\n"
        "  --iters N              fixed-work mode: run exactly N steady\n"
        "                         iterations and measure elapsed time.\n"
        "                         Required for meaningful EDP / EDPp.\n"
        "  --duty-on-ms MS        O2: burst length at full blast    (0 = off)\n"
        "  --duty-off-ms MS       O2: idle gap between bursts\n"
        "\n"
        "Device / output:\n"
        "  --device N             CUDA device index                 (default: 0)\n"
        "  --out-metrics PATH     NDJSON per-sample telemetry file  (optional)\n"
        "  --out-summary PATH     summary JSON output               (default: stdout)\n"
        "  --rung-id STRING       label written into the summary    (auto if omitted)\n"
        "  --stop-on-throttle     abort steady phase on first throttle event\n"
        "  --raise-power-limit    set the enforced power limit to the device\n"
        "                         maximum before measuring (needs root)\n"
        "  --probe                report what this platform permits, then exit\n"
        "\n"
        "  -h, --help             show this help and exit\n",
        argv0);
}

int gpl_args_parse(int argc, char **argv, gpl_args_t *a) {
    gpl_args_defaults(a);

    static struct option long_opts[] = {
        {"op",                required_argument, 0, 'o'},
        {"precision",         required_argument, 0, 'p'},
        {"size",              required_argument, 0, 's'},
        {"streams",           required_argument, 0, 'S'},
        {"warmup-sec",        required_argument, 0, 'w'},
        {"steady-sec",        required_argument, 0, 't'},
        {"sample-hz",         required_argument, 0, 'H'},
        {"device",            required_argument, 0, 'd'},
        {"out-metrics",       required_argument, 0, 'M'},
        {"out-summary",       required_argument, 0, 'J'},
        {"rung-id",           required_argument, 0, 'r'},
        {"stop-on-throttle",  no_argument,       0, 'T'},
        {"mix-tensor",        required_argument, 0, 1001},
        {"mix-fma",           required_argument, 0, 1002},
        {"mix-dram",          required_argument, 0, 1003},
        {"duty-on-ms",        required_argument, 0, 1004},
        {"duty-off-ms",       required_argument, 0, 1005},
        {"iters",             required_argument, 0, 1006},
        {"raise-power-limit", no_argument,       0, 1007},
        {"probe",             no_argument,       0, 1008},
        {"help",              no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "h", long_opts, NULL)) != -1) {
        switch (c) {
            case 'o':
                if (parse_op(optarg, &a->op)) {
                    fprintf(stderr, "invalid --op: %s\n", optarg); return 2;
                }
                break;
            case 'p':
                if (parse_prec(optarg, &a->precision)) {
                    fprintf(stderr, "invalid --precision: %s\n", optarg); return 2;
                }
                break;
            case 's': a->size = atoi(optarg); break;
            case 'S': a->streams = atoi(optarg); break;
            case 'w': a->warmup_sec = atof(optarg); break;
            case 't': a->steady_sec = atof(optarg); break;
            case 'H': a->sample_hz = atoi(optarg); break;
            case 'd': a->device = atoi(optarg); break;
            case 'M': a->out_metrics = optarg; break;
            case 'J': a->out_summary = optarg; break;
            case 'r': a->rung_id = optarg; break;
            case 'T': a->stop_on_throttle = true; break;
            case 1001: a->mix_tensor = atoi(optarg); break;
            case 1002: a->mix_fma = atoi(optarg); break;
            case 1003: a->mix_dram = atoi(optarg); break;
            case 1004: a->duty_on_ms = atof(optarg); break;
            case 1005: a->duty_off_ms = atof(optarg); break;
            case 1006: a->iters = atoll(optarg); break;
            case 1007: a->raise_power_limit = true; break;
            case 1008: a->probe = true; break;
            case 'h': usage(stdout, argv[0]); return 1;
            default:  usage(stderr, argv[0]); return 2;
        }
    }

    if (a->probe) return 0;   /* probe ignores workload validation */

    if (a->size < 16) { fprintf(stderr, "--size too small\n"); return 2; }
    if (a->mix_tensor < 0 || a->mix_fma < 0 || a->mix_dram < 0) {
        fprintf(stderr, "--mix-* weights must be >= 0\n"); return 2;
    }
    if (a->op == GPL_OP_POWERVIRUS &&
        a->mix_tensor + a->mix_fma + a->mix_dram == 0) {
        fprintf(stderr, "powervirus needs at least one non-zero --mix-* weight\n");
        return 2;
    }
    if (a->duty_on_ms < 0 || a->duty_off_ms < 0) {
        fprintf(stderr, "--duty-*-ms must be >= 0\n"); return 2;
    }
    if (a->iters < 0) { fprintf(stderr, "--iters must be >= 0\n"); return 2; }
    if (a->streams < 1 || a->streams > 64) { fprintf(stderr, "--streams out of range\n"); return 2; }
    if (a->sample_hz < 1 || a->sample_hz > 1000) { fprintf(stderr, "--sample-hz out of range\n"); return 2; }
    if (a->steady_sec < 0.1) { fprintf(stderr, "--steady-sec too small\n"); return 2; }
    if (a->warmup_sec < 0) { fprintf(stderr, "--warmup-sec negative\n"); return 2; }

    return 0;
}
