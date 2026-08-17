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
    a->tensor_cublas = false;
    a->mix_tensor = 1;
    a->mix_fma = 1;
    a->mix_dram = 1;
    a->mix_sfu = 0;
    a->mix_int32 = 0;
    a->mix_smem = 0;
    a->mix_l2 = 0;
    a->mix_atomic = 0;
    a->duty_on_ms = 0.0;
    a->duty_off_ms = 0.0;
    a->duty_ramp = GPL_RAMP_NONE;
    a->duty_ramp_ms = 0.0;
    a->iters = 0;
    a->raise_power_limit = false;
    a->power_limit_w = 0.0;
    a->lock_clocks = false;
    a->probe = false;
    a->nccl_op = "allreduce";
    a->nccl_bytes = 0;
    a->nccl_devices = 0;
}

const char *gpl_op_name(gpl_op_t op) {
    switch (op) {
        case GPL_OP_SGEMM:     return "sgemm";
        case GPL_OP_FFT:       return "fft";
        case GPL_OP_MEMSTREAM: return "memstream";
        case GPL_OP_POWERVIRUS: return "powervirus";
        case GPL_OP_OBSERVE:    return "observe";
        case GPL_OP_NCCL:       return "nccl";
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
        case GPL_PREC_FP64: return "fp64";
        case GPL_PREC_INT8: return "int8";
        case GPL_PREC_FP4:  return "fp4";
        default:            return "unknown";
    }
}

static int parse_op(const char *s, gpl_op_t *out) {
    if      (!strcmp(s, "sgemm"))     *out = GPL_OP_SGEMM;
    else if (!strcmp(s, "fft"))       *out = GPL_OP_FFT;
    else if (!strcmp(s, "memstream")) *out = GPL_OP_MEMSTREAM;
    else if (!strcmp(s, "powervirus")) *out = GPL_OP_POWERVIRUS;
    else if (!strcmp(s, "observe"))    *out = GPL_OP_OBSERVE;
    else if (!strcmp(s, "nccl"))       *out = GPL_OP_NCCL;
    else return -1;
    return 0;
}

static int parse_prec(const char *s, gpl_prec_t *out) {
    if      (!strcmp(s, "fp32")) *out = GPL_PREC_FP32;
    else if (!strcmp(s, "tf32")) *out = GPL_PREC_TF32;
    else if (!strcmp(s, "fp16")) *out = GPL_PREC_FP16;
    else if (!strcmp(s, "bf16")) *out = GPL_PREC_BF16;
    else if (!strcmp(s, "fp8"))  *out = GPL_PREC_FP8;
    else if (!strcmp(s, "fp64")) *out = GPL_PREC_FP64;
    else if (!strcmp(s, "int8")) *out = GPL_PREC_INT8;
    else if (!strcmp(s, "fp4"))  *out = GPL_PREC_FP4;
    else return -1;
    return 0;
}

static void usage(FILE *f, const char *argv0) {
    fprintf(f,
        "Usage: %s [options]\n"
        "\n"
        "Workload:\n"
        "  --op OP                sgemm | fft | memstream | powervirus | observe\n"
        "                         | nccl\n""                         observe = sample only, drive nothing: for\n""                         measuring an external workload or the idle floor\n"
        "                                                           (default: sgemm)\n"
        "  --precision PREC       fp32 | tf32 | fp16 | bf16 | fp8 | fp64 | int8 | fp4\n"
        "                                                           (default: fp32)\n"
        "  --size N               square matrix dimension           (default: 4096)\n"
        "  --streams N            number of CUDA streams            (default: 1)\n"
        "\n"
        "Powervirus mix (relative weights, 0 disables the unit):\n"
        "  --mix-tensor N         tensor-core warps                 (default: 1)\n"
        "  --tensor-backend B     wmma | cublas                     (default: wmma)\n"
        "                         cublas runs a concurrent GEMM stream; on\n"
        "                         Blackwell that is the path to tcgen05.\n"
        "  --mix-fma N            FP32 FFMA-chain warps             (default: 1)\n"
        "  --mix-dram N           DRAM-streaming warps              (default: 1)\n"
        "  --mix-sfu N            transcendental (SFU) warps        (default: 0)\n"
        "  --mix-int32 N          integer-pipe warps                (default: 0)\n"
        "  --mix-smem N           shared-memory warps               (default: 0)\n"
        "  --mix-l2 N             L2-resident streaming warps       (default: 0)\n"
        "  --mix-atomic N         atomic (L2) warps                 (default: 0)\n"
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
        "  --duty-ramp SHAPE      none | linear | exp  (default: none)\n"
        "                         shape of the rising/falling edge. A square\n"
        "                         wave is the worst case upstream; a ramp is\n"
        "                         what a considerate scheduler would do.\n"
        "  --duty-ramp-ms MS      edge duration (default: a quarter of on-time)\n"
        "\n"
        "Device / output:\n"
        "  --device N             CUDA device index                 (default: 0)\n"
        "  --out-metrics PATH     NDJSON per-sample telemetry file  (optional)\n"
        "  --out-summary PATH     summary JSON output               (default: stdout)\n"
        "  --rung-id STRING       label written into the summary    (auto if omitted)\n"
        "  --stop-on-throttle     abort steady phase on first throttle event\n"
        "  --raise-power-limit    set the enforced power limit to the device\n"
        "                         maximum before measuring (needs root)\n"
        "  --power-limit W        set the enforced limit to W watts. Needed for\n"
        "                         a cap sweep: parts whose default already is\n"
        "                         the maximum have no headroom above, but the\n"
        "                         interesting curve is below.\n"
        "  --lock-clocks          pin the SM clock (boost-off, reproducible)\n"
        "  --probe                report what this platform permits, then exit\n"
        "\n"
        "Multi-GPU (--op nccl):\n"
        "  --nccl-op OP           allreduce | allgather | reducescatter |\n"
        "                         broadcast | alltoall | sendrecv\n"
        "  --nccl-bytes N         message size per rank    (default: 256 MiB)\n"
        "  --nccl-devices N       how many GPUs            (default: all)\n"
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
        {"tensor-backend",    required_argument, 0, 1009},
        {"mix-fma",           required_argument, 0, 1002},
        {"mix-dram",          required_argument, 0, 1003},
        {"mix-sfu",           required_argument, 0, 1012},
        {"mix-int32",         required_argument, 0, 1013},
        {"mix-smem",          required_argument, 0, 1014},
        {"mix-l2",            required_argument, 0, 1015},
        {"mix-atomic",        required_argument, 0, 1016},
        {"duty-on-ms",        required_argument, 0, 1004},
        {"duty-off-ms",       required_argument, 0, 1005},
        {"duty-ramp",         required_argument, 0, 1017},
        {"duty-ramp-ms",      required_argument, 0, 1018},
        {"iters",             required_argument, 0, 1006},
        {"raise-power-limit", no_argument,       0, 1007},
        {"power-limit",       required_argument, 0, 1010},
        {"lock-clocks",       no_argument,       0, 1011},
        {"probe",             no_argument,       0, 1008},
        {"nccl-op",           required_argument, 0, 1019},
        {"nccl-bytes",        required_argument, 0, 1020},
        {"nccl-devices",      required_argument, 0, 1021},
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
            case 1009:
                if      (!strcmp(optarg, "cublas")) a->tensor_cublas = true;
                else if (!strcmp(optarg, "wmma"))   a->tensor_cublas = false;
                else { fprintf(stderr, "invalid --tensor-backend: %s\n", optarg); return 2; }
                break;
            case 1002: a->mix_fma = atoi(optarg); break;
            case 1003: a->mix_dram = atoi(optarg); break;
            case 1012: a->mix_sfu = atoi(optarg); break;
            case 1013: a->mix_int32 = atoi(optarg); break;
            case 1014: a->mix_smem = atoi(optarg); break;
            case 1015: a->mix_l2 = atoi(optarg); break;
            case 1016: a->mix_atomic = atoi(optarg); break;
            case 1004: a->duty_on_ms = atof(optarg); break;
            case 1005: a->duty_off_ms = atof(optarg); break;
            case 1017:
                if      (!strcmp(optarg, "none"))   a->duty_ramp = GPL_RAMP_NONE;
                else if (!strcmp(optarg, "linear")) a->duty_ramp = GPL_RAMP_LINEAR;
                else if (!strcmp(optarg, "exp"))    a->duty_ramp = GPL_RAMP_EXP;
                else { fprintf(stderr, "invalid --duty-ramp: %s\n", optarg); return 2; }
                break;
            case 1018: a->duty_ramp_ms = atof(optarg); break;
            case 1006: a->iters = atoll(optarg); break;
            case 1007: a->raise_power_limit = true; break;
            case 1010: a->power_limit_w = atof(optarg); break;
            case 1011: a->lock_clocks = true; break;
            case 1008: a->probe = true; break;
            case 1019: a->nccl_op = optarg; break;
            case 1020: a->nccl_bytes = atoll(optarg); break;
            case 1021: a->nccl_devices = atoi(optarg); break;
            case 'h': usage(stdout, argv[0]); return 1;
            default:  usage(stderr, argv[0]); return 2;
        }
    }

    if (a->probe) return 0;   /* probe ignores workload validation */
    if (a->op == GPL_OP_OBSERVE) return 0;  /* nothing to configure */
    if (a->op == GPL_OP_NCCL) return 0;     /* validated in the workload */

    if (a->size < 16) { fprintf(stderr, "--size too small\n"); return 2; }
    if (a->mix_tensor < 0 || a->mix_fma < 0 || a->mix_dram < 0 ||
        a->mix_sfu < 0 || a->mix_int32 < 0 || a->mix_smem < 0 ||
        a->mix_l2 < 0 || a->mix_atomic < 0) {
        fprintf(stderr, "--mix-* weights must be >= 0\n"); return 2;
    }
    if (a->op == GPL_OP_POWERVIRUS &&
        a->mix_tensor + a->mix_fma + a->mix_dram + a->mix_sfu +
        a->mix_int32 + a->mix_smem + a->mix_l2 + a->mix_atomic == 0) {
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
