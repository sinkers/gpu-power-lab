/*
 * NCCL collectives — what the fabric costs, and what it does to the shape of
 * a node's power draw.
 *
 * Everything measured so far has been one GPU with an idle interconnect. Two
 * things only exist at node scale, and both matter more for the grid question
 * than anything a single card can show:
 *
 *   1. NVLink, the switches and the inter-die paths draw real power that a
 *      lone GPU cannot produce. DESIGN.md argues this is part of why a single
 *      B300 may be unable to reach its own board rating.
 *
 *   2. A collective is a *synchronised* event across every GPU in the node.
 *      All eight ramp together and idle together. That is precisely the
 *      in-phase swing we were planning to synthesise with duty cycling - and
 *      real training already does it, once per step, forever. A training
 *      cluster is a square-wave load generator that nobody designed as one.
 *
 * Single process, all devices, via ncclCommInitAll. Multi-process with a
 * shared unique ID would be more like a real job, but it puts a barrier and
 * an IPC path between us and the measurement for no benefit: NCCL itself
 * synchronises the ranks, and one process means one clock for all the
 * samplers.
 */

#include "workload.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(GPL_HAVE_NCCL)
#include <nccl.h>
#endif

#include "util.h"

#if !defined(GPL_HAVE_NCCL)

int gpl_workload_nccl_run(gpl_workload_ctx_t *ctx) {
    ctx->error = "built without NCCL: install libnccl-dev and rebuild";
    gpl_errf("%s", ctx->error);
    return -1;
}

#else

#define NCCL_OK(x) do {                                                    \
    ncclResult_t _r = (x);                                                 \
    if (_r != ncclSuccess) {                                               \
        ctx->error = ncclGetErrorString(_r);                               \
        gpl_errf("NCCL: %s (at %s:%d)", ctx->error, __FILE__, __LINE__);   \
        return -1;                                                         \
    }                                                                      \
} while (0)

#define CUDA_OK(x) do {                                                    \
    cudaError_t _e = (x);                                                  \
    if (_e != cudaSuccess) {                                               \
        ctx->error = cudaGetErrorString(_e);                               \
        gpl_errf("CUDA: %s (at %s:%d)", ctx->error, __FILE__, __LINE__);   \
        return -1;                                                         \
    }                                                                      \
} while (0)

/*
 * Each collective stresses the fabric differently, and the differences are
 * the point of sweeping them rather than just running all-reduce:
 *
 *   allreduce      2(N-1)/N bytes per rank on a ring. The training workhorse,
 *                  and the heaviest: every byte crosses the fabric twice.
 *   allgather      (N-1)/N per rank. FSDP parameter gathering.
 *   reducescatter  (N-1)/N per rank. The other half of FSDP.
 *   broadcast      One rank sends, everyone receives. Asymmetric, so it also
 *                  shows whether power follows the traffic or the topology.
 *   alltoall       Every rank to every rank. MoE routing, and the pattern
 *                  most likely to saturate the switch rather than the links.
 *   sendrecv       Neighbour pairs only. Pipeline parallelism, and the
 *                  cheapest way to light up a subset of the links.
 */
typedef enum {
    GPL_NCCL_ALLREDUCE = 0,
    GPL_NCCL_ALLGATHER,
    GPL_NCCL_REDUCESCATTER,
    GPL_NCCL_BROADCAST,
    GPL_NCCL_ALLTOALL,
    GPL_NCCL_SENDRECV,
} gpl_nccl_op_t;

static int parse_nccl_op(const char *s, gpl_nccl_op_t *out) {
    if      (!strcmp(s, "allreduce"))     *out = GPL_NCCL_ALLREDUCE;
    else if (!strcmp(s, "allgather"))     *out = GPL_NCCL_ALLGATHER;
    else if (!strcmp(s, "reducescatter")) *out = GPL_NCCL_REDUCESCATTER;
    else if (!strcmp(s, "broadcast"))     *out = GPL_NCCL_BROADCAST;
    else if (!strcmp(s, "alltoall"))      *out = GPL_NCCL_ALLTOALL;
    else if (!strcmp(s, "sendrecv"))      *out = GPL_NCCL_SENDRECV;
    else return -1;
    return 0;
}

/* Bytes actually moved across the fabric per rank, per call. Used for the
 * bus-bandwidth figure, which is what makes different collectives comparable
 * - message size alone is not, because a ring all-reduce moves roughly twice
 * what an all-gather of the same buffer does. */
static double fabric_bytes_per_rank(gpl_nccl_op_t op, size_t bytes, int n) {
    double N = (double)n;
    switch (op) {
        case GPL_NCCL_ALLREDUCE:     return 2.0 * bytes * (N - 1.0) / N;
        case GPL_NCCL_ALLGATHER:     return       bytes * (N - 1.0) / N;
        case GPL_NCCL_REDUCESCATTER: return       bytes * (N - 1.0) / N;
        case GPL_NCCL_BROADCAST:     return       bytes;
        case GPL_NCCL_ALLTOALL:      return       bytes * (N - 1.0) / N;
        case GPL_NCCL_SENDRECV:      return       bytes;
        default:                     return       bytes;
    }
}

static int run_one(gpl_workload_ctx_t *ctx, gpl_nccl_op_t op,
                   ncclComm_t *comms, cudaStream_t *streams,
                   void **sendbuf, void **recvbuf,
                   size_t count, int ndev) {
    NCCL_OK(ncclGroupStart());
    for (int i = 0; i < ndev; i++) {
        switch (op) {
            case GPL_NCCL_ALLREDUCE:
                NCCL_OK(ncclAllReduce(sendbuf[i], recvbuf[i], count,
                                      ncclFloat, ncclSum, comms[i], streams[i]));
                break;
            case GPL_NCCL_ALLGATHER:
                NCCL_OK(ncclAllGather(sendbuf[i], recvbuf[i], count / ndev,
                                      ncclFloat, comms[i], streams[i]));
                break;
            case GPL_NCCL_REDUCESCATTER:
                NCCL_OK(ncclReduceScatter(sendbuf[i], recvbuf[i], count / ndev,
                                          ncclFloat, ncclSum, comms[i], streams[i]));
                break;
            case GPL_NCCL_BROADCAST:
                NCCL_OK(ncclBroadcast(sendbuf[i], recvbuf[i], count,
                                      ncclFloat, 0, comms[i], streams[i]));
                break;
            case GPL_NCCL_ALLTOALL: {
                /* No ncclAllToAll in the API; it is a group of send/recv. */
                size_t chunk = count / ndev;
                for (int r = 0; r < ndev; r++) {
                    NCCL_OK(ncclSend((char *)sendbuf[i] + r * chunk * sizeof(float),
                                     chunk, ncclFloat, r, comms[i], streams[i]));
                    NCCL_OK(ncclRecv((char *)recvbuf[i] + r * chunk * sizeof(float),
                                     chunk, ncclFloat, r, comms[i], streams[i]));
                }
                break;
            }
            case GPL_NCCL_SENDRECV: {
                int peer = (i % 2 == 0) ? (i + 1) % ndev : (i - 1 + ndev) % ndev;
                NCCL_OK(ncclSend(sendbuf[i], count, ncclFloat, peer, comms[i], streams[i]));
                NCCL_OK(ncclRecv(recvbuf[i], count, ncclFloat, peer, comms[i], streams[i]));
                break;
            }
        }
    }
    NCCL_OK(ncclGroupEnd());
    return 0;
}

int gpl_workload_nccl_run(gpl_workload_ctx_t *ctx) {
    const gpl_args_t *a = ctx->args;

    gpl_nccl_op_t op;
    if (parse_nccl_op(a->nccl_op ? a->nccl_op : "allreduce", &op) != 0) {
        ctx->error = "invalid --nccl-op";
        gpl_errf("%s: %s", ctx->error, a->nccl_op);
        return -1;
    }

    int avail = 0;
    CUDA_OK(cudaGetDeviceCount(&avail));
    int ndev = a->nccl_devices > 0 ? a->nccl_devices : avail;
    if (ndev > avail) ndev = avail;
    if (ndev < 2) {
        ctx->error = "NCCL rungs need at least 2 GPUs";
        gpl_errf("%s (found %d)", ctx->error, avail);
        return -1;
    }

    size_t bytes = a->nccl_bytes > 0 ? (size_t)a->nccl_bytes : (256u << 20);
    size_t count = bytes / sizeof(float);
    /* All-gather and reduce-scatter divide by rank count; keep it exact so
     * the byte figures in the summary are not quietly off by a remainder. */
    count = (count / ndev) * ndev;
    bytes = count * sizeof(float);

    int *devs = (int *)calloc(ndev, sizeof(int));
    for (int i = 0; i < ndev; i++) devs[i] = i;

    ncclComm_t   *comms   = (ncclComm_t *)calloc(ndev, sizeof(ncclComm_t));
    cudaStream_t *streams = (cudaStream_t *)calloc(ndev, sizeof(cudaStream_t));
    void **sendbuf = (void **)calloc(ndev, sizeof(void *));
    void **recvbuf = (void **)calloc(ndev, sizeof(void *));

    /* All-gather's receive buffer is ndev times the send buffer; size every
     * receive buffer that way so one allocation path covers all ops. */
    for (int i = 0; i < ndev; i++) {
        CUDA_OK(cudaSetDevice(i));
        CUDA_OK(cudaStreamCreate(&streams[i]));
        CUDA_OK(cudaMalloc(&sendbuf[i], bytes));
        CUDA_OK(cudaMalloc(&recvbuf[i], bytes * (size_t)ndev));
        CUDA_OK(cudaMemset(sendbuf[i], 0x3c, bytes));
        CUDA_OK(cudaMemset(recvbuf[i], 0, bytes * (size_t)ndev));
    }

    NCCL_OK(ncclCommInitAll(comms, ndev, devs));

    double fb = fabric_bytes_per_rank(op, bytes, ndev);
    gpl_logf("nccl: op=%s devices=%d message=%.1f MiB "
             "fabric=%.1f MiB/rank/call",
             a->nccl_op ? a->nccl_op : "allreduce", ndev,
             bytes / 1048576.0, fb / 1048576.0);

    /* ---- warmup ---- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_WARMUP);
    uint64_t t0 = gpl_mono_ns();
    while (gpl_mono_ns() - t0 < (uint64_t)(a->warmup_sec * 1e9)) {
        if (run_one(ctx, op, comms, streams, sendbuf, recvbuf, count, ndev) != 0) return -1;
        for (int i = 0; i < ndev; i++) {
            CUDA_OK(cudaSetDevice(i));
            CUDA_OK(cudaStreamSynchronize(streams[i]));
        }
        ctx->iterations_warmup++;
    }

    /* ---- steady ---- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_STEADY);
    uint64_t s0 = gpl_mono_ns();
    uint64_t deadline = s0 + (uint64_t)(a->steady_sec * 1e9);
    const bool fixed_work = (a->iters > 0);
    const bool duty = (a->duty_on_ms > 0.0);
    uint64_t burst_ns = (uint64_t)(a->duty_on_ms * 1e6);

    while (1) {
        if (fixed_work) { if ((long long)ctx->iterations_steady >= a->iters) break; }
        else            { if (gpl_mono_ns() >= deadline) break; }

        if (duty) {
            /* Duty-cycled collectives. This is the closest synthetic analogue
             * to a training job: a burst of fabric traffic, then a gap, over
             * and over, with every rank in lockstep. */
            uint64_t b0 = gpl_mono_ns();
            while (gpl_mono_ns() - b0 < burst_ns) {
                if (run_one(ctx, op, comms, streams, sendbuf, recvbuf, count, ndev) != 0) return -1;
                for (int i = 0; i < ndev; i++) {
                    CUDA_OK(cudaSetDevice(i));
                    CUDA_OK(cudaStreamSynchronize(streams[i]));
                }
                ctx->iterations_steady++;
            }
            if (a->duty_off_ms > 0) {
                struct timespec ts;
                ts.tv_sec  = (time_t)(a->duty_off_ms / 1000.0);
                ts.tv_nsec = (long)((a->duty_off_ms - ts.tv_sec * 1000.0) * 1e6);
                nanosleep(&ts, NULL);
            }
        } else {
            if (run_one(ctx, op, comms, streams, sendbuf, recvbuf, count, ndev) != 0) return -1;
            for (int i = 0; i < ndev; i++) {
                CUDA_OK(cudaSetDevice(i));
                CUDA_OK(cudaStreamSynchronize(streams[i]));
            }
            ctx->iterations_steady++;
        }
    }
    ctx->compute_seconds_steady = (double)(gpl_mono_ns() - s0) / 1e9;

    /* Bus bandwidth, the figure that makes collectives comparable. */
    if (ctx->compute_seconds_steady > 0.0 && ctx->iterations_steady > 0) {
        double gbs = fb * (double)ctx->iterations_steady
                   / ctx->compute_seconds_steady / 1e9;
        gpl_logf("nccl: %llu calls in %.2fs = %.1f GB/s per rank across the fabric",
                 (unsigned long long)ctx->iterations_steady,
                 ctx->compute_seconds_steady, gbs);
    }

    for (int i = 0; i < ndev; i++) {
        ncclCommDestroy(comms[i]);
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
        cudaFree(sendbuf[i]);
        cudaFree(recvbuf[i]);
    }
    free(comms); free(streams); free(sendbuf); free(recvbuf); free(devs);
    return 0;
}

#endif /* GPL_HAVE_NCCL */
