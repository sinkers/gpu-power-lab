/*
 * powervirus — mixed-unit maximum-power workload.
 *
 * The hypothesis (DESIGN.md, "Reaching the ceiling"): peak-FLOPS GEMM is a
 * narrow power profile because it saturates one pipe and leaves the FP32
 * units, the register file and DRAM comparatively quiet. Driving several
 * units concurrently should draw more total power than driving any one of
 * them at peak.
 *
 * Structure: warp specialization inside each block. Warps are assigned a
 * role by the --mix-* weights and never diverge from it, so the tensor
 * cores, the FFMA pipe and the memory path are all busy in the same cycles
 * on the same SM. This is a better shot at real concurrency than
 * interleaving instruction types in a single stream, where issue bandwidth
 * would serialize them.
 *
 * On Blackwell (sm_100/sm_103) the tensor role should eventually issue
 * tcgen05.mma, which is asynchronous and single-thread-issued (verified —
 * see docs/blackwell-cuda-notes.md §3), freeing even more issue bandwidth
 * for the other roles. This file is the portable wmma backend that runs
 * everywhere from Volta up, including Blackwell; the tcgen05 backend is
 * separate and selected at runtime.
 *
 * Nothing here checks numerical results. Buffers are filled with a bit
 * pattern and the accumulators are consumed by an impossible branch so the
 * optimizer cannot delete the loop. This is a power harness.
 */

#include "workload.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 9)
#include <mma.h>
#endif

#include "util.h"

#define CUDA_OK(x) do {                                              \
    cudaError_t _e = (x);                                            \
    if (_e != cudaSuccess) {                                         \
        ctx->error = cudaGetErrorString(_e);                         \
        gpl_errf("CUDA: %s (at %s:%d)", ctx->error, __FILE__, __LINE__); \
        return -1;                                                   \
    }                                                                \
} while (0)

/* Warp roles. Assigned per warp at block entry and fixed for the launch. */
#define GPL_ROLE_TENSOR 0
#define GPL_ROLE_FMA    1
#define GPL_ROLE_DRAM   2

#define GPL_WARP_SIZE   32
#define GPL_BLOCK_WARPS 8
#define GPL_BLOCK_DIM   (GPL_WARP_SIZE * GPL_BLOCK_WARPS)

/*
 * One launch runs `inner` iterations of whichever role this warp owns.
 *
 * role_map is a 32-entry lookup: role_map[warp_id % 32]. The host builds it
 * from the mix weights, so a mix of 2:1:1 gives half the warps to tensor,
 * a quarter to FMA and a quarter to DRAM.
 */
__global__ __launch_bounds__(GPL_BLOCK_DIM)
void gpl_powervirus_kernel(const unsigned char *__restrict__ role_map,
                           const float4 *__restrict__ stream_buf,
                           size_t        stream_elems,
                           int           inner,
                           float        *__restrict__ sink)
{
    const int warp_in_block = threadIdx.x / GPL_WARP_SIZE;
    const int lane          = threadIdx.x % GPL_WARP_SIZE;
    const int global_warp   = blockIdx.x * GPL_BLOCK_WARPS + warp_in_block;
    const unsigned char role = role_map[global_warp % 32];

    float keep = 0.0f;

    if (role == GPL_ROLE_TENSOR) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        using namespace nvcuda::wmma;
        fragment<matrix_a, 16, 16, 16, __half, row_major> a0, a1;
        fragment<matrix_b, 16, 16, 16, __half, col_major> b0, b1;
        fragment<accumulator, 16, 16, 16, float> c0, c1;

        /* Register-resident operands. 0x3c00 is half(1.0); we want the
         * tensor cores busy, not a correct product. */
        fill_fragment(a0, __float2half(1.0f));
        fill_fragment(a1, __float2half(0.99f));
        fill_fragment(b0, __float2half(1.0f));
        fill_fragment(b1, __float2half(0.98f));
        fill_fragment(c0, 0.0f);
        fill_fragment(c1, 0.0f);

        /* Two independent accumulator chains so the MMA pipe is not stalled
         * waiting on a single dependency chain. */
        for (int i = 0; i < inner; i++) {
            mma_sync(c0, a0, b0, c0);
            mma_sync(c1, a1, b1, c1);
            mma_sync(c0, a1, b0, c0);
            mma_sync(c1, a0, b1, c1);
        }
        keep = c0.x[0] + c1.x[0];
#else
        keep = 0.0f;
#endif
    } else if (role == GPL_ROLE_FMA) {
        /* Eight independent FFMA chains: enough ILP to keep the FP32 pipe
         * issuing every cycle without spilling. Deliberately high register
         * pressure, but well inside the budget — a spill would turn this
         * into a memory workload and the T0 gate fails the build on one. */
        float x0 = 1.0001f + lane, x1 = 1.0002f, x2 = 1.0003f, x3 = 1.0004f;
        float x4 = 1.0005f, x5 = 1.0006f, x6 = 1.0007f, x7 = 1.0008f;
        const float m = 0.9999f, c = 1e-7f;
        for (int i = 0; i < inner * 16; i++) {
            x0 = fmaf(x0, m, c); x1 = fmaf(x1, m, c);
            x2 = fmaf(x2, m, c); x3 = fmaf(x3, m, c);
            x4 = fmaf(x4, m, c); x5 = fmaf(x5, m, c);
            x6 = fmaf(x6, m, c); x7 = fmaf(x7, m, c);
        }
        keep = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    } else {
        /* DRAM: 128-bit loads striding far enough to miss L2 on every
         * access. The stride is a large odd multiple of the warp width so
         * consecutive warps never share a cache line. */
        const size_t stride = 65537;  /* prime, ~1MB apart at 16B/elem */
        size_t idx = ((size_t)global_warp * 2654435761u + lane) % stream_elems;
        float acc = 0.0f;
        for (int i = 0; i < inner * 4; i++) {
            float4 v = stream_buf[idx];
            acc += v.x + v.y + v.z + v.w;
            idx += stride;
            if (idx >= stream_elems) idx -= stream_elems;
        }
        keep = acc;
    }

    /* Impossible branch: keeps the optimizer honest without ever writing. */
    if (keep == 1.2345678e-30f) sink[0] = keep;
}

/* Build the 32-entry role map from the mix weights. */
static void build_role_map(const gpl_args_t *a, unsigned char map[32]) {
    int wt = a->mix_tensor, wf = a->mix_fma, wd = a->mix_dram;
    int total = wt + wf + wd;
    if (total <= 0) { wt = 1; total = 1; wf = wd = 0; }

    int n_t = (32 * wt) / total;
    int n_f = (32 * wf) / total;
    /* Remainder goes to DRAM so the map is always exactly 32 entries. */
    int i = 0;
    for (int k = 0; k < n_t; k++) map[i++] = GPL_ROLE_TENSOR;
    for (int k = 0; k < n_f; k++) map[i++] = GPL_ROLE_FMA;
    while (i < 32) map[i++] = (wd > 0) ? GPL_ROLE_DRAM
                            : (wf > 0 ? GPL_ROLE_FMA : GPL_ROLE_TENSOR);
}

static void sleep_ms(double ms) {
    if (ms <= 0) return;
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000.0);
    ts.tv_nsec = (long)((ms - ts.tv_sec * 1000.0) * 1e6);
    nanosleep(&ts, NULL);
}

int gpl_workload_powervirus_run(gpl_workload_ctx_t *ctx) {
    const gpl_args_t *a = ctx->args;

    int dev = a->device;
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));

    /* Fill every SM several times over so there is no idle partition. */
    const int blocks = prop.multiProcessorCount * 4;

    /* Streaming buffer: big enough to miss L2 comfortably. Cap at a
     * fraction of free memory so we coexist with anything else resident. */
    size_t freeb = 0, totalb = 0;
    CUDA_OK(cudaMemGetInfo(&freeb, &totalb));
    size_t buf_bytes = freeb / 4;
    if (buf_bytes > (size_t)2 << 30) buf_bytes = (size_t)2 << 30;
    if (buf_bytes < (size_t)64 << 20) buf_bytes = (size_t)64 << 20;
    buf_bytes &= ~(size_t)0xFFF;
    size_t stream_elems = buf_bytes / sizeof(float4);

    float4 *d_stream = NULL;
    unsigned char *d_role = NULL;
    float *d_sink = NULL;
    CUDA_OK(cudaMalloc((void **)&d_stream, buf_bytes));
    CUDA_OK(cudaMalloc((void **)&d_role, 32));
    CUDA_OK(cudaMalloc((void **)&d_sink, sizeof(float)));
    CUDA_OK(cudaMemset(d_stream, 0x3c, buf_bytes));
    CUDA_OK(cudaMemset(d_sink, 0, sizeof(float)));

    unsigned char role_map[32];
    build_role_map(a, role_map);
    CUDA_OK(cudaMemcpy(d_role, role_map, 32, cudaMemcpyHostToDevice));

    int n_t = 0, n_f = 0, n_d = 0;
    for (int i = 0; i < 32; i++) {
        if (role_map[i] == GPL_ROLE_TENSOR) n_t++;
        else if (role_map[i] == GPL_ROLE_FMA) n_f++;
        else n_d++;
    }
    gpl_logf("powervirus: %d blocks x %d warps | warp mix tensor=%d fma=%d dram=%d /32 | stream=%.0f MiB",
             blocks, GPL_BLOCK_WARPS, n_t, n_f, n_d, buf_bytes / 1048576.0);

    int nstreams = a->streams;
    cudaStream_t *streams = (cudaStream_t *)calloc(nstreams, sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; i++) {
        CUDA_OK(cudaStreamCreate(&streams[i]));
    }

    /* `inner` sets the granularity of one iteration. Keep a launch short
     * enough that duty cycling has millisecond resolution; the host loop
     * supplies the volume. */
    const int inner = a->size >= 4096 ? 2048 : 512;

    /* Queue several launches per stream before synchronizing. A sync after
     * every launch leaves the GPU idle for the round-trip, which shows up
     * directly as lost power — the first run on an A10G sat at 48% of TDP
     * largely because of this. */
    const int batch = 8;

    const bool duty = (a->duty_on_ms > 0.0);
    const bool fixed_work = (a->iters > 0);

    /* ---- warmup ---- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_WARMUP);
    uint64_t t_start = gpl_mono_ns();
    uint64_t warmup_ns = (uint64_t)(a->warmup_sec * 1e9);
    while (gpl_mono_ns() - t_start < warmup_ns) {
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < nstreams; s++)
                gpl_powervirus_kernel<<<blocks, GPL_BLOCK_DIM, 0, streams[s]>>>(
                    d_role, d_stream, stream_elems, inner, d_sink);
        CUDA_OK(cudaDeviceSynchronize());
        ctx->iterations_warmup += (uint64_t)nstreams * batch;
    }

    /* ---- steady ---- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_STEADY);
    uint64_t s_start = gpl_mono_ns();
    uint64_t steady_ns = (uint64_t)(a->steady_sec * 1e9);
    uint64_t burst_ns = (uint64_t)(a->duty_on_ms * 1e6);

    while (1) {
        if (fixed_work) {
            if ((long long)ctx->iterations_steady >= a->iters) break;
        } else {
            if (gpl_mono_ns() - s_start >= steady_ns) break;
        }
        if (a->stop_on_throttle && gpl_telemetry_throttled_now(ctx->tele)) {
            ctx->aborted_on_throttle = true;
            break;
        }

        if (duty) {
            /* Burst at full blast for duty_on_ms, then a real idle gap.
             * The gap must be genuinely idle — the swing we are measuring
             * is from the idle floor, so spinning here would destroy the
             * measurement. */
            uint64_t b0 = gpl_mono_ns();
            while (gpl_mono_ns() - b0 < burst_ns) {
                for (int s = 0; s < nstreams; s++)
                    gpl_powervirus_kernel<<<blocks, GPL_BLOCK_DIM, 0, streams[s]>>>(
                        d_role, d_stream, stream_elems, inner, d_sink);
                CUDA_OK(cudaDeviceSynchronize());
                ctx->iterations_steady += nstreams;
            }
            sleep_ms(a->duty_off_ms);
        } else {
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < nstreams; s++)
                    gpl_powervirus_kernel<<<blocks, GPL_BLOCK_DIM, 0, streams[s]>>>(
                        d_role, d_stream, stream_elems, inner, d_sink);
            CUDA_OK(cudaDeviceSynchronize());
            ctx->iterations_steady += (uint64_t)nstreams * batch;
        }
    }
    ctx->compute_seconds_steady = (double)(gpl_mono_ns() - s_start) / 1e9;

    for (int i = 0; i < nstreams; i++) cudaStreamDestroy(streams[i]);
    free(streams);
    cudaFree(d_stream);
    cudaFree(d_role);
    cudaFree(d_sink);
    return 0;
}
