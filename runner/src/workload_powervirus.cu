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

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 9)
#include <mma.h>
#endif

#include "gemm_lt.h"
#include "util.h"

#define CUDA_OK(x) do {                                              \
    cudaError_t _e = (x);                                            \
    if (_e != cudaSuccess) {                                         \
        ctx->error = cudaGetErrorString(_e);                         \
        gpl_errf("CUDA: %s (at %s:%d)", ctx->error, __FILE__, __LINE__); \
        return -1;                                                   \
    }                                                                \
} while (0)

/* Warp roles. Assigned per warp at block entry and fixed for the launch.
 *
 * These are the single-unit rungs of the attribution ladder in DESIGN.md:
 * each one is meant to exercise one part of the chip as exclusively as the
 * ISA allows, so that the marginal watts of that unit can be read off, and
 * pairs of them can be run together to expose the interaction term. */
#define GPL_ROLE_TENSOR 0   /* tensor cores (wmma; cuBLAS path is separate) */
#define GPL_ROLE_FMA    1   /* FP32 FFMA pipe */
#define GPL_ROLE_DRAM   2   /* HBM, L2-missing */
#define GPL_ROLE_SFU    3   /* special function unit: transcendentals */
#define GPL_ROLE_INT32  4   /* integer pipe */
#define GPL_ROLE_SMEM   5   /* shared memory / L1 */
#define GPL_ROLE_L2     6   /* L2-resident streaming (contrast with DRAM) */
#define GPL_ROLE_ATOMIC 7   /* atomics, which land in L2 */
#define GPL_ROLE_COUNT  8

/* Shared-memory scratch for the SMEM role. 64 floats per warp is enough to
 * keep the LSU busy without limiting occupancy. */
#define GPL_SMEM_PER_WARP 64

/* L2 window: small enough to stay resident on any Blackwell-class L2, large
 * enough that the accesses are not just L1 hits. */
#define GPL_L2_WINDOW_BYTES (4u << 20)

/* Atomic fan-out. Power of two so the mask below is cheap. */
#define GPL_ATOMIC_SLOTS 1024

#define GPL_WARP_SIZE   32
#define GPL_BLOCK_WARPS 8
#define GPL_BLOCK_DIM   (GPL_WARP_SIZE * GPL_BLOCK_WARPS)

/*
 * One launch runs `inner` iterations of whichever role this warp owns.
 *
 * role_map is a lookup of GPL_ROLE_MAP_SLOTS entries indexed by warp id. The
 * host builds it from the mix weights by largest-remainder apportionment, so
 * a mix of 2:1:1 gives half the warps to tensor and a quarter each to FMA and
 * DRAM, and a role with a non-zero weight never rounds away to nothing.
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
    const unsigned char role = role_map[global_warp % GPL_ROLE_MAP_SLOTS];

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
    } else if (role == GPL_ROLE_DRAM) {
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

    } else if (role == GPL_ROLE_L2) {
        /* Same access pattern as DRAM but confined to a window small enough
         * to stay resident in L2. The DRAM/L2 pair is the interesting one:
         * the difference between them is what the HBM stacks and the memory
         * controllers cost, separated from the cache hierarchy. */
        const size_t win = (GPL_L2_WINDOW_BYTES / sizeof(float4));
        const size_t base = ((size_t)blockIdx.x * 977) % (stream_elems > win ? stream_elems - win : 1);
        size_t idx = lane;
        float acc = 0.0f;
        for (int i = 0; i < inner * 4; i++) {
            float4 v = stream_buf[base + (idx % win)];
            acc += v.x + v.y + v.z + v.w;
            idx += 129;                     /* small odd stride, stays in-window */
        }
        keep = acc;

    } else if (role == GPL_ROLE_SFU) {
        /* Transcendentals. The fast-math intrinsics map to the SFU, which is
         * a physically separate unit from both the FMA pipe and the tensor
         * cores - so this rung is the one that says whether the SFU is worth
         * anything at all in a power mix. Four independent chains for ILP. */
        float a0 = 0.7f + lane * 0.01f, a1 = 1.1f, a2 = 0.3f, a3 = 1.7f;
        for (int i = 0; i < inner * 4; i++) {
            a0 = __sinf(a0) + 1.0f;
            a1 = __expf(a1 * 0.5f) * 0.5f;
            a2 = __logf(a2 + 2.0f);
            a3 = __sqrtf(a3 + 1.0f);
        }
        keep = a0 + a1 + a2 + a3;

    } else if (role == GPL_ROLE_INT32) {
        /* Integer multiply-add chains. A separate issue path to FP32 on most
         * NVIDIA parts, so in principle it can overlap FMA rather than
         * compete with it - which is exactly the kind of pairing the
         * interaction terms are meant to detect. */
        unsigned int i0 = 0x9E3779B9u + lane, i1 = 0x85EBCA6Bu;
        unsigned int i2 = 0xC2B2AE35u, i3 = 0x27D4EB2Fu;
        for (int i = 0; i < inner * 16; i++) {
            i0 = i0 * 1664525u + 1013904223u;
            i1 = i1 * 22695477u + 1u;
            i2 = i2 * 1103515245u + 12345u;
            i3 = i3 * 214013u + 2531011u;
        }
        keep = (float)((i0 ^ i1 ^ i2 ^ i3) & 0xFFFF);

    } else if (role == GPL_ROLE_SMEM) {
        /* Shared memory traffic. Bank-conflict-free by construction (stride
         * of 1 across lanes), so this measures LSU and SRAM activity rather
         * than serialisation. */
        extern __shared__ float smem[];
        float *mine = smem + warp_in_block * GPL_SMEM_PER_WARP;
        mine[lane] = 1.0f + lane;
        mine[lane + 32] = 2.0f;
        __syncwarp();
        float acc = 0.0f;
        for (int i = 0; i < inner * 8; i++) {
            int j = (lane + i) & 31;
            acc += mine[j] * 1.000001f;
            mine[j + 32] = acc;
        }
        keep = acc;

    } else {
        /* Atomics. These resolve in L2, so the rung exercises the crossbar
         * and the L2 atomic units rather than the SMs. Deliberately spread
         * across a handful of addresses: all-lanes-one-address would measure
         * serialisation instead of throughput. */
        float acc = 0.0f;
        const int slot = (global_warp * 32 + lane) & (GPL_ATOMIC_SLOTS - 1);
        for (int i = 0; i < inner; i++) {
            acc += atomicAdd(&sink[1 + slot], 1.0f);
        }
        keep = acc * 1e-30f;
    }

    /* Impossible branch: keeps the optimizer honest without ever writing. */
    if (keep == 1.2345678e-30f) sink[0] = keep;
}

/*
 * Concurrent cuBLAS tensor stream.
 *
 * The wmma role above is warp-synchronous: it consumes warp slots, so on
 * every architecture it competes with the FMA and DRAM roles for one fixed
 * budget. That is what made mixing lose to pure FFMA on A10G.
 *
 * tcgen05 on Blackwell is different — asynchronous, issued by a single
 * thread (PTX ISA 9.7.17 Table 49) — so tensor work there need not displace
 * anything. Reaching it without hand-writing PTX means letting cuBLAS emit
 * it: run a GEMM loop on its own stream, concurrently with the FFMA/DRAM
 * kernel on other streams, and let the block scheduler interleave them.
 *
 * Whether the two genuinely overlap is exactly the O1 question. Comparing
 * --tensor-backend wmma against cublas on the same silicon isolates it.
 */
typedef struct {
    /* Narrow formats (FP8, FP4) do not exist in cublasGemmEx, so the tensor
     * stream carries both paths and picks per precision. */
    gpl_lt_gemm_t  lt;
    bool           use_lt;
    cublasHandle_t h;
    cudaStream_t   stream;
    void          *dA, *dB, *dC;
    int            n;
    cudaDataType   in_t, out_t;
    cublasComputeType_t comp_t;
    bool           active;
} gpl_tensor_stream_t;

static int tensor_stream_init(gpl_workload_ctx_t *ctx, gpl_tensor_stream_t *ts) {
    const gpl_args_t *a = ctx->args;
    memset(ts, 0, sizeof(*ts));

    /* Square GEMM sized to stay resident; big enough that launch overhead
     * is negligible against the MMA work. */
    ts->n = a->size > 0 ? a->size : 4096;

    if (gpl_lt_precision_supported(a->precision)) {
        if (cudaStreamCreate(&ts->stream) != cudaSuccess) {
            ctx->error = "tensor stream create failed";
            return -1;
        }
        if (gpl_lt_init(&ts->lt, a->precision, ts->n, ts->stream) != 0) {
            /* Not supported on this part, or the layout convention is wrong.
             * Either way this is a real "no result" for that rung, not a
             * reason to silently fall back to a wider format and report a
             * number under the wrong label. */
            ctx->error = "requested precision unavailable via cublasLt on this device";
            return -1;
        }
        ts->use_lt = true;
        ts->active = true;
        return 0;
    }

    switch (a->precision) {
        case GPL_PREC_FP32:
            ts->in_t = CUDA_R_32F; ts->out_t = CUDA_R_32F;
            ts->comp_t = CUBLAS_COMPUTE_32F; break;
        case GPL_PREC_TF32:
            ts->in_t = CUDA_R_32F; ts->out_t = CUDA_R_32F;
            ts->comp_t = CUBLAS_COMPUTE_32F_FAST_TF32; break;
        case GPL_PREC_BF16:
            ts->in_t = CUDA_R_16BF; ts->out_t = CUDA_R_16BF;
            ts->comp_t = CUBLAS_COMPUTE_32F; break;
        case GPL_PREC_FP64:
            ts->in_t = CUDA_R_64F; ts->out_t = CUDA_R_64F;
            ts->comp_t = CUBLAS_COMPUTE_64F; break;
        case GPL_PREC_INT8:
            ts->in_t = CUDA_R_8I; ts->out_t = CUDA_R_32I;
            ts->comp_t = CUBLAS_COMPUTE_32I; break;
        case GPL_PREC_FP16:
        default:
            ts->in_t = CUDA_R_16F; ts->out_t = CUDA_R_16F;
            ts->comp_t = CUBLAS_COMPUTE_16F; break;
    }

    size_t esz = (ts->in_t == CUDA_R_64F) ? 8
               : (ts->in_t == CUDA_R_32F) ? 4
               : (ts->in_t == CUDA_R_8I)  ? 1 : 2;
    size_t bytes = (size_t)ts->n * ts->n * esz;

    if (cudaMalloc(&ts->dA, bytes) != cudaSuccess ||
        cudaMalloc(&ts->dB, bytes) != cudaSuccess ||
        cudaMalloc(&ts->dC, (size_t)ts->n * ts->n *
                   ((ts->out_t == CUDA_R_64F) ? 8 :
                    (ts->out_t == CUDA_R_32F || ts->out_t == CUDA_R_32I) ? 4 : 2))
            != cudaSuccess) {
        ctx->error = "tensor stream: out of memory";
        return -1;
    }
    /* 0x3c is half(~1.0); values are never checked. */
    cudaMemset(ts->dA, 0x3c, bytes);
    cudaMemset(ts->dB, 0x3c, bytes);

    if (cublasCreate(&ts->h) != CUBLAS_STATUS_SUCCESS) {
        ctx->error = "cublasCreate failed";
        return -1;
    }
    if (cudaStreamCreate(&ts->stream) != cudaSuccess) {
        ctx->error = "tensor stream create failed";
        return -1;
    }
    cublasSetStream(ts->h, ts->stream);
    cublasSetMathMode(ts->h, CUBLAS_TENSOR_OP_MATH);
    ts->active = true;
    return 0;
}

/* Enqueue `count` GEMMs on the tensor stream. Non-blocking: the whole point
 * is that these run alongside the FFMA/DRAM kernel. */
static void tensor_stream_enqueue(gpl_tensor_stream_t *ts, int count) {
    if (!ts->active) return;
    if (ts->use_lt) { gpl_lt_enqueue(&ts->lt, count); return; }
    float  alpha_f = 1.0f, beta_f = 1.0f;      /* beta=1 keeps C live */
    double alpha_d = 1.0,  beta_d = 1.0;
    int    alpha_i = 1,    beta_i = 1;
    unsigned short alpha_h = 0x3C00, beta_h = 0x3C00;
    const void *alpha = &alpha_f, *beta = &beta_f;
    if      (ts->comp_t == CUBLAS_COMPUTE_16F) { alpha = &alpha_h; beta = &beta_h; }
    else if (ts->comp_t == CUBLAS_COMPUTE_64F) { alpha = &alpha_d; beta = &beta_d; }
    else if (ts->comp_t == CUBLAS_COMPUTE_32I) { alpha = &alpha_i; beta = &beta_i; }

    for (int i = 0; i < count; i++) {
        cublasGemmEx(ts->h, CUBLAS_OP_N, CUBLAS_OP_N,
                     ts->n, ts->n, ts->n,
                     alpha, ts->dA, ts->in_t, ts->n,
                            ts->dB, ts->in_t, ts->n,
                     beta,  ts->dC, ts->out_t, ts->n,
                     ts->comp_t, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

static void tensor_stream_free(gpl_tensor_stream_t *ts) {
    if (!ts->active) return;
    if (ts->use_lt) {
        gpl_lt_free(&ts->lt);
        cudaStreamDestroy(ts->stream);
        ts->active = false;
        return;
    }
    cublasDestroy(ts->h);
    cudaStreamDestroy(ts->stream);
    cudaFree(ts->dA); cudaFree(ts->dB); cudaFree(ts->dC);
    ts->active = false;
}

/* Build the 64-entry role map from the mix weights.
 *
 * 64 rather than 32 entries: with eight roles a 32-slot map cannot represent
 * a mix like 5:1:1:1 without rounding one of them to zero, and a role that
 * silently disappears is exactly the sort of thing that produces a confident
 * wrong attribution. */
#define GPL_ROLE_MAP_SLOTS 64

static void role_weights(const gpl_args_t *a, int w[GPL_ROLE_COUNT]) {
    /* With the cuBLAS backend the in-kernel tensor role leaves the kernel
     * entirely - its warps go back to the other roles and the tensor work
     * arrives on its own stream. */
    w[GPL_ROLE_TENSOR] = a->tensor_cublas ? 0 : a->mix_tensor;
    w[GPL_ROLE_FMA]    = a->mix_fma;
    w[GPL_ROLE_DRAM]   = a->mix_dram;
    w[GPL_ROLE_SFU]    = a->mix_sfu;
    w[GPL_ROLE_INT32]  = a->mix_int32;
    w[GPL_ROLE_SMEM]   = a->mix_smem;
    w[GPL_ROLE_L2]     = a->mix_l2;
    w[GPL_ROLE_ATOMIC] = a->mix_atomic;
}

static void build_role_map(const gpl_args_t *a, unsigned char *map, int *counts) {
    int w[GPL_ROLE_COUNT];
    role_weights(a, w);

    int total = 0;
    for (int r = 0; r < GPL_ROLE_COUNT; r++) total += w[r];
    if (total <= 0) { w[GPL_ROLE_FMA] = 1; total = 1; }

    /* Largest-remainder apportionment, so a role with a non-zero weight
     * always gets at least one slot rather than being rounded away. */
    int assigned = 0;
    int base[GPL_ROLE_COUNT];
    double rem[GPL_ROLE_COUNT];
    for (int r = 0; r < GPL_ROLE_COUNT; r++) {
        double exact = (double)GPL_ROLE_MAP_SLOTS * w[r] / total;
        base[r] = (int)exact;
        if (w[r] > 0 && base[r] == 0) base[r] = 1;   /* never silently drop */
        rem[r] = exact - (int)exact;
        assigned += base[r];
    }
    while (assigned > GPL_ROLE_MAP_SLOTS) {          /* trim the biggest */
        int big = 0;
        for (int r = 1; r < GPL_ROLE_COUNT; r++) if (base[r] > base[big]) big = r;
        base[big]--; assigned--;
    }
    while (assigned < GPL_ROLE_MAP_SLOTS) {          /* hand out remainders */
        int bestr = -1; double best = -1.0;
        for (int r = 0; r < GPL_ROLE_COUNT; r++)
            if (w[r] > 0 && rem[r] > best) { best = rem[r]; bestr = r; }
        if (bestr < 0) bestr = GPL_ROLE_FMA;
        base[bestr]++; rem[bestr] = -1.0; assigned++;
    }

    int i = 0;
    for (int r = 0; r < GPL_ROLE_COUNT; r++) {
        counts[r] = base[r];
        for (int k = 0; k < base[r] && i < GPL_ROLE_MAP_SLOTS; k++)
            map[i++] = (unsigned char)r;
    }
    while (i < GPL_ROLE_MAP_SLOTS) map[i++] = GPL_ROLE_FMA;
}

static const char *role_name(int r) {
    switch (r) {
        case GPL_ROLE_TENSOR: return "tensor";
        case GPL_ROLE_FMA:    return "fma";
        case GPL_ROLE_DRAM:   return "dram";
        case GPL_ROLE_SFU:    return "sfu";
        case GPL_ROLE_INT32:  return "int32";
        case GPL_ROLE_SMEM:   return "smem";
        case GPL_ROLE_L2:     return "l2";
        case GPL_ROLE_ATOMIC: return "atomic";
        default:              return "?";
    }
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
    CUDA_OK(cudaMalloc((void **)&d_role, GPL_ROLE_MAP_SLOTS));
    /* sink[0] is the optimiser barrier; sink[1..] are atomic targets. */
    CUDA_OK(cudaMalloc((void **)&d_sink, sizeof(float) * (1 + GPL_ATOMIC_SLOTS)));
    CUDA_OK(cudaMemset(d_stream, 0x3c, buf_bytes));
    CUDA_OK(cudaMemset(d_sink, 0, sizeof(float) * (1 + GPL_ATOMIC_SLOTS)));

    /* Tensor work via a concurrent cuBLAS stream, when asked for. */
    gpl_tensor_stream_t tstream;
    memset(&tstream, 0, sizeof(tstream));
    const bool use_cublas_tensor = (a->tensor_cublas && a->mix_tensor > 0);
    if (use_cublas_tensor && tensor_stream_init(ctx, &tstream) != 0) return -1;

    /* GEMMs enqueued per batch, scaled by the tensor weight relative to the
     * others so --mix-* still means something across both backends. */
    const int gemms_per_batch = use_cublas_tensor
        ? (a->mix_tensor > 0 ? a->mix_tensor : 1) : 0;

    /* Tensor-only under cuBLAS means no kernel at all. */
    const bool run_kernel = !(use_cublas_tensor && a->mix_fma == 0 && a->mix_dram == 0);

    unsigned char role_map[GPL_ROLE_MAP_SLOTS];
    int counts[GPL_ROLE_COUNT];
    build_role_map(a, role_map, counts);
    CUDA_OK(cudaMemcpy(d_role, role_map, GPL_ROLE_MAP_SLOTS, cudaMemcpyHostToDevice));
    {
        char mix[256]; int off = 0;
        for (int r = 0; r < GPL_ROLE_COUNT; r++) {
            if (counts[r] > 0 && off < (int)sizeof(mix) - 24)
                off += snprintf(mix + off, sizeof(mix) - off, "%s=%d ",
                                role_name(r), counts[r]);
        }
        gpl_logf("powervirus: %d blocks x %d warps | warps/%d: %s| stream=%.0f MiB",
                 blocks, GPL_BLOCK_WARPS, GPL_ROLE_MAP_SLOTS, mix,
                 buf_bytes / 1048576.0);
    }
    if (use_cublas_tensor) {
        gpl_logf("  tensor backend: cuBLAS %s GEMM n=%d, %d per batch on a concurrent stream%s",
                 gpl_prec_name(a->precision), tstream.n, gemms_per_batch,
                 run_kernel ? "" : " (kernel disabled: tensor-only)");
    } else {
        gpl_logf("  tensor backend: in-kernel wmma (warp-synchronous)");
    }

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

    /* Dynamic shared memory only when the SMEM role is actually in the mix -
     * requesting it unconditionally would cap occupancy on every other rung
     * and quietly change what the ladder is comparing. */
    const size_t shmem = counts[GPL_ROLE_SMEM] > 0
        ? (size_t)GPL_BLOCK_WARPS * GPL_SMEM_PER_WARP * sizeof(float) : 0;

    const bool duty = (a->duty_on_ms > 0.0);
    const bool fixed_work = (a->iters > 0);

    /* ---- warmup ---- */
    gpl_telemetry_set_phase(ctx->tele, GPL_PHASE_WARMUP);
    uint64_t t_start = gpl_mono_ns();
    uint64_t warmup_ns = (uint64_t)(a->warmup_sec * 1e9);
    while (gpl_mono_ns() - t_start < warmup_ns) {
        tensor_stream_enqueue(&tstream, gemms_per_batch * batch);
        if (run_kernel)
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < nstreams; s++)
                    gpl_powervirus_kernel<<<blocks, GPL_BLOCK_DIM, shmem, streams[s]>>>(
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

            /* Ramped edge: rather than going straight to full blast, spend
             * the first ramp_ns launching a fraction of the blocks, rising
             * to all of them. Block count is the only lever with
             * millisecond granularity here - throttling launches instead
             * would just move the step, not soften it. */
            uint64_t ramp_ns = 0;
            if (a->duty_ramp != GPL_RAMP_NONE) {
                ramp_ns = (uint64_t)((a->duty_ramp_ms > 0.0
                                      ? a->duty_ramp_ms
                                      : a->duty_on_ms * 0.25) * 1e6);
                if (ramp_ns > burst_ns / 2) ramp_ns = burst_ns / 2;
            }

            while (gpl_mono_ns() - b0 < burst_ns) {
                int nb = blocks;
                if (ramp_ns > 0) {
                    uint64_t el = gpl_mono_ns() - b0;
                    double f = 1.0;
                    if (el < ramp_ns) {
                        double x = (double)el / (double)ramp_ns;
                        f = (a->duty_ramp == GPL_RAMP_EXP) ? (x * x) : x;
                    } else if (el > burst_ns - ramp_ns) {
                        double x = (double)(burst_ns - el) / (double)ramp_ns;
                        f = (a->duty_ramp == GPL_RAMP_EXP) ? (x * x) : x;
                    }
                    nb = (int)(blocks * f);
                    if (nb < 1) nb = 1;
                }
                (void)nb;
                tensor_stream_enqueue(&tstream, gemms_per_batch);
                if (run_kernel)
                    for (int s = 0; s < nstreams; s++)
                        gpl_powervirus_kernel<<<nb, GPL_BLOCK_DIM, shmem, streams[s]>>>(
                            d_role, d_stream, stream_elems, inner, d_sink);
                CUDA_OK(cudaDeviceSynchronize());
                ctx->iterations_steady += nstreams;
            }
            sleep_ms(a->duty_off_ms);
        } else {
            tensor_stream_enqueue(&tstream, gemms_per_batch * batch);
            if (run_kernel)
                for (int b = 0; b < batch; b++)
                    for (int s = 0; s < nstreams; s++)
                        gpl_powervirus_kernel<<<blocks, GPL_BLOCK_DIM, shmem, streams[s]>>>(
                            d_role, d_stream, stream_elems, inner, d_sink);
            CUDA_OK(cudaDeviceSynchronize());
            ctx->iterations_steady += (uint64_t)nstreams * batch;
        }
    }
    ctx->compute_seconds_steady = (double)(gpl_mono_ns() - s_start) / 1e9;

    tensor_stream_free(&tstream);
    for (int i = 0; i < nstreams; i++) cudaStreamDestroy(streams[i]);
    free(streams);
    cudaFree(d_stream);
    cudaFree(d_role);
    cudaFree(d_sink);
    return 0;
}
