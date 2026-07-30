# gpu-power-lab — Design

Author: Claw (with Andrew)
Status: **v0.1 — scaffold + first working slice**
Date: 2026-07-28

## Goal

Characterize a single NVIDIA GPU (and later, multi-GPU nodes) by
pushing a controlled ramp of compute workloads while sampling power,
temperature, clocks, utilization, and throttle reasons at high rate.
Determine — for each (op, precision, size, concurrency) tuple — the
delivered TFLOPS, energy consumed, and thermal/power headroom, on a
range of AWS GPU instance types.

## Non-goals

- Training real models. This is a synthetic microbenchmark harness.
- Cross-vendor. NVIDIA-only. NVML and cuBLAS are the primary hooks.
- Anomaly detection or online decisioning. Offline analysis only.

## Design principles

1. **The measurement window is a clean C process.** No Python
   interpreter, no GC, no allocator surprises inside the steady-state
   window. Everything else lives outside it.
2. **One rung, one process.** The C binary runs a single rung and
   exits. Python orchestrates many rungs. This gives a hard reset
   between measurements and makes each rung reproducible from a shell.
3. **Data-first contract.** The JSON summary schema is the source of
   truth. Both sides validate against it. Schema is versioned.
4. **Cooldown between rungs is a first-class thing.** Without it,
   thermal history dominates power readings and results are not
   comparable.
5. **Sample throttle reasons, not just power.** Power alone is
   ambiguous — 700W drawn while throttled is a very different result
   from 700W drawn cleanly.

## Layer split

### C runner (`runner/`)

Owns everything on the hot path:

- Workload execution (initial: GEMM via `cublasGemmEx` in FP32 / TF32 /
  FP16 / BF16; later: FFT, memory-bandwidth, conv, all-reduce).
- Configurable concurrency via N CUDA streams.
- Warmup → steady-state → stop state machine, driven by wall-clock.
- Telemetry sampler thread:
  - Pinned to a core (best-effort via `pthread_setaffinity_np`).
  - Uses `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` for
    jitter-free periodic wakeups.
  - Reads NVML fields listed below.
  - Emits one NDJSON line per sample to `--out-metrics`.
  - Maintains running aggregates for the summary.
- Summary aggregation and JSON emit.

**NVML fields sampled every tick:**

| Field | NVML call | Notes |
|---|---|---|
| `power_mw` | `nvmlDeviceGetPowerUsage` | Instantaneous |
| `energy_mj` | `nvmlDeviceGetTotalEnergyConsumption` | Monotonic; delta gives true energy |
| `temp_gpu_c` | `nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)` | |
| `temp_mem_c` | `nvmlDeviceGetTemperature(NVML_TEMPERATURE_MEM)` | Where available |
| `clock_sm_mhz` | `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` | |
| `clock_mem_mhz` | `nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)` | |
| `util_sm_pct` | `nvmlDeviceGetUtilizationRates` | |
| `util_mem_pct` | `nvmlDeviceGetUtilizationRates` | |
| `throttle_reasons` | `nvmlDeviceGetCurrentClocksEventReasons` | Bitmask; OR across window |
| `fan_pct` | `nvmlDeviceGetFanSpeed` | Where available |

**Workload rungs (initial):**

- `sgemm` — `cublasGemmEx` with configurable precision and matrix
  size. Buffers allocated once, reused. One kernel dispatch per
  iteration per stream. Runs until the phase deadline expires.

**TFLOPS calc:** `2 * M * N * K * iterations / seconds / 1e12` (M=N=K
for the initial square-matrix sweep).

### Python orchestrator (`orchestrator/`)

Owns everything else:

- YAML plan expansion (`plan.py`) — one plan file expands to N rung
  invocations. Cartesian product of ops × precisions × sizes ×
  streams, with a shared warmup/steady/sample-hz section.
- Campaign runner (`campaign.py`) — iterates the rung list, invokes
  the C binary via `subprocess.run`, captures its two output files,
  handles non-zero exit codes.
- Cooldown (`cooldown.py`) — between rungs, polls NVML (via
  `pynvml`, fine here — not in the measurement window) and waits
  until GPU temperature drops to baseline+Δ or a timeout elapses.
- Result collation — reads all summary JSONs into a pandas DataFrame,
  writes a campaign-level Parquet + a manifest.
- Upload (`upload.py`) — writes results to S3 and optionally to
  Timestream. TBD.
- Report (`report.py`) — HTML/PDF campaign report. TBD.

## Contract: rung summary schema

See `schema/rung-summary.schema.json`. The high-level shape:

```
{
  "schema_version": 1,
  "rung_id": "...",
  "invocation": { args echoed back },
  "device": { uuid, name, driver, power_limit_w, ... },
  "timing": { start_utc, warmup_ms, steady_ms },
  "compute": { iterations, tflops_measured, tflops_theoretical, efficiency },
  "power": { avg_w, peak_w, p50, p95, p99, energy_j },
  "thermal": { avg_c, peak_c, throttled_sec },
  "throttle": { reasons: [...], any_throttled: bool, mask: uint64 },
  "utilization": { sm_avg_pct, mem_avg_pct },
  "clocks": { sm_avg_mhz, sm_min_mhz, mem_avg_mhz, sm_boost_dropped },
  "result": "ok" | "throttle_hit" | "oom" | "cuda_error" | "nvml_error"
}
```

`schema_version` is bumped whenever a field is renamed or removed.
Additions are additive without a bump.

## Rung state machine (inside the C runner)

```
    ┌─────────┐   deadline_warmup   ┌────────┐   deadline_steady   ┌──────┐
    │ WARMUP  │────────────────────▶│ STEADY │────────────────────▶│ DONE │
    └─────────┘                     └────────┘                     └──────┘
         │                              │
         └──── telemetry samples ───────┴──── (recorded, tagged by phase)
```

- **WARMUP**: workload runs, telemetry samples flow to NDJSON tagged
  `"phase": "warmup"`, but they are excluded from summary aggregates.
- **STEADY**: workload runs, telemetry samples tagged `"phase": "steady"`
  and included in summary aggregates.
- **DONE**: workload thread and telemetry thread both stopped, buffers
  flushed, summary written.

## Cooldown (Python side)

Between rungs, wait until any of:

- GPU temp ≤ `baseline_c + cooldown_delta_c` (default: `40 + 5 = 45°C`)
- `cooldown_max_sec` elapsed (default: 120s)
- `cooldown_min_sec` elapsed AND no active workload from external
  processes (best effort — check `nvmlDeviceGetComputeRunningProcesses`).

Baseline is captured at campaign start with a 60-second idle window.

## Deployment on AWS

Out of scope for v0.1. Sketch:

- Terraform → single-instance ASG with NVIDIA GPU-Optimized AMI
- user-data pulls tarball from S3 (binary + venv + plan)
- systemd unit runs `campaign.py`
- Results upload to `s3://<bucket>/campaigns/<campaign_id>/`

Instance types of interest:

- `g5.xlarge` — A10G (300W) — smoke test, cheap
- `g6e.xlarge` — L40S (350W)
- `p4d.24xlarge` — 8× A100 40GB (400W each)
- `p5.48xlarge` — 8× H100 80GB (700W each) — headline target

## What "maxed out" means

Report all three; the interesting finding is which hits first:

1. **Power-max** — sustained draw ≥ 99% of power limit for ≥ 30s
2. **Thermal-max** — throttle bitmask contains any thermal slowdown reason
3. **Compute-max** — measured TFLOPS within 5% of vendor spec for that precision

## Roadmap

**v0.1 (this commit)**
- [x] Repo scaffold, design doc, schema
- [x] C runner: args, NVML sampler, GEMM workload, summary JSON
- [x] CMake build
- [x] Python: plan expander, campaign runner, cooldown, example plan
- [ ] End-to-end test on a real GPU

**v0.2**
- [ ] BF16, TF32, FP32 precisions verified against vendor peak
- [x] Memory-bandwidth workload (STREAM-style kernel, `memstream` op)
- [x] FFT workload (`cuFFT`, `fft` op)
- [ ] DCGM sampler integration (SM active %, tensor active %, DRAM active %)
- [ ] `pytest` harness for the Python side
- [ ] Schema validation via `jsonschema` on load

**v0.3**
- [ ] Multi-GPU: per-device workload + per-device sampler threads
- [ ] NVLink stress via NCCL all-reduce
- [ ] Terraform for AWS bootstrap
- [ ] Upload to S3 + Timestream

**v0.4**
- [ ] HTML/PDF report generation
- [ ] Grafana dashboards for the campaign time series
- [ ] CI: build + smoke test on a `g5.xlarge` runner per PR

**v0.5**
- [ ] FP8 (H100 only) via `cublasLtMatmul`
- [ ] CUPTI kernel-level counters for deep-dive rungs
- [ ] Regression mode: compare a campaign against a named baseline

## Open questions

1. Sample rate default — 100 Hz or 250 Hz? Trade-off is file size vs
   ability to catch fast transients. Default 100 Hz for now.
2. DCGM sampler in-process (dlopen `libdcgm.so`) or side-car
   (`dcgm-exporter` + scrape)? Side-car is simpler and standard;
   in-process gives tighter time correlation. Start with side-car.
3. Multi-GPU: one binary per device, or one binary that fans out?
   One-per-device keeps the C code simple and matches the "one rung,
   one process" model. Prefer that.
4. Persistence mode + fixed clocks — should we lock clocks to defeat
   boost jitter? Two campaign profiles: `boost-on` (realistic) and
   `boost-off` (reproducible). Default `boost-on`.
