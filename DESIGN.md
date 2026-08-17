# gpu-power-lab — Design

Author: Claw (with Andrew)
Status: **v0.2 — scope revised: maximum power and transient behaviour**
Date: 2026-08-17

## Goal

Answer two questions about a single NVIDIA GPU (and later, multi-GPU
nodes), by writing synthetic workloads deliberately designed to be
worst-case for the power delivery system rather than representative of
real jobs:

- **O1 — Can we hit 100%?** Is it possible to write code that drives a
  GPU to sustained draw at 100% of its enforced power limit — pinned to
  the cap, power-limited rather than compute- or thermal-limited?
- **O2 — How spiky can we make it?** Having reached the ceiling, can we
  generate extreme transient behaviour: large, fast, repeatable swings
  between idle and maximum, with instantaneous excursions that
  *overshoot* the rated limit before the GPU's own power controller
  claws them back?

Characterization across (op, precision, size, concurrency) — delivered
TFLOPS, energy, thermal and power headroom — remains in scope, but is
now the *supporting* measurement layer rather than the objective.

### Why we care

DAME's interest is grid-side. A GPU fleet that can be driven to its
ceiling and then modulated on demand is a controllable load; the same
capability, uncontrolled, is a power-quality hazard. Both directions
need the same number: **how much power can this hardware actually
swing, and how fast?** The `nvidia-load-shave` work measured the
downward direction (capping power, measuring lost throughput) using
vLLM as the load. It topped out at ~71% of TDP, which is a property of
the workload, not the GPU. This harness removes the workload as the
limiting factor.

### Success criteria

**O1 is met** when, for a given device, a rung sustains:

- mean power ≥ 99% of the enforced limit over a ≥ 60s steady window, and
- `throttle_reasons` shows the power cap (`SW_POWER_CAP` 0x4 and/or
  `HW_POWER_BRAKE` 0x80) as the *only* active reason — no thermal
  slowdown, no SW thermal, no clock-setting reason.

Thermal-limited at 99% is a different (and less interesting) result and
must be reported as such.

**O2 is met** when we can quantify, with defensible instrumentation:

- **Overshoot** — peak instantaneous power as a fraction of the enforced
  limit, and how long the excursion persists before the controller
  clamps it.
- **Slew rate** — dP/dt in W/ms on both the rising and falling edge.
- **Repeatability** — the same square wave sustained for minutes at a
  chosen period, with jitter characterized.
- **Depth** — idle-floor to peak swing in watts.

A negative result is a real result. "Instantaneous draw never exceeds
the cap at any observable timescale" is a finding worth publishing, and
the instrumentation section below exists so that we can distinguish it
from "our sampler was too slow to see it."

### Working hypothesis: 100% is probably unreachable

The expected answer to O1 is **no** — that no workload we can write
holds a modern datacenter GPU at its full rated power, and that the
rated number is a thermal-design envelope for worst-case mixed
activity plus margin, not an operating point any single kernel
achieves. `nvidia-load-shave` saw ~71% of TDP from real LLM inference.
A synthetic virus should beat that substantially. Reaching 100% would
be the surprise.

That reframes the deliverable. If the answer is "no", the useful
output is not the failure but **the breakdown: how much power each
class of work contributes, why each one runs out of headroom, and
what the residual gap is made of.** That breakdown is what makes the
result transferable — it says what a hypothetical worst-case tenant
could do to a site, which is the grid-side question underneath all of
this.

Why we expect the ceiling to be out of reach, and what each reason
implies for the experiment:

| Mechanism | Consequence for reaching TDP |
|---|---|
| **Issue bandwidth** — a warp scheduler issues one instruction per cycle per scheduler | Weaker on Blackwell than expected: `tcgen05.mma` is asynchronous and issued by a *single thread* (verified, see `docs/blackwell-cuda-notes.md` §3), so one thread can keep the tensor core busy while other warps drive the FMA pipe and memory. Issue bandwidth is a real constraint for the non-tensor units but does not block mixing |
| **DVFS** — the controller drops clocks as power approaches the cap | Approaching the ceiling is self-limiting: the closer you get, the lower the frequency, so power converges below the cap rather than hitting it |
| **Datapath exclusivity** — operands come from register file, shared memory, or tensor memory | Kernels that saturate one operand path starve the others; the "mix" that maximizes power is not the mix that maximizes any one unit |
| **Memory power is traffic-proportional** | HBM only draws its share under genuine bandwidth load, which a register-resident GEMM deliberately avoids — the two goals fight |
| **Uncore is idle in single-GPU tests** | NVLink, the switch fabric and inter-die traffic contribute real watts on a full node but nothing on a lone GPU. A single B300 may be *structurally* incapable of its own rated draw |
| **Vendor margin** | TDP is set for a worst case across silicon lottery, ambient, and workload, with headroom on top |
| **Thermal arrival** | On air-cooled or marginally-cooled parts, thermal limits bind before power limits, converting the experiment into a cooling measurement |

Note the tension this creates with the `powervirus` design: mixing
units raises total power *only* if the units genuinely run concurrently
rather than stealing issue slots and operand bandwidth from each other.
Whether the mix beats a pure tensor-core loop is an empirical question,
not a given — and measuring it is the point. The verified `tcgen05`
issue model makes the mix plausible; it does not make it true.

### The real O1 deliverable: a power attribution breakdown

Rather than one number, O1 produces a table of **marginal watts per
class of work**, built by a deliberate ladder:

1. **Idle floor** — persistence on, clocks settled, nothing running.
   Everything else is measured against this.
2. **Single-unit rungs** — each isolated as far as the ISA allows:
   FP32 FFMA chain; FP64 (expected to be a weak path on Blackwell
   datacenter parts, which is itself worth confirming); INT32; SFU /
   transcendentals; tensor core at each supported precision (TF32,
   BF16, FP16, FP8, and the low-precision Blackwell formats);
   register-file-only movement; shared-memory traffic; L2-resident
   streaming; HBM streaming at peak bandwidth; atomics; PCIe DMA.
3. **Pairwise rungs** — the interesting combinations: tensor+DRAM,
   tensor+FMA, FMA+DRAM, and so on. The difference between the pair
   and the sum of its parts is the **interaction term**, and it is the
   number that tells us whether mixing helps or the units are simply
   competing for issue slots.
4. **Full mix sweep** — the `--mix-*` weight sweep from `powervirus`,
   searching for the maximum.
5. **Multi-GPU / NVLink** (v0.3) — adds the uncore contribution that a
   single device cannot produce.

Each rung reports watts above idle, watts per unit of work, achieved
clocks, and which limiter bound it (power / thermal / issue / memory).
The output is a stacked attribution: *this much from tensor cores,
this much from DRAM traffic, this much overlap, this much still
unaccounted for* — and the unaccounted-for portion is the honest
statement of what we could not make the chip do.

The same ladder run per precision answers the question directly:
**FP32 versus BF16 versus FP8 versus FP4 is not just a throughput
comparison, it is a watts-per-flop and a total-watts comparison**, and
the highest-throughput format is not necessarily the highest-power one.
Lower precision moves more flops per joule but also finishes its
operands faster, shifting the bottleneck toward memory. Which format
maximizes *power* is genuinely unknown until measured, and it is one of
the more interesting things this harness can report.

## Non-goals

- Training real models. This is a synthetic microbenchmark harness.
- Cross-vendor. NVIDIA-only. NVML and cuBLAS are the primary hooks.
- Anomaly detection or online decisioning. Offline analysis only.
- Defeating, disabling, or circumventing hardware protection. We drive
  the GPU hard through supported APIs and observe what its own
  controllers do. Power brake and thermal shutdown stay armed. See
  **Safety envelope**.
- Damage testing. We are characterizing the ceiling, not looking for the
  point of failure.

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
6. **Never claim a transient the instrument can't resolve.** Every
   power figure carries the measurement path that produced it and that
   path's effective bandwidth. A 100 Hz poll of a driver-averaged
   register cannot substantiate a claim about a 2 ms spike. See
   **Instrumenting transients**.

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

**Workload rungs:**

- `sgemm` — `cublasGemmEx` with configurable precision and matrix
  size. Buffers allocated once, reused. One kernel dispatch per
  iteration per stream. Runs until the phase deadline expires.
- `memstream` — STREAM-triad bandwidth kernel (FP32).
- `fft` — batched 1-D cuFFT C2C (FP32).
- `powervirus` — **new, O1.** See below.
- Any of the above under `--duty` modulation — **new, O2.** See below.

**TFLOPS calc:** `2 * M * N * K * iterations / seconds / 1e12` (M=N=K
for the initial square-matrix sweep).

## Reaching the ceiling (O1)

A large tensor-core GEMM is the obvious candidate and is probably not
the answer. Peak-FLOPS GEMM is a narrow power profile: the tensor cores
are saturated, but the FP32/INT pipes, the register file, the LSU path
and DRAM are all comparatively quiet, and a well-tuned GEMM is
deliberately L2-friendly. Published power-virus work consistently beats
peak-FLOPS GEMM by mixing units that draw from *different* parts of the
chip's power budget simultaneously.

`powervirus` is therefore a hand-written kernel, not a cuBLAS call,
targeting concurrent activity across:

- **Tensor cores** — back-to-back `mma.sync` on register-resident
  fragments, no global traffic in the inner loop.
- **CUDA cores** — an independent FFMA chain interleaved with the MMA
  stream so both pipes issue in the same cycles.
- **Register file / operand collectors** — wide live sets, deliberately
  high register pressure, minimal reuse of the same operand banks.
- **DRAM and L2** — a background stream of strided reads sized to miss
  L2, running concurrently on separate SMs or separate streams.
- **Occupancy** — enough resident warps per SM that there is no
  issue-stall idle time anywhere in the loop.

Tunables exposed per rung: `--mix-tensor`, `--mix-fma`, `--mix-dram`
(relative weights), plus `--size` and `--streams` as today. The
campaign sweeps the mix and reports which combination lands highest —
that sweep *is* the O1 experiment, and the mix that wins is a result in
itself.

Supporting conditions the runner must establish before the steady
window, all through supported NVML calls:

- Persistence mode on.
- Power limit raised to the device maximum
  (`nvmlDeviceGetPowerManagementLimitConstraints` → max, then
  `nvmlDeviceSetPowerManagementLimit`). Requires root. Recorded in the
  summary as `enforced_limit_w` alongside the default, so every power
  figure is expressed against the limit actually in force.
- Fans to maximum where the platform allows it, so that O1 is not
  spuriously converted into a thermal-limited result.
- Clocks: both profiles from open question 4 apply, and here the
  distinction matters — `boost-on` finds the true ceiling, `boost-off`
  with locked clocks gives the reproducible number.

## Generating transients (O2)

Once a rung can hold the ceiling, spikiness is a scheduling problem:
drive the O1 workload in a square wave and characterize the edges.

`--duty` adds a modulation layer over any op:

```
--duty-on-ms N      # burst length at full blast
--duty-off-ms N     # idle gap
--duty-cycles N     # or run until the steady deadline
--duty-ramp none|linear|exp   # optional soft edge, for contrast
```

The gap must be a *real* idle, not a spin — the interesting number is
the swing from the true idle floor, so the off phase parks the workload
threads and lets the GPU clock down naturally.

Sweeping the period from ~1 ms to ~10 s crosses several control
boundaries and we expect visibly different regimes:

| Period | What we expect to dominate |
|---|---|
| ~1 ms | Below the power controller's response time — draw may look like a smeared average; best chance of a genuine over-limit excursion |
| ~10 ms | Controller reacting; overshoot then clamp should be resolvable |
| ~100 ms | Clean square wave; DVFS boost/drop visible on each edge |
| ~1–10 s | Full thermal excursion per cycle; fan response enters the loop |

Derived metrics per duty rung: overshoot ratio (peak instantaneous ÷
enforced limit), time-above-limit per cycle, rising and falling slew in
W/ms, swing depth, and cycle-to-cycle jitter.

**Multi-GPU and multi-node phase alignment** is the extension that
makes this grid-relevant: eight H100s swinging 500 W each *in phase* is
a 4 kW step at the node, and a rack of them is a site-level event. v0.3
adds a shared start barrier so the per-device runners begin their duty
cycles on a common timebase; the interesting comparison is aligned vs
deliberately phase-staggered.

## Instrumenting transients

This is the part that decides whether O2 produces a defensible answer
or an artefact. Four measurement paths, in increasing order of
trustworthiness for fast events:

1. **`nvmlDeviceGetPowerUsage`** — what the runner samples today.
   Convenient, but on most parts it returns a driver-side *averaged*
   value over an internal window, so it structurally cannot show a
   short excursion. Fine for O1, insufficient for O2 on its own.
2. **`nvmlDeviceGetFieldValues` with `NVML_FI_DEV_POWER_INSTANT`** —
   the instantaneous counterpart to `NVML_FI_DEV_POWER_AVERAGE`, and
   the cheapest real improvement. Both must be logged side by side:
   the *divergence* between them is itself evidence of a transient.
3. **`nvmlDeviceGetSamples(NVML_TOTAL_POWER_SAMPLES)`** — the driver
   buffers power samples at its own rate and hands back a batch with
   device-side timestamps. Draining this buffer beats polling in a
   loop, because the sample rate stops being bounded by our thread's
   wakeup jitter. This becomes the primary path for duty rungs.
4. **Out-of-band** — the only true ground truth for sub-millisecond
   behaviour and the only path that can see the board-level draw NVML
   reports second-hand: PMBus/Redfish telemetry from the BMC where the
   platform exposes it, or a bench instrument (clamp meter or
   current-shunt on the 12 V rails into a scope) on hardware we
   physically own. Cloud instances give us nothing here, so the
   in-band claims from paths 2–3 need at least one on-premises
   cross-check before we publish an over-limit number.

Bench instrumentation (path 4) is not available at this stage, so
until it is, every over-limit figure is reported as an **in-band**
result — "NVML's instantaneous counter reported X% of the enforced
limit", never "the GPU drew X%".

The strongest substitute available in-band is an **energy-integral
cross-check**: integrate the sampled power curve over a duty rung and
compare it against the delta in
`nvmlDeviceGetTotalEnergyConsumption`. The energy counter is a
monotonic hardware accumulator — it cannot miss a spike the way a
sampler can. It won't reveal the peak, but a persistent gap between
counted energy and integrated power is quantitative evidence that the
sampler is missing area, which is what an unresolved excursion looks
like. Cheap to add and it belongs on every duty rung.

Also worth capturing during duty rungs, since they detect the clamp
even when we can't resolve the spike that triggered it:

- `nvmlDeviceGetViolationStatus(NVML_PERF_POLICY_POWER)` — cumulative
  throttled microseconds, which gives an integral measure of how much
  clamping the controller did per cycle.
- The `HW_POWER_BRAKE` bit specifically: it firing is direct evidence
  that the hardware saw an excursion serious enough to act on.

Every emitted power sample gains a `source` field naming which of the
four paths produced it. No mixing of paths inside one aggregate.

## Safety envelope

- Hardware protections stay armed. We never disable power brake or
  thermal shutdown, and we do not modify VBIOS, use vendor-unsupported
  overvolting tools, or push power limits past the range the driver
  itself reports as valid.
- `--stop-on-throttle` gains a hard-stop sibling that aborts the rung
  on thermal slowdown or on temperature within a margin of the shutdown
  threshold (`nvmlDeviceGetTemperatureThreshold`).
- O2 work runs on rented cloud instances first. Sustained multi-kW
  square waves on our own hardware go ahead only after the rack's
  breaker headroom and UPS transient response have been checked — the
  point of the experiment is that this load pattern is hostile to
  upstream power delivery, which applies to our own upstream too.

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

## Efficiency outputs: EDP, EDPp, and %TDP

Every rung reports the standard energy-efficiency trio alongside the
raw power numbers. These are the metrics that make the results
comparable to published work and, more usefully here, they identify
*operating points* rather than just ceilings.

| Metric | Definition | What it favours |
|---|---|---|
| **Energy** | E, joules for the work | Lowest power / lowest clocks |
| **EDP** | E × D | Energy and performance weighted equally |
| **EDPp** | E × D² | Performance-leaning; tolerates more energy for speed |
| **%TDP** | mean power ÷ TDP | Where we sit against the thermal design envelope |

**Naming.** We use NVIDIA's terms — **EDP** for the energy-delay
product and **EDPp** for the performance-weighted variant (E × D²,
written ED²P in much of the academic literature). Field names follow
suit: `edp_j_s` and `edpp_j_s2`. Keep this consistent everywhere,
including in reports, so results line up with NVIDIA's own published
figures without a translation step.

### Delay needs fixed work, which the runner does not currently do

EDP and EDPp are only meaningful when **D is the time to complete a
fixed quantity of work.** Today the runner is fixed-*time*: it runs
until a wall-clock deadline and counts however many iterations it
managed. Under that model delay is a constant by construction and EDP
degenerates into energy.

So this needs a runner change: a **fixed-work mode** (`--iters N`, or
`--work <quantum>`), where the rung completes a defined amount of work
and the elapsed time is the measurement. Steady-state duration then
becomes an output rather than an input. Both modes stay available —
fixed-time for power characterization and duty rungs, fixed-work for
the efficiency trio.

### Work units, and the comparison trap

Delay is per unit of work, so the work unit must be stated and must
match for any comparison to mean anything:

- GEMM rungs: **per PFLOP** (10¹⁵ flops)
- Bandwidth rungs: **per TB moved**
- FFT rungs: per transform, or per PFLOP under the standard 5·N·log₂N
  convention — pick one and record it

**Cross-precision comparison is the trap.** A PFLOP of FP4 and a PFLOP
of FP32 are not the same work in any application sense, so their EDPs
are not directly comparable even though both are "per PFLOP". Report
them side by side, labelled, and let the reader draw the inference —
do not rank across precisions as if the unit were equivalent. Within a
precision, across power caps or clock settings, the comparison is
sound and is exactly what these metrics are for.

Every emitted EDP/EDPp carries its `work_unit`, and the collation
refuses to aggregate across differing units.

### The output that matters: EDP versus power cap

The single most useful thing this trio produces is a sweep of EDP and
EDPp **against the enforced power limit**, per workload. That curve
locates the efficiency sweet spot: the cap that minimizes EDP is the
best energy-performance tradeoff, and the EDPp minimum sits at a higher
cap because it weights speed more heavily.

This connects the harness directly back to `nvidia-load-shave`. That
project measured throughput lost per watt saved on real inference; this
one produces the same curve on synthetic workloads where we control
every variable, plus the second derivative that says whether the
operating point a site chooses is efficient or merely slow. A demand
response scheme that caps GPUs down to the EDP minimum is taking free
efficiency; capping below it is genuinely trading performance for grid
service, and these numbers say exactly where that line falls.

### TDP, and being careful what we divide by

Four different denominators get loosely called "TDP" and mixing them
produces nonsense:

- **TDP** — the vendor's thermal design power for the board. A
  specification, recorded from our device table, not readable from
  NVML.
- **Default power limit** — `nvmlDeviceGetPowerManagementDefaultLimit`.
  Often equal to TDP, not always.
- **Enforced limit** — what is actually set right now, which for O1
  rungs we deliberately raise to the maximum.
- **Max limit** — the top of
  `GetPowerManagementLimitConstraints`, which on some parts exceeds
  TDP.

Summaries carry all four, plus `pct_of_tdp` and
`pct_of_enforced_limit` as separate fields. O1's success criterion is
defined against the *enforced* limit; `%TDP` is the number for
cross-device comparison and for the grid-side conversation, since a
site cares about the board's rating, not what we configured.

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
  "power": { avg_w, peak_w, p50, p95, p99, energy_j,
             tdp_w, default_limit_w, enforced_limit_w, max_limit_w,
             pct_of_tdp, pct_of_enforced_limit, sample_source },
  "efficiency": {
     "mode": "fixed_work" | "fixed_time",
     "work_unit": "pflop" | "tb" | "transform" | "iteration",
     "work_done": .., "delay_s": .., "energy_j": ..,
     "energy_per_work_j": .., "delay_per_work_s": ..,
     "edp_j_s": ..,        // E × D
     "edpp_j_s2": ..,      // E × D²
     "perf_per_watt": ..   // work_unit per second per watt
  },
  "transient": {                       // duty rungs only
     "on_ms": .., "off_ms": .., "cycles": ..,
     "peak_instant_w": .., "overshoot_ratio": ..,
     "time_above_limit_ms": ..,
     "slew_rise_w_per_ms": .., "slew_fall_w_per_ms": ..,
     "swing_w": .., "cycle_jitter_ms_p95": ..,
     "power_violation_us": ..,          // nvmlDeviceGetViolationStatus
     "energy_counter_j": .., "energy_integrated_j": ..,
     "energy_gap_pct": ..               // >0 ⇒ sampler missing area
  },
  "thermal": { avg_c, peak_c, throttled_sec },
  "throttle": { reasons: [...], any_throttled: bool, mask: uint64,
                hw_power_brake_seen: bool },
  "utilization": { sm_avg_pct, mem_avg_pct },
  "clocks": { sm_avg_mhz, sm_min_mhz, mem_avg_mhz, sm_boost_dropped },
  "result": "ok" | "throttle_hit" | "oom" | "cuda_error" | "nvml_error"
             | "thermal_abort"
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

1. **Power-max** — sustained draw ≥ 99% of the *enforced* limit for ≥ 30s
2. **Thermal-max** — throttle bitmask contains any thermal slowdown reason
3. **Compute-max** — measured TFLOPS within 5% of vendor spec for that precision

For O1 only power-max counts as success. A rung that reaches 99% while
also thermally throttled has found the cooling limit, not the power
ceiling, and the campaign report must not conflate the two. The
headline O1 deliverable is two tables, per device: the highest
sustained `pct_of_enforced_limit` achieved and which workload mix got
there, and — more useful if nothing reaches 100%, which we expect —
the **power attribution breakdown** described above, showing marginal
watts per class of work, the interaction terms, and the size of the
unexplained residual.

The headline O2 deliverable is the overshoot number: peak instantaneous
draw as a percentage of the enforced limit, with the measurement path
and its bandwidth stated next to it, plus the slew rates and the
period at which overshoot is largest.

## Roadmap

**v0.1 (this commit)**
- [x] Repo scaffold, design doc, schema
- [x] C runner: args, NVML sampler, GEMM workload, summary JSON
- [x] CMake build
- [x] Python: plan expander, campaign runner, cooldown, example plan
- [ ] End-to-end test on a real GPU

**v0.2 — O1: reach the ceiling**
- [ ] BF16, TF32, FP32 precisions verified against vendor peak
- [x] Memory-bandwidth workload (STREAM-style kernel, `memstream` op)
- [x] FFT workload (`cuFFT`, `fft` op)
- [ ] `powervirus` kernel: mixed tensor / FMA / DRAM inner loop with
      `--mix-*` weights
- [ ] Raise power limit to device max + persistence mode at rung start;
      record `enforced_limit_w` in the summary
- [ ] Mix-sweep plan (`plans/o1-ceiling.yaml`) + per-device ceiling table
- [ ] Attribution ladder plan (`plans/o1-attribution.yaml`): idle floor,
      single-unit, pairwise, mix — with interaction terms computed
- [ ] Fixed-work mode (`--iters` / `--work`) so delay is measurable
- [ ] EDP / EDPp / %TDP in the summary, with `work_unit` guarding
      aggregation
- [ ] EDP-vs-power-cap sweep (`plans/efficiency-sweep.yaml`) and the
      curve that locates the EDP and EDPp minima per workload
- [ ] DCGM sampler integration (SM active %, tensor active %, DRAM active %)
- [x] `pytest` harness for the Python side
- [ ] Schema validation via `jsonschema` on load

**v0.2.5 — O2: transients**
- [ ] `NVML_FI_DEV_POWER_INSTANT` alongside `POWER_AVERAGE`, both logged
- [ ] `nvmlDeviceGetSamples(NVML_TOTAL_POWER_SAMPLES)` drain path, with
      `source` tagging on every emitted sample
- [ ] `--duty-on-ms` / `--duty-off-ms` / `--duty-cycles` / `--duty-ramp`
- [ ] Transient metrics block in the summary (overshoot, slew, jitter)
- [ ] `nvmlDeviceGetViolationStatus` + `HW_POWER_BRAKE` capture
- [ ] Period sweep plan (`plans/o2-transient.yaml`), 1 ms → 10 s
- [ ] Thermal hard-stop guard against the shutdown threshold

**v0.3**
- [ ] Multi-GPU: per-device workload + per-device sampler threads
- [ ] Phase-aligned duty cycles across devices (shared start barrier),
      aligned vs staggered comparison
- [ ] Out-of-band cross-check on owned hardware (BMC/PMBus or shunt+scope)
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
   ability to catch fast transients. Default 100 Hz for steady rungs.
   For duty rungs the polling rate is largely beside the point: the
   binding constraint is the driver's own averaging window, which is
   why the samples-buffer path exists. Open: what *is* that window on
   H100 / L40S / A10G? Measure it empirically per device — drive a
   known square wave and look at where `POWER_INSTANT` and
   `POWER_AVERAGE` diverge.
2. DCGM sampler in-process (dlopen `libdcgm.so`) or side-car
   (`dcgm-exporter` + scrape)? Side-car is simpler and standard;
   in-process gives tighter time correlation. Start with side-car.
3. Multi-GPU: one binary per device, or one binary that fans out?
   One-per-device keeps the C code simple and matches the "one rung,
   one process" model. Prefer that.
4. Persistence mode + fixed clocks — should we lock clocks to defeat
   boost jitter? Two campaign profiles: `boost-on` (realistic) and
   `boost-off` (reproducible). Default `boost-on`.
5. Does the virtualization layer on `g5`/`p5` clamp or filter what we
   can do here — can we actually raise the power limit on a non-metal
   instance, and does NVML report board power or a virtualized view?
   The `nvidia-load-shave` work established that *lowering* the limit
   works on `g5.xlarge`; raising it to max is untested.
6. Is the largest overshoot a single-GPU property at all, or does it
   only show up as a board/node-level effect once several GPUs share a
   power delivery path? Argues for getting to v0.3 multi-GPU before
   drawing conclusions.
7. Duty-cycle floor — how short can the off phase be before the launch
   and synchronization overhead dominates, i.e. what is the shortest
   period the runner can actually produce cleanly? Measure before
   choosing the bottom of the period sweep.
