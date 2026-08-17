# Real workloads and the node: R1, R2, R3

Status: **design, not yet built**
Date: 2026-08-17

The synthetic ceiling is now measured: **90.4% of 1100 W** on a single B300,
never throttled (`results/t2-b300`). That number is only useful as a
reference point. Three questions follow from it, and they are what this
plan covers.

| Phase | Question | Hardware |
|---|---|---|
| **R1** | What does a *training* step actually draw, and where does the gap to 90.4% go? | 1 × B300 |
| **R2** | What is the maximum a *real inference server* can be made to draw? | 1 × B300 |
| **R3** | What happens at node scale, where the fabric contributes and eight GPUs can move in phase? | 8 × B300 |

The unifying idea is the **overhead waterfall**. We know the ceiling. Each
real-workload rung adds one layer of overhead — backward pass, optimizer,
dataloader, collectives, scheduling — and each layer costs watts. Attributing
those watts is more valuable than any single benchmark number, because it says
which part of a production stack is leaving the power on the table, and
therefore how much of it a demand-response scheme could ever reclaim.

---

## Prerequisite: the runner must stop owning the workload

Every rung so far has been the runner driving its own kernel. Training and
inference are separate processes — PyTorch, vLLM — so the runner needs a mode
where it samples while something else drives the GPU.

**`--op observe`**: no workload, just the telemetry loop for a fixed duration,
emitting the same NDJSON and the same summary schema. Everything downstream —
the report generator, the schema, the energy cross-check — then works unchanged
on a training run.

Two additions that only matter in this mode:

- **Per-process attribution.** `nvmlDeviceGetComputeRunningProcesses` records
  which PIDs are resident and their memory, written into the summary. On a
  shared or accidentally-busy box this is the difference between a valid
  measurement and a quietly contaminated one.
- **Phase markers.** A named pipe or a marker file the workload writes to
  (`{"t": <mono_ns>, "phase": "backward"}`), which the sampler interleaves into
  the NDJSON. Coarse, but enough to attribute power to forward / backward /
  optimizer / dataloader without linking CUPTI into a Python process.

Both are small. `--op observe` is the blocking one — nothing in R1 or R2 can be
measured without it.

---

## R1 — A small training workload

**Question:** a training step contains the same dense GEMMs that drew 995 W
here. What does the whole step draw, and what does each layer of overhead cost?

### Model

**Llama-3.1 8B, full fine-tune, bf16.** Chosen because it fits a single B300
with room to spare — weights 16 GB, gradients 16 GB, AdamW fp32 optimizer state
~64 GB, ≈100 GB against 275 GB — so batch size and sequence length can be swept
without hitting an OOM wall midway through the sweep. Well-known and
reproducible; nobody has to trust our model code.

Deliberately *not* a LoRA or a tiny GPT: the point is to include the optimizer
and gradient traffic that a real step pays for.

**Write the loop ourselves**, in one file, rather than using a Trainer. Not
NIH — phase attribution requires knowing exactly where forward ends and
backward begins, and a framework that fuses or reorders phases makes the
waterfall unattributable.

### The waterfall

Each rung adds one layer. The difference between consecutive rungs is the cost
of that layer, in watts.

| Rung | What it adds | Expectation |
|---|---|---|
| `ceiling` | The synthetic tensor rung, as reference | 995 W (measured) |
| `gemm-shapes` | Just the GEMMs of one transformer layer, at the real shapes, in a loop | slightly below ceiling — real shapes are less friendly than a square 8192 |
| `forward` | Full forward pass, no autograd | + attention, norms, activations to HBM |
| `fwd-bwd` | Add the backward pass | roughly 2× the FLOPs, but more memory traffic per FLOP |
| `+optimizer` | Add AdamW step | a bandwidth-bound burst over ~100 GB of state |
| `+dataloader` | Real data pipeline, not a resident synthetic batch | host round-trips, possible bubbles |
| `full-step` | Everything, sustained for 5 minutes | the honest training number |

### Knobs worth sweeping, because they should move power

- **Batch size and sequence length** — arithmetic intensity. Low intensity
  moves the bottleneck to memory, which the B300 data says is a *70.9%* regime
  rather than a 90% one.
- **Activation checkpointing on/off** — recompute trades memory for extra
  FLOPs. It should *raise* power while lowering memory. Worth confirming: it is
  a rare knob that makes a job both slower and hungrier.
- **`torch.compile` on/off** — fusion removes kernel launches and memory
  round-trips. Expect higher power and shorter steps.
- **bf16 vs fp8** — the same watts-per-flop question the ladder raised, now on
  a real model.
- **Gradient accumulation depth** — more accumulation means fewer optimizer
  bursts, so a flatter power profile. Directly relevant to R3: a flat profile
  is a *worse* controllable load than a spiky one.

### Deliverable

A waterfall chart from 995 W down to the real training figure, with each
layer's cost labelled, plus a time-series of one step showing the phase
structure. That time-series is the thing to put in front of a grid person: it
is what a training cluster's power actually looks like, at a second resolution.

Budget: ~2 h on one B300 including model download. Roughly $10–20.

---

## R2 — Maximum inference

**Question:** how hard can a real serving stack be pushed, and how far apart
are the prefill and decode regimes in power terms?

This is where `nvidia-load-shave` already did the work. **Reuse
`scripts/benchmark.py` from that repo** — the concurrency ramp, the sustained
load phase, the TTFT/throughput capture — rather than writing a second one. It
tops out at 128 concurrent requests, which was a limitation there and needs
raising here.

### Model

**Llama-3.3 70B, FP8.** ~70 GB of weights leaves ~200 GB for KV cache, which
matters because the decode regime is exactly what large KV cache exercises. A
70B on one B300 is also a realistic single-GPU serving configuration, not a
contrivance.

### Rungs

Two regimes, deliberately separated, because averaging them hides the finding:

**Prefill-dominated** — long inputs, short outputs (e.g. 8k in, 16 out), high
concurrency. Compute-bound, should approach the GEMM numbers.
**Decode-dominated** — short inputs, long outputs (128 in, 2048 out), high
concurrency, large batch. Memory-bound and latency-bound; the DRAM-only rung at
70.9% is the prediction.

Then a sweep across the space between them: input length × output length ×
concurrency, recording %TDP at each point. Also worth toggling **chunked
prefill** and **CUDA graphs**, both of which change how continuous the power
draw is.

### The output that matters

**Power as a function of request rate.** Not peak power — the whole curve, from
idle through saturation. That is the demand-response curve for an inference
fleet: it says what a site's draw does as traffic varies, where the knee is,
and how much of the swing is available without touching a power cap at all.

Pair it with the load-shave result — capping power costs throughput
non-linearly — and there are two independent levers: shape the traffic, or cap
the hardware. Knowing which is cheaper at a given load point is the actual
operational question.

Budget: ~3 h on one B300 including model download. Roughly $15–30.

---

## R3 — Eight B300s

**Question:** what does a full node do, and can eight GPUs be made to move
together hard enough to matter upstream?

### What is genuinely new here

Everything so far has been one GPU. Three things only exist at node scale:

1. **The fabric draws power.** NVLink, the switch, inter-die traffic. A lone
   B300 cannot produce it, and `DESIGN.md` argues that is part of why a single
   card may be structurally unable to reach its own board rating.
2. **Phase alignment.** Eight GPUs swinging 881 W each *in phase* is a ~7 kW
   step. Staggered, it is nearly nothing. The difference between those two is a
   scheduling decision, which is precisely why it is interesting.
3. **Node power is measurable** — if, and only if, the node is genuinely bare
   metal with BMC access. This is the one hard platform requirement in the whole
   plan.

### Code needed

- **One runner process per device**, each with its own pinned sampler. Eight
  samplers on eight pinned cores; `isolcpus` if the host allows it.
- **A start barrier.** All processes agree an epoch on `CLOCK_MONOTONIC_RAW`
  before the steady window, so samples from different devices can be summed
  into a node curve. Without a common timebase, the aggregate is meaningless.
- **`--phase-offset-ms`** so duty cycles can be deliberately staggered — the
  control case against which "in phase" is measured.
- **An `allreduce` rung** using NCCL across all eight, to isolate what the
  fabric contributes on its own.
- **A node collator** that sums per-device curves and, where available, joins
  them against BMC or PDU samples.

### Node power measurement

Ranked, from `TESTPLAN.md`:

1. **Redfish / IPMI** from the BMC — true chassis draw including PSU losses.
   Typically ~1 Hz, so good for the sustained figure and the node-level step
   size, useless for millisecond transients.
2. **PSU PMBus** where exposed — higher rate on some platforms.
3. **Rack PDU per-outlet** — we build these (`DAME/control-pdu-anen`). For a
   colo or on-site node this is the easiest true wall measurement, and the
   number a grid conversation actually wants.
4. **Sum of eight NVML readings** — always recorded, always labelled an
   estimate of GPU-only draw, never presented as machine power.

**The gap between (1) and (4) is a headline result in itself**: it quantifies
how much of a GPU node's power is *not* the GPUs.

### Experiments

| Rung | Purpose |
|---|---|
| 8× synthetic, in phase | Maximum node draw and maximum node swing |
| 8× synthetic, staggered 45° apart | Same total energy, minimal swing — the control |
| NCCL all-reduce, varying message size | Fabric power in isolation |
| 8× tensor-parallel inference (R2 model, TP=8) | A real workload that uses the fabric |
| Node duty cycle, 1 ms → 10 s | The grid-facing number: how big a step, how fast |

### Safety

A deliberately square-waved ~9 kW load is hostile to upstream power delivery.
Before running the duty sweep: confirm the provider tolerates it, and on our own
hardware check breaker headroom and UPS transient response first. The hazard
being characterised applies to our own upstream too. Ramp the swing depth
gradually rather than starting at full amplitude, and stop if anything upstream
complains.

Budget: 8×B300 is capacity-block territory. Rehearse the entire campaign at R1
and R2 scale so the booked window is spent measuring, not debugging. Estimate
4–6 h at roughly $40–70/hr, so **$200–400** — the first genuinely expensive run
in this project, and the reason everything before it was done on cheap silicon.

---

## Sequence and blockers

```
  --op observe  ─────►  R1 training  ─────►  R2 inference  ─────►  R3 node
  (blocks both)         1×B300, ~2h          1×B300, ~3h           8×B300, 4-6h
                        waterfall            demand curve          fabric + phase
```

Open decisions, in the order they bite:

1. **`--op observe` first.** Nothing else starts without it. Half a day.
2. **R3 venue.** AWS `p6-b300.48xlarge` gives GPU-level NVML but no BMC, so no
   chassis power. If node power is required — and for the grid story it is —
   R3 needs genuinely bare metal. That changes the provider, not the config, so
   decide before booking. Spheron's single-GPU box was a KVM guest; ask whether
   they offer whole bare-metal nodes.
3. **Do R1 and R2 in one booking.** Same model download infrastructure, same
   box, and the marginal cost of the second campaign is an hour of GPU time.
4. **FP8 / FP4 on the synthetic ladder** is still outstanding from T2 and
   should ride along with R1, since both need `cublasLtMatmul` work.
