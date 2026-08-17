# gpu-power-lab — Development and test workflow

Status: **draft**
Date: 2026-08-17
Target architecture: **Blackwell**, ending on a bare-metal 8×B300 node.

Two rules shape the whole plan:

1. **Never debug on expensive silicon.** Each tier has an entry gate,
   and nothing moves up until the gate below is green. Cost per hour
   rises roughly two orders of magnitude from T0 to T3, while the
   failures that actually waste that money — a schema mismatch, an
   unhandled NVML return, a kernel that won't launch — are all
   findable on a GPU that costs less than a CI runner.
2. **Debugging and measurement have different platform requirements.**
   Debugging can happen anywhere the code runs, including cheap shared
   containers. Measurement cannot: it needs bare metal. Keeping these
   separate is what makes rule 1 affordable.

## Platform constraint: bare metal, not containers

**Every tier that produces a power number requires a machine where we
own the NVIDIA driver and have real root. Container and Kubernetes
based GPU platforms cannot run this harness.** This is not a
preference — the APIs the project is built on are unavailable there.

We already learned this the hard way on `nvidia-load-shave`: RunPod
blocked `nvidia-smi -pl` and clock locking on a shared H100, which is
what forced that project onto AWS in the first place. Same wall here,
and higher, because this harness needs *more* privilege than that one
did.

What a containerized / k8s GPU platform denies us:

| Need | Why containers fail |
|---|---|
| `nvmlDeviceSetPowerManagementLimit` — raise limit to max (O1) | Host owns the driver; write calls are blocked or silently ignored |
| Persistence mode | Driver-level, host-owned |
| Clock locking (`boost-off` profile) | Same |
| `SCHED_RR` on the sampler thread | Needs `CAP_SYS_NICE`, rarely granted |
| Core pinning / `isolcpus` for 8 samplers | No control over host CPU topology |
| Un-shared, whole-device power readings | NVML reports device-wide power — on a shared host that includes the neighbour's work |
| Clean idle floor for duty cycles (O2) | A noisy neighbour means the "off" phase is not idle, which destroys the swing measurement |
| BMC / Redfish / chassis power (T3) | No hardware access at all |

The noisy-neighbour point deserves emphasis, because it is subtler than
the permissions ones and it invalidates results rather than blocking
them: **O2 measures the swing between the idle floor and the peak.**
On a shared GPU or a shared chassis, neither end of that swing is ours.
The run will complete and produce numbers that are quietly wrong.

Three platform classes, and what each is good for:

1. **Container / k8s on shared hardware** — RunPod pods, Vast
   containers, CoreWeave k8s, any "serverless GPU", any MIG slice.
   **No measurement tier may use these.** Usable only for
   does-it-compile-and-run checks.
2. **Dedicated VM with full GPU passthrough**, where the guest owns the
   driver and we have root — the AWS `p6`/`p5`/`g6e` family behaves
   this way. NVML power control works here; `nvidia-load-shave`
   successfully set power limits on a `g5.xlarge` VM. Good enough for
   **all GPU-level work: O1 and O2 on single and multi-GPU.** What it
   cannot give is chassis-level power, because there is no BMC.
3. **True bare metal** — `.metal` instances, dedicated servers
   (Hetzner, OVH, Latitude, DataCrunch bare-metal SKUs), colo, or our
   own hardware. Everything in class 2, **plus** BMC/Redfish/IPMI and
   PDU access. **Required for T3's total machine power**, and the
   safest default everywhere else.

Practical rule: prefer class 3; accept class 2 when it is materially
cheaper or the only place a part is available; never class 1.

The first thing to verify on any new box, before running a campaign, is
a **capability probe** — the runner gains a `--probe` mode that
attempts each privileged operation and reports what the platform
actually permits (power-limit write, persistence, clock lock,
`SCHED_RR`, samples buffer, energy counter, BMC reachability). Run it
as the first step of every campaign, record the result in the campaign
manifest, and fail fast with a clear message rather than discovering
mid-sweep that half the rungs were unenforced. This is cheap to write
at T0 and it turns "why are all these numbers identical?" into a
one-line answer.

## The architecture problem that shapes everything

Blackwell is not one target. It is two, and cheap Blackwell is the
wrong one for the part of the code that matters most:

| Part | Arch | tcgen05 / tensor memory | Role here |
|---|---|---|---|
| RTX 5090 | `sm_120` | **No** — `mma.sync`-style tensor cores | Cheap harness validation |
| RTX PRO 6000 Blackwell | `sm_120` | **No** | Cheap validation, 600 W class |
| B200 SXM | `sm_100` | Yes | Datacenter dress rehearsal |
| B300 SXM (Blackwell Ultra) | `sm_103` | Yes | The target |

The consequence: a cheap Blackwell box validates the build, the NVML
paths, the duty-cycle timing, the schema, the orchestrator, and the
analysis maths — **but not the `powervirus` tensor-core inner loop that
is supposed to reach the ceiling on B300.** The peak-power path on
`sm_100`/`sm_103` goes through 5th-gen tensor cores and tensor memory,
which `sm_120` does not have.

So `powervirus` is built as one interface with two backends behind it:

```
gpl_powervirus_run()
  ├── backend_sm120.cu   — mma.sync + FFMA + DRAM mix   (T1 validates)
  └── backend_sm100.cu   — tcgen05 + FFMA + DRAM mix    (T2 validates, sm_103 too)
```

Both must satisfy the same rung contract, and CI compiles both for all
of `sm_120`, `sm_100`, `sm_103` (`nvcc -gencode` fat binary) even where
it cannot run them. **A compile failure on `sm_103` must never be
discovered on a rented B300.**

There is one cheap way to touch the real ISA early: RunPod lists a
**B300 MIG** slice. A MIG instance cannot set power limits and reports
device-wide power that isn't yours, so it is useless for O1/O2 numbers —
but it will run and correctness-check the `sm_103` backend for minutes
of spend. Use it as a T1.5 compile-and-run gate only.

## Tiers

### T0 — It compiles. Cost: $0. Days, not weeks.

Deliberately thin. There is little value in elaborately simulating a
GPU we are about to rent for a couple of dollars an hour — real
debugging happens on real silicon at T1. T0 exists only to stop us
paying for failures that a compiler would have caught.

- **Compiles for all three arches** — `sm_120`, `sm_100`, `sm_103`, one
  fat binary, in CI on a CPU-only runner. A compile failure on
  `sm_103` must never be discovered on a rented B300.
- **No register spills**, per arch. `nvcc --ptxas-options=-v` reports
  this for free at build time. Spilling silently turns the power virus
  from compute-bound into memory-bound, so fail the build on it.
- **The inner loop survived the optimizer.** One `cuobjdump -sass`
  assertion that the tensor-core loop still issues `tcgen05` MMA back
  to back with the FFMA chain interleaved. A power virus the compiler
  quietly folded into something tame looks fine until it draws 400 W on
  a 1200 W part. This is a grep over compiler output, not a simulator.
- **The existing `pytest` suite still passes**, plus a small unit test
  for the transient maths — feed a canned NDJSON with known overshoot
  and slew through the analysis path and assert the numbers come back
  right. Minutes of work; it means a wrong number at T1 points at the
  hardware, not at our arithmetic.
- **Plan files written** — `plans/o1-ceiling.yaml`,
  `plans/o2-transient.yaml` — and validated against the schema, so the
  first paid hour starts a campaign instead of starting an argument.

- **The architecture study** — `docs/blackwell-cuda-notes.md` worked
  through against the CUDA Programming Guide, the PTX ISA reference and
  CUTLASS's Blackwell kernels, with its `[?]` markers resolved to
  `[V]`. This is the one substantial piece of desk work that pays for
  itself: item 1 on its open-items list (`tcgen05` async, single-thread
  issue, CTA-pair shapes) determines whether the mixed-unit virus is
  even a coherent idea, and item 2 (`sm_100a`/`sm_103a` targets)
  determines whether our fat binary actually contains the tensor-core
  instructions at all. Getting either wrong turns a rented B300 into an
  expensive way to run the wrong kernel.

That is the whole of T0. Everything else waits for a GPU.

### T1 — Cheap Blackwell (`sm_120`). Where the debugging happens.

RTX 5090 or RTX PRO 6000. Split in two by *purpose*, because the
platform constraint applies to measurement but not to debugging:

**T1a — debug box (container is fine). ~$0.50–1.50/hr.** RunPod, Vast,
anything cheap. We are not measuring anything here, so shared hardware,
no root and a blocked power-limit write cost us nothing. This is where
the code actually gets shaken out: does the kernel launch, does the
duty scheduler keep time, does the sampler thread run, does a campaign
complete, does the summary validate, does the upload work. Iterate here
freely — it is cheaper than most CI runners. Expect
`nvmlDeviceSetPowerManagementLimit` to fail; the runner should record
`limit_raise: denied` and carry on rather than abort, which is itself a
behaviour worth testing here.

**T1b — measurement box (bare metal, mandatory). Costlier, short runs.**
Once the code is debugged, move the same binary to a dedicated
bare-metal 5090 / PRO 6000 for the numbers: Hetzner lists RTX PRO 6000,
and OVH, Latitude and LeaderGPU are comparable; a `.metal` cloud
instance or any workstation we can get root on also qualifies. Runs
here are short and scripted because the debugging is already done.

Filter `data/rollups/gpu/latest.json` in `DAME/llm-providers` by
platform class before costing anything — the cheapest listings in it
are container platforms, fine for T1a and useless for T1b.

What T1b genuinely proves:

- Real NVML on real silicon: all four sampling paths from `DESIGN.md`
  — `GetPowerUsage`, `POWER_INSTANT` vs `POWER_AVERAGE`,
  `GetSamples` buffer drain, `GetViolationStatus`.
- **The driver's power-averaging window, measured empirically.** Drive
  a known square wave, find where INSTANT and AVERAGE diverge. Do this
  per device family; it is the number that decides whether an O2 claim
  is resolvable at all.
- Duty-cycle machinery end to end: shortest clean period, launch
  overhead floor, cycle jitter.
- The `sm_120` `powervirus` backend vs plain cuBLAS GEMM — does the
  mixed-unit kernel actually beat peak-FLOPS GEMM on power? If the
  mix hypothesis is wrong, we learn it here for $1.50/hr, not on a
  B300.
- A 575–600 W part is a legitimate O1/O2 subject in its own right.
  Publish it as the first result.

**On out-of-band ground truth.** `DESIGN.md` wants an external
measurement before any over-limit claim is published. Bench
instrumentation on hardware we own is **not available at this stage**,
so that check is deferred rather than dropped, and the consequence is
explicit: until it exists, any overshoot number we produce is an
*in-band* result and must be reported as one — "NVML's instantaneous
counter reported X% of the enforced limit", not "the GPU drew X%".

The strongest cross-checks available without external instruments,
which T1 should establish as standard practice:

- `POWER_INSTANT` vs `POWER_AVERAGE` divergence, which bounds what the
  driver is smoothing.
- The `GetSamples` buffer, whose device-side timestamps are independent
  of our thread's wakeup jitter.
- `GetViolationStatus` throttled-microseconds and the `HW_POWER_BRAKE`
  bit — the controller reacting is corroborating evidence for an
  excursion we cannot directly resolve.
- Energy: integrating `nvmlDeviceGetTotalEnergyConsumption` over a duty
  rung and comparing it against the integral of the sampled power
  curve. A mismatch means the sampled curve is missing area — which is
  exactly what an unresolved spike looks like.

That last one is worth emphasizing: **the energy counter is a
monotonic accumulator that cannot miss a spike the way a sampler
can.** It won't tell us the peak, but a persistent gap between energy
and integrated power is quantitative evidence that peaks exist above
what we can see. It is the closest thing to ground truth reachable
without a bench, and it costs nothing to add.

The BMC/PDU path at T3 (below) remains the eventual real answer.

**Gate to T2:** full O1+O2 campaign completes unattended on bare-metal
`sm_120`; overshoot/slew numbers reproduce across three runs; averaging
window documented; results uploaded to S3 by the campaign itself. In
other words, the only thing left untested is the tensor-core backend
that `sm_120` cannot run.

### T2 — Single B300 (`sm_103`). Cost: ~$3–6/hr per GPU

The rollup lists single-GPU B300 at DataCrunch, Nebius, CoreWeave,
Oracle, Scaleway, RunPod and Vast — but most of those are class 1
container platforms and are therefore out for measurement. **Required
here: a whole non-MIG device on bare metal or a passthrough VM with
root**, which in practice means a bare-metal SKU or a `p6-b300`-class
instance rather than a pod. Check platform class per provider before
booking; it is the difference between a result and a wasted day.

Optionally stage through a single **B200** (`sm_100`) first — it is
cheaper and more available, and it validates the tcgen05 backend one
arch before the target. Skip it only if B300 supply is good.

What T2 is for:

- The `sm_103` tcgen05 backend at full size, and the mix sweep that
  answers O1: **can we pin a 1200 W-class part at 100% of its enforced
  limit, power-limited and not thermally limited?**
- The O2 period sweep, 1 ms → 10 s, on hardware with a far larger swing
  than T1 — this is where overshoot, if it exists, should be most
  visible.
- First look at how a Blackwell Ultra part's own power controller
  behaves: how fast it clamps, whether `HW_POWER_BRAKE` fires.

Discipline: the campaign runs **unattended** — boot, run the whole plan
from a systemd unit, upload to S3, terminate. Interactive debugging on
a B300 is the single easiest way to waste money here. Reuse the
lifecycle automation from `DAME/nvidia-load-shave` (`aws_boot.py`,
Makefile targets) rather than writing it again.

**Gate to T3:** single-GPU O1 and O2 results in hand, campaign runs
start-to-finish unattended, per-rung wall-clock known well enough to
budget the 8× node run to ±20%.

### T3 — Bare-metal 8×B300 node. Cost: ~$30–70/hr

`p6-b300.48xlarge` is already in `check_gpu_availability.py`; run it
for live pricing and capacity-block options. **But see the platform
constraint below — AWS may be the wrong venue for this tier.**

New problems that only appear at T3:

- **Eight samplers, one timebase.** One runner process per device,
  each with its own pinned sampler thread. Samples must be correlated
  across devices to compute a node-level power curve, so all processes
  stamp from `CLOCK_MONOTONIC_RAW` against a barrier-established
  epoch. Pin samplers to dedicated cores (`isolcpus`) so eight of them
  plus eight workload threads don't interfere.
- **Phase alignment is the experiment.** Eight GPUs swinging ~1 kW each
  *in phase* is a ~10 kW step at the node. Aligned vs deliberately
  staggered duty cycles is the headline T3 comparison, and it is the
  result that matters for grid work.
- **Aggregate vs measured.** Sum-of-NVML across eight GPUs is not node
  power. The gap — CPUs, NVSwitch, DRAM, fans, PSU losses — is
  precisely what we need to quantify.

#### Total machine power — what's actually available

Ranked by trustworthiness:

1. **Redfish / IPMI from the BMC** — `PowerSubsystem`/`Chassis.Power`
   or `ipmitool dcmi power reading`. True chassis draw including PSU
   losses. Requires **real** bare metal with BMC access: a neocloud
   bare-metal SKU, a colo box, or our own hardware. Sampling rate is
   typically ~1 Hz and it is *slow* — good for the sustained O1 number
   and for the node-level step size, useless for millisecond
   transients.
2. **PSU PMBus telemetry** where the BMC exposes per-PSU rails. Higher
   rate than DCMI on some platforms; check what the specific chassis
   offers.
3. **Rack PDU per-outlet metering** — we already build and control
   these (`DAME/control-pdu-anen`). For a colo or on-site node this is
   the easiest true wall measurement, and it is the number a grid
   conversation actually cares about.
4. **Sum of NVML across 8 devices** — always recorded, but labelled as
   an estimate of GPU-only draw, never as machine power.

**The constraint to resolve early:** AWS `p6-b300.48xlarge` is a
virtualized instance — no BMC, no Redfish, no PDU. It can give us the
8×GPU aggregate and the phase-alignment experiment, but **not** total
machine power. If chassis-level power is a hard requirement (and for
the grid story it probably is), T3 needs a genuinely bare-metal
provider that exposes BMC telemetry, or an 8×B300 node we host
ourselves behind our own metered PDU. **Decide this before booking
T3** — it changes the provider, not just the config.

Also expect an availability constraint: 8×B300 is capacity-block
territory, so T3 is a scheduled run, not an on-demand one. Have the
plan file, the campaign, and the teardown fully rehearsed at T2 so the
booked window is spent measuring.

**Node-level safety:** ~10 kW of deliberately square-wave load is
hostile to upstream power delivery. On rented infrastructure, confirm
the provider tolerates it. On our own hardware, check breaker headroom
and UPS transient response first — as `DESIGN.md` says, the hazard we
are characterizing applies to our own upstream too.

## Build and test mechanics

- **One CMake, three arches.** `-gencode arch=compute_120,code=sm_120
  -gencode arch=compute_100,code=sm_100 -gencode
  arch=compute_103,code=sm_103`, fat binary, so one artifact runs
  everywhere and arch selection is a runtime dispatch, not a rebuild.
- **CI** (GitHub Actions, CPU-only): compile all arches, run the C unit
  tests and mock e2e, run `pytest`, validate golden summaries against
  the schema. No GPU runner needed until we want per-PR smoke tests,
  which is a T1-class expense and can wait.
- **Artifact promotion.** The exact binary tested at T0/T1 is the one
  shipped to T2/T3 — build once, tarball with the plan and venv, pull
  from S3 in user-data. No compiling on the expensive box.
- **Per-tier results are kept and compared.** Each tier writes to
  `s3://<bucket>/campaigns/<tier>/<campaign_id>/`. The 5090 numbers are
  the regression baseline that tells us if a code change broke the
  measurement, cheaply.

## Sequence

```
T0  it compiles       ──►  T1a  cheap 5090 pod      ──►  T1b  5090 / PRO 6000
   3 arches, no spills,     container OK — debug          BARE METAL — measure
   SASS check, plan files   here, cheaply                 O1 + O2 at 600 W
   ▲ we are here                                                 │
                                                                 ▼
                                                          T1.5  B300 MIG
                                                          run sm_103 kernel
                                                          (not a measurement)
                                                                 │
        T3  8×B300 bare metal   ◄──   T2  single B300   ◄────────┘
        (+ node power via BMC/PDU)    (◄ optional B200)
        BARE METAL                    bare metal or passthrough VM

   Class 1 (container / k8s / MIG): debugging only, never measurement.
   Out-of-band ground truth: deferred — no bench instrumentation yet, so
   overshoot numbers are reported as in-band results until there is.
```

## Open decisions

1. **T3 venue.** Given the bare-metal requirement, AWS
   `p6-b300.48xlarge` only half qualifies: GPU-level NVML works on a
   passthrough VM, but there is no BMC, so no total machine power.
   Since node-level draw is an explicit goal, the default answer is a
   genuinely bare-metal 8×B300 node — neocloud bare-metal SKU, colo, or
   our own hardware behind a metered PDU. Confirm the provider and
   whether their BMC exposes Redfish power before booking. Blocks T3
   only.
2. When does out-of-band measurement become available, and by which
   route — a metered PDU outlet at one of our own sites, BMC telemetry
   on a bare-metal rental, or bench instrumentation later? Not blocking
   any tier; it blocks only the point at which an overshoot claim can
   drop its in-band qualifier.
3. Stage through B200 at T2, or go straight to B300? Depends on B300
   single-GPU supply at the time.
4. Capacity-block lead time for 8×B300 — determines how far ahead T3
   has to be scheduled, and therefore when T2 must be finished.
