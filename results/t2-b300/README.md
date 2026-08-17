# T2 — B300 SXM6, 2026-08-17

Spheron, single **NVIDIA B300 SXM6 AC**, driver 610.57.04, CUDA 13.0
toolkit / 13.3 UMD, 275 GB HBM, **1100 W** default = max = enforced
limit. KVM guest with GPU passthrough, passwordless sudo — class 2:
full NVML control, no BMC, so no chassis power.

Probe: everything green. Power read, `POWER_INSTANT`, samples buffer
(120), energy counter, violation status, **limit write yes**,
persistence, clock lock, `SCHED_RR`, no MIG. O1 and O2 both measurable.

Note the limit constraints: default, max and current are all 1100 W, so
`--raise-power-limit` is a no-op here. There is no headroom above stock
to raise into.

## O1 — the Ampere result inverts completely

12s steady, 1100 W enforced, fp16 unless noted, size 8192.

| Rung | Avg W | %TDP |
|---|---|---|
| **tensor-only cuBLAS bf16** | **994.9** | **90.4** |
| tensor-only cuBLAS fp16 | 988.8 | 89.9 |
| tensor4+dram1 cuBLAS bf16 16k | 976.8 | 88.8 |
| tensor-only cuBLAS bf16 16k, 2 streams | 966.0 | 87.8 |
| tensor-only cuBLAS bf16 16k | 964.2 | 87.7 |
| dram-only | 779.7 | 70.9 |
| mix tensor+fma+dram cuBLAS | 733.2 | 66.7 |
| mix tensor2+fma1 cuBLAS | 671.2 | 61.0 |
| mix tensor+fma cuBLAS | 581.8 | 52.9 |
| tensor-only wmma | 460.6 | 41.9 |
| fma-only | 412.1 | 37.5 |
| mix tensor+fma wmma | 331.0 | 30.1 |

Nothing throttled. Clocks flat at ~1935-1970 MHz throughout.

**The ordering is inverted versus A10G.** On Ampere, FFMA was king
(75% TDP) and tensor-only was the weakest (37%). On B300 it is exactly
the other way round: tensor-only via cuBLAS reaches **90% of an 1100 W
part**, while the same FFMA kernel that dominated Ampere manages 37.5%.

Three things follow.

**1. The backend distinction was worth building.** `tensor-only wmma`
draws 460 W; `tensor-only cublas` draws 989 W on the same silicon and
the same nominal work. Warp-synchronous `wmma` cannot reach the 5th-gen
tensor core — only cuBLAS's tcgen05 path does, and the gap is **528 W**.
Had we run the original ladder on this box we would have measured 460 W,
concluded Blackwell's tensor cores were unremarkable, and been wrong by
more than a factor of two.

**2. The mixed-unit hypothesis is dead, for the opposite reason to
Ampere.** Every mix is worse than tensor-only, monotonically as work is
moved away from the tensor stream: 989 → 733 → 671 → 582 W. On Ampere
mixing lost because FFMA had the best watts-per-warp and mixing gave
warps away. Here it loses because the cuBLAS GEMM is the strongest
consumer and our kernel *displaces its blocks off the SMs*. The tcgen05
async issue model does free the instruction stream, but that does not
help when the competing resource is SM residency rather than issue
slots. Adding a small DRAM stream alongside a large GEMM (`tensor4+dram1`
at 976.8 W) is roughly power-neutral, not additive.

The falsifiable prediction stated before the run — that
`cublas-tensor+fma` exceeding `fma-only` would prove overlap — resolved
**no**: 582 W against 412 W is higher, but both are far below
tensor-only's 989 W, so the mix is still pure dilution.

**3. HBM3e is a big power consumer in its own right.** `dram-only` at
780 W is 71% of TDP from memory traffic alone, and it beat every mixed
configuration. On A10G the same role drew 47%.

## O1 verdict: no, and it is close

**Best sustained: 90.4% of an 1100 W enforced limit, with no throttle
bit ever set.** So the part was never power-limited — the remaining
~105 W is workload shortfall, not the controller clamping us. Consistent
with the working hypothesis, and much nearer the ceiling than the ~71%
that real vLLM inference reached on A10G-class hardware.

What might close the gap and was not tested: FP8 and FP4. Both need
`cublasLtMatmul` — `cublasGemmEx` cannot express them — and FP4
(`mxf4`/`mxf4nvf4`) additionally requires the `sm_103a` arch target,
which we now build. That is the obvious next experiment.

## O2 — a 0.88 kW swing on one GPU

`--duty-on-ms 200 --duty-off-ms 200`, tensor-only cuBLAS bf16, 100 Hz
requested (77.9 Hz achieved).

| Path | Min W | Max W | Swing |
|---|---|---|---|
| `GetPowerUsage` (averaged) | 534.7 | 1080.6 | 545.9 |
| `POWER_INSTANT` | 195.6 | 1076.4 | **880.8** |

**Peak instantaneous draw is 1080.6 W against an 1100 W cap — 98.2%.**
Bursting reaches materially closer to the limit than any sustained rung
did, which is itself a finding: the ceiling is easier to touch
transiently than to hold.

The averaged path again fails at the bottom, and much worse than on
A10G: it reports a 535 W floor where the instantaneous counter says
196 W. **It understates the swing by 335 W, or 38%.** Same conclusion as
Ampere, at seven times the magnitude.

For the grid conversation, the headline is simple: **one B300 can be
made to swing ~0.9 kW at a 2.5 Hz square wave.** Eight of them in phase
on a node is a ~7 kW step, and that is before the uncore.

Energy-integral cross-check: -3.6% gap, so the sampled curve accounts
for the energy — no evidence of unresolved area at this period.

**Suspect number:** `power_violation_us` reported 6.65e9 µs (~1.8 h),
which is far longer than the rung. That is almost certainly a
since-boot counter rather than the delta the code intends to compute,
so the violation-time field should not be trusted until the start/end
anchoring is re-checked. Flagged, not used in any conclusion above.

## Caveats

- Single GPU. No NVLink or fabric load, which `DESIGN.md` argues may
  make a lone B300 structurally unable to reach its board rating.
- In-band measurement only. No BMC on a KVM guest, so every figure here
  is what NVML reported, not what the board drew.
- 12s steady windows, one run each. Enough to rank configurations, not
  enough for a published number.
