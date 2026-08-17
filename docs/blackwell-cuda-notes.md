# Blackwell from a CUDA programming perspective

Status: **living document — unverified until checked against the CUDA
docs for our pinned toolkit and confirmed on hardware.**
Date started: 2026-08-17

Purpose: the architectural understanding needed to write a power virus
for Blackwell and to interpret what the power numbers mean. This is
working knowledge for the project, not a tutorial.

**How to read this document.** Items marked **[V]** are verified
against official documentation or observed on hardware, with the source
noted. Items marked **[?]** are working assumptions that must be
checked before any of them shows up in a report or drives a design
decision. Blackwell details moved quickly across CUDA releases, so
assume anything unmarked is provisional. The discipline matters here:
this document exists to prevent expensive assumptions, so an unverified
claim recorded as verified is worse than no claim at all.

---

## 1. Which Blackwell are we talking about

Blackwell is not one architecture, and conflating the two costs real
money on this project.

| Family | Compute capability | Parts | Tensor core generation |
|---|---|---|---|
| Datacenter Blackwell | `sm_100` **[?]** | B100, B200 | 5th gen, tensor memory |
| Blackwell Ultra | `sm_103` **[?]** | B300, GB300 | 5th gen, tensor memory |
| Consumer / workstation | `sm_120` **[?]** | RTX 5090, RTX PRO 6000 | Tensor cores without the datacenter tensor-memory path |

**Practical consequence for us:** the tensor-core inner loop that we
expect to dominate power on B300 cannot be developed or validated on a
5090. Two backends, one interface. See `TESTPLAN.md`.

### Arch-specific target suffixes **[?]**

Family-specific instructions — including, we believe, the 5th-gen
tensor core instructions — require an *architecture-specific* target
rather than the plain compute capability: `sm_100a` / `sm_103a` /
`sm_120a` rather than `sm_100` / `sm_103` / `sm_120`. Code compiled for
the plain target will not have access to them, and PTX built for an
`a` target is not forward-portable to later architectures the way
ordinary PTX is.

**Verify first.** This affects our CMake flags directly: if it is
right, the fat binary needs both the portable and the arch-specific
variants, and a build that "compiles fine" for `sm_103` may silently
lack the very instructions the power virus depends on. Check against
the CUDA Programming Guide for our pinned toolkit before trusting
`TESTPLAN.md`'s T0 gate.

---

## 2. The SM, and where the watts live

For power purposes the SM is a collection of units that draw
independently, fed by a shared instruction issue path. Roughly, per
SM partition:

- **CUDA cores** — FP32 / INT32 pipes. Classic FFMA throughput.
- **FP64 pipe** — heavily cut on datacenter Blackwell relative to
  Hopper **[?]**. If so, FP64 is a poor power path here, which is a
  change from older GPUs where DGEMM was a reliable virus. Worth
  measuring precisely because it inverts prior intuition.
- **Tensor cores** — the dominant flops, and presumably the dominant
  power, at low precision.
- **SFU** — transcendentals. Small unit, but a genuinely separate one,
  so it contributes to a mix.
- **Register file** — large, and reads/writes cost real energy. Operand
  collector activity is a power path in its own right.
- **Shared memory / L1** — configurable split; distributed shared
  memory across a cluster adds a path Hopper introduced.
- **LSU / TMA** — address generation and bulk async copy.

Off the SM: **L2**, **HBM stacks**, **NVLink and the die-to-die
fabric**, and the memory controllers. On a multi-die part these are a
substantial and often overlooked share of total board power.

**The power-virus implication:** these draw from different budgets and
can in principle overlap, but they are fed by one issue path per warp
scheduler. Concurrency between them is bounded by instruction issue,
not by the units themselves. This is the central tension in the
`powervirus` design and the reason the pairwise attribution rungs in
`DESIGN.md` exist.

---

## 3. Tensor cores and tensor memory (`tcgen05`) **[?]**

The 5th-generation tensor core on datacenter Blackwell introduces a
dedicated **tensor memory (TMEM)** as the accumulator/operand store,
distinct from the register file, along with a new instruction family
(referred to as `tcgen05.*` in PTX).

What we believe matters for us:

- MMA operations are issued against TMEM rather than accumulating in
  registers, which changes register pressure characteristics
  substantially versus Hopper's `wgmma`.
- The operation is **asynchronous** and often **single-thread-issued**
  on behalf of a larger group, rather than warp-collective in the way
  earlier `mma.sync` was.
- Some shapes operate across a **CTA pair** — two cooperating thread
  blocks in a cluster — rather than a single block.

Each of these has a direct consequence for a power virus: if the MMA
is asynchronous and issued by one thread, then **the issue-bandwidth
objection above is weaker than it first appears** — a single thread can
keep the tensor core busy while other warps saturate the FMA pipe and
the memory path. That would make a mixed virus much more effective than
on earlier architectures, and it is the single most important thing on
this page to verify, because the whole `--mix-*` hypothesis rests on
it.

**To verify:** exact PTX instruction names and shapes for our toolkit;
TMEM allocation and its occupancy cost; whether the CTA-pair
requirement constrains our launch geometry; what the compiler emits
from CUTLASS-style code versus hand-written PTX.

---

## 4. Asynchrony, and why it matters for power

Modern NVIDIA GPUs are built around overlapping work, and a power virus
is essentially an exercise in maximizing overlap:

- **TMA** — bulk async global↔shared copies driven by a descriptor,
  freeing threads from address arithmetic. For us: a way to keep DRAM
  traffic running underneath a compute loop without spending issue
  slots on it. **Probably the key to combining tensor and memory
  activity.**
- **`mbarrier` / async barriers** — producer/consumer synchronization
  without full block barriers.
- **Thread block clusters (CGA)** — blocks co-scheduled on a GPC with
  distributed shared memory between them.
- **Warp specialization** — dedicating some warps to data movement and
  others to compute, standard in modern GEMM kernels. For a virus:
  dedicate warp groups to *different units* deliberately, rather than
  to a producer/consumer split.
- **`setmaxnreg`** — dynamic register reallocation between warp groups,
  letting producer warps give registers to compute warps.

Warp specialization is likely the right structure for `powervirus`:
one warp group driving tensor cores, another running an FFMA chain,
another issuing TMA loads that miss L2 — all resident on the same SM.
That is a far better shot at genuine concurrency than interleaving
instruction types in one stream.

---

## 5. Precision formats and the watts-per-flop question

Blackwell's headline is very low precision — FP8 and the new
narrower formats (FP6 / FP4 **[?]**), with a second-generation
Transformer Engine and microscaling block formats **[?]**.

For power characterization the ranking we care about is *not* the
throughput ranking:

| Format | Throughput | Energy per op | Total power — the open question |
|---|---|---|---|
| FP64 | Low **[?]** | High per op | Probably low total. Verify. |
| FP32 (FFMA) | Moderate | Moderate | The classic baseline; the honest reference point |
| TF32 / BF16 / FP16 tensor | High | Lower | Likely strong |
| FP8 | Very high | Lower still | Strong, but may become memory-starved |
| FP4 / FP6 **[?]** | Highest | Lowest per op | **Unknown** — more ops/s but each cheaper, and operands are consumed so fast the bottleneck may move to memory |

The hypothesis worth testing explicitly: **maximum power is probably
not at maximum throughput.** A format fast enough to starve itself on
operand supply spends cycles waiting, and waiting is cheap. There may
be a sweet spot at middling precision where the arithmetic is still
expensive per operation but the operand path can keep up. This is
precisely the sweep described in `DESIGN.md` §"power attribution
breakdown", and it is one of the more publishable things here.

---

## 6. Clocks, power capping, and what the controller does

- Power capping is a closed-loop DVFS response: as draw approaches the
  cap, clocks drop. Practical effect — **you cannot sit at 100% of the
  cap by definition of how the loop works**; you converge just below it
  while the controller trims frequency.
- `nvidia-load-shave` measured this on A10G: at a 240 W cap the part
  drew 225 W, and clocks fell from 1710 to 1686 MHz. The controller
  leaves margin.
- Throttle reason bits distinguish cause: SW power cap (0x4), HW
  slowdown (0x8), SW thermal (0x20 **[?]**), HW thermal (0x40), HW
  power brake (0x80).
- Locked clocks (`boost-off`) remove DVFS from the loop and make rungs
  comparable, at the cost of realism. Both profiles, always.

**Consequence for O1's definition:** "100% of the enforced limit" may
be unreachable *as a control-theory matter* rather than a workload
matter. If we consistently converge at 97–99% with only the power-cap
bit set, that is arguably success, and `DESIGN.md`'s ≥99% threshold may
need revisiting once we see how tight the loop actually is on
Blackwell. Decide with data from T1b, not now.

---

## 7. Multi-GPU and node-level (relevant from v0.3 / T3)

- NVLink and the switch fabric draw real power and are entirely absent
  from single-GPU numbers. A lone B300 may be structurally unable to
  reach its own board rating.
- On GB-series parts the CPU shares a package/tray power budget with
  the GPUs **[?]**, so "GPU power" and "node power" can move
  independently in ways single-device NVML cannot show.
- NCCL all-reduce is the natural way to add fabric load, and should be
  a rung in its own right in the attribution ladder.

---

## 8. Open items to verify

Ordered by how much they would change the design:

1. `tcgen05` semantics — async, single-thread issue, CTA-pair shapes.
   The `--mix-*` hypothesis depends on it (§3).
2. `sm_100a` / `sm_103a` arch-specific targets and what our CMake must
   emit (§1).
3. FP4/FP6 availability from plain CUDA C++ versus requiring CUTLASS or
   hand-written PTX (§5).
4. Whether datacenter Blackwell FP64 is as weak as assumed (§2).
5. TMEM capacity and allocation rules, and their effect on occupancy
   (§3).
6. Actual DVFS control-loop behaviour near the cap on Blackwell (§6).
7. B300 board TDP — our scanner records 1200 W for
   `p6-b300.48xlarge`; confirm against NVIDIA's specification and
   against `nvmlDeviceGetPowerManagementLimitConstraints` on the real
   part.

## Sources to work through

- CUDA C++ Programming Guide — compute capability chapter for our
  pinned toolkit version
- PTX ISA reference — `tcgen05` family, TMA, `mbarrier`, cluster
  instructions
- CUTLASS: the Blackwell kernel implementations are the most reliable
  worked examples of the new tensor core path
- NVIDIA Blackwell architecture whitepapers (B200 and Blackwell Ultra)
- NVML API reference — power, clocks, throttle reasons, violation
  status
- Published power-virus / peak-power literature for methodology and
  for what fraction of TDP others have achieved
