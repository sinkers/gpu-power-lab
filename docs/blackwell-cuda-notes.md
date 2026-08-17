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
| Datacenter Blackwell | `sm_100` | B100, B200 **[?]** | 5th gen, tensor memory |
| Blackwell Ultra | `sm_103` | B300, GB300 **[?]** | 5th gen, tensor memory |
| Consumer / workstation | `sm_120` | RTX 5090, RTX PRO 6000 **[?]** | Tensor cores without the datacenter tensor-memory path |

The compute capabilities themselves are **[V]** — `sm_100`, `sm_103`,
`sm_110`, `sm_120`, `sm_121` and their `f`/`a` variants are all in
nvcc's allowed-target list. The *product* mapping in the middle column
is still **[?]**: confirm against NVIDIA's GPU compute-capability table,
or just read it off the device at T1/T2.

**Practical consequence for us:** the tensor-core inner loop that we
expect to dominate power on B300 cannot be developed or validated on a
5090. Two backends, one interface. See `TESTPLAN.md`.

### Target suffixes: baseline, `f`, and `a` **[V]**

**Verified** against the PTX ISA guide §11.1.2 (`.target`) and the CUDA
Programming Guide §5.1.2 (v13.3). There are *three* target classes, not
two — the `f` family targets were the piece we did not know about:

| Target | Meaning | Portability |
|---|---|---|
| `sm_XY` | Baseline feature set | Onion-layer: runs on later generations |
| `sm_XYf` | **Family**-specific features | Runs only on devices in the same family, that generation or later |
| `sm_XYa` | **Architecture**-specific features | Runs *only* on that exact compute capability |

Exact wording from the PTX guide: *"Target architectures with suffix
'a' … do not follow the onion layer model. Therefore, PTX code
generated for such targets cannot be run on later generation devices."*
And: *"Target architectures with suffix 'f' … can run only on later
generation devices in the same family."*

Declared families **[V]**: `sm_10x` = {`sm_100f`, `sm_103f`, future
`sm_10x`}; `sm_11x` = {`sm_110f`, `sm_101f`}; `sm_12x` = {`sm_120f`,
`sm_121f`}. Note `sm_101` was renamed to `sm_110` from PTX ISA 9.0.

ptxas compile rules **[V]**, from the nvcc guide §4.2.9.1.13:

> PTX for `.target sm_XY` can be compiled to all GPU targets `sm_MN`,
> `sm_MNa`, `sm_MNf` where MN >= XY. PTX for `.target sm_XYf` can be
> compiled to GPU targets `sm_XZ`, `sm_XZf`, `sm_XZa` where Z >= Y and
> `sm_XY` and `sm_XZ` belong in same family. PTX with `.target sm_XYa`
> can only be compiled to GPU target `sm_XYa`.

**The finding that matters: `sm_100f` covers both B200 and B300.**
Because 10.0 and 10.3 are in the same family, one family-target build
gets the 5th-gen tensor core on both parts — we do not need separate
arch-specific builds just to span the datacenter line. We still need
`sm_103a` for the handful of MMA kinds excluded from family targets
(see §3).

Applied in `runner/CMakeLists.txt`: `sm_120a`, `sm_100f`, `sm_100a`,
`sm_103a`, set through explicit `-gencode` rather than
`CMAKE_CUDA_ARCHITECTURES` so the suffixes work across CMake versions.

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

## 3. Tensor cores and tensor memory (`tcgen05`) **[V]**

Verified against PTX ISA §9.7.17, "TensorCore 5th Generation Family
Instructions". This section was the project's biggest open question and
the answer is favourable.

### Tensor memory (TMEM) **[V]**

Dedicated on-chip memory for tensor core operations, separate from the
register file. On `sm_100a`/`sm_100f`: **512 columns × 128 lanes per
CTA, 32 bits per cell** — 256 KB per CTA. Addresses are 32-bit, packed
as lane index in the high half and column index in the low half.

Allocation rules **[V]**:

- Dynamically allocated, **by a single warp**, via `tcgen05.alloc`.
- Allocation granularity is **32 columns**, and the count must be a
  **power of two**. Allocating a column allocates all 128 lanes.
- Everything allocated must be explicitly deallocated before the kernel
  exits, or behaviour is undefined.
- `tcgen05.ld`/`st` from one warp reach only **a quarter** of the CTA's
  TMEM, so a full warpgroup is needed to cover it.

### Issue granularity — the important one **[V]**

From PTX ISA Table 49:

| Operation | `.cta_group` | Who issues |
|---|---|---|
| `.mma`, `.cp`, `.shift`, `.commit` | `::1` | **A single thread** in the CTA initiates the operation |
| `.mma`, `.cp`, `.shift`, `.commit` | `::2` | A single thread from the CTA pair; the peer CTA must be active and not exited |
| `.alloc`, `.dealloc`, `.relinquish_alloc_permit` | `::1` | A single warp |
| `.alloc`, … | `::2` | Two warps, one per CTA, collectively |
| `.ld`, `.st`, `.wait` | n/a | Per warp, quarter of TMEM each |

And from §9.7.17.6.1, `.mma`, `.cp`, `.shift`, `.ld` and `.st` are
**asynchronous**; `.alloc`, `.dealloc`, `.fence`, `.wait` and `.commit`
are synchronous.

**This confirms the hypothesis the whole `--mix-*` design rests on.**
A single thread issues an async MMA that then runs on the tensor core
without consuming further issue slots. The issue-bandwidth objection in
`DESIGN.md` is therefore much weaker on Blackwell than on earlier
architectures: one thread can keep the tensor core saturated while
other warps in the same CTA hammer the FMA pipe and stream from HBM.
Mixing units is not fighting for issue bandwidth the way it would on
Hopper — **the mixed-unit power virus is a coherent design**, and
warp specialization is the natural way to build it.

### MMA target requirements **[V]**

`tcgen05.mma` is **not in the baseline feature set**. Per its Target ISA
Notes it is supported on:

- `sm_100a`, `sm_110a` (formerly `sm_101a`), and by the family rule
  `sm_103a`;
- `sm_100f` or higher in the same family, from **PTX ISA 8.8** — but
  **excluding certain kinds**: `.kind::i8` on most shapes, and
  `.kind::mxf4` / `.kind::mxf4nvf4` on others.

Two direct consequences:

1. A build targeting plain `sm_100`/`sm_103` compiles cleanly and
   contains **no tcgen05 at all**. This is exactly the silent failure
   the T0 gate exists to catch.
2. **FP4 (`mxf4`/`mxf4nvf4`) needs the arch-specific target.** If the
   virus wants the lowest-precision formats — and §5 argues they are
   worth measuring — then `sm_103a` is required for B300, not just
   `sm_100f`. Both are in our gencode list.

### Still to verify

Exact MMA shapes and the descriptor formats (§9.7.17.4) we will need to
hand-build; TMEM allocation's practical effect on occupancy; whether to
hand-write PTX or lean on CUTLASS's Blackwell kernels for the inner
loop.

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

## 8. Open items

**Resolved 2026-08-17** (PTX ISA v9.x, nvcc guide, Programming Guide
v13.3, all via docs.nvidia.com):

- ~~`tcgen05` semantics~~ → **[V]** async, single-thread issue,
  CTA-pair optional. The mixed-unit virus is a coherent design (§3).
- ~~arch-specific targets~~ → **[V]** three classes, and `sm_100f`
  spans B200 and B300. Applied to CMake (§1).
- ~~TMEM capacity and allocation rules~~ → **[V]** 512×128×32-bit per
  CTA, 32-column power-of-two allocation by a single warp (§3).

Still open, ordered by how much they would change the design:

1. Exact `tcgen05.mma` shapes and descriptor construction — and whether
   to hand-write PTX or start from CUTLASS's Blackwell kernels (§3).
2. FP4/FP6 reachability from CUDA C++ versus requiring PTX/CUTLASS,
   now knowing `mxf4` needs an `a` target (§5).
3. Whether datacenter Blackwell FP64 is as weak as assumed (§2). The
   Programming Guide's arithmetic-throughput table moved in the v13.3
   restructure; find it or measure directly at T2.
4. Actual DVFS control-loop behaviour near the cap on Blackwell (§6).
5. B300 board TDP — our scanner records 1200 W for
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
