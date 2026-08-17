# T1 first run — A10G (sm_86), 2026-08-17

First end-to-end run of the harness on real hardware. AWS `g5.xlarge`
spot, us-east-1, ~$0.65/hr, driver 595.91.07, CUDA 13.2, root.

**A10G is not Blackwell.** This validates the harness, not the target.
The `sm_103a` path compiles but has never executed. Everything below is
an Ampere result and the numbers do not transfer.

## Probe

All capabilities present: power read, `POWER_INSTANT`, `POWER_AVERAGE`,
samples buffer (120 buffered), energy counter, violation status,
power-limit write, persistence, clock lock, `SCHED_RR`, no MIG.
Verdict: O1 and O2 both measurable. Limits: default 300 W, max 300 W,
min 100 W — so `--raise-power-limit` is a no-op on this part.

## O1 — power attribution ladder

15s steady, 300 W enforced limit, all rungs at 1710 MHz, nothing
throttled.

| Rung | Avg W | %TDP |
|---|---|---|
| tensor-only (wmma fp16) | 106.2 | 35.4 |
| **fma-only (FP32 FFMA)** | **221.5** | **73.8** |
| dram-only | 141.1 | 47.0 |
| tensor + fma | 165.5 | 55.2 |
| tensor + dram | 139.9 | 46.6 |
| mix 1:1:1 | 146.2 | 48.7 |
| mix 2:1:1 | 150.7 | 50.2 |
| fma:tensor 7:1 | 207.2 | 69.1 |
| fma:tensor 15:1 | 213.5 | 71.2 |
| fma:dram 7:1 | 180.5 | 60.2 |
| cuBLAS sgemm fp16 8192, 2 streams | 172.7 | 57.6 (68.6 TFLOPS) |

**The mixing hypothesis is contradicted on this part.** Every mix draws
less than pure FMA, and the more warps are moved away from FMA the lower
the power goes — the ordering is monotonic. The reason is the one
`DESIGN.md` flagged as the risk: on Ampere all three roles consume warp
slots on the same SM, so mixing is not adding concurrent activity, it is
*reallocating* a fixed warp budget to units with lower marginal watts.
FMA has the best watts-per-warp by a wide margin, so pure FMA wins.

This does not settle the question for Blackwell, and the reason is
specific: `tcgen05.mma` is asynchronous and single-thread-issued
(`docs/blackwell-cuda-notes.md` §3), so tensor work there does *not*
consume a warp slot the way `wmma` does here. The A10G result is
evidence that the mix only helps when the units are genuinely
independent of the warp budget — which is exactly the property
Blackwell has and Ampere does not.

Worth noting against the project's premise: the best synthetic result,
73.8% of TDP, is only modestly above the ~71% `nvidia-load-shave`
measured from real vLLM inference on the same GPU class. **Nothing came
close to 100%**, and no rung ever set a throttle bit — the part was
never power-limited. Consistent with the working hypothesis.

## O2 — transients

`--duty-on-ms 200 --duty-off-ms 200`, fma-only, 100 Hz, 1201 steady
samples.

| Path | Min W | Max W | Swing |
|---|---|---|---|
| `nvmlDeviceGetPowerUsage` | 127.4 | 223.8 | 96.4 |
| `NVML_FI_DEV_POWER_INSTANT` | 65.0 | 224.1 | 159.1 |

**The two paths agree at the top and disagree at the bottom, by 62 W.**
Both see the peak, but the averaged counter never follows the idle gap
down — it reports a 127 W floor for a GPU that the instantaneous
counter says fell to 65 W. At a 200 ms period, which is slow, the
averaged reading already understates the swing by 39%.

This is the design's instrumentation argument confirmed on hardware at
the first attempt: `GetPowerUsage` alone cannot substantiate a claim
about a transient, and any O2 number sourced from it understates the
swing. Both paths are now logged per sample with a `source` tag.

No power-cap violations recorded (`violation_us` = 0), as expected on a
part that never reached its limit.

## Harness bugs this run caught

1. **Energy integral used the nominal sample rate.** Reported a 52%
   gap between the energy counter and the integrated power curve —
   which reads exactly like the "unresolved spike" signal the design
   wants that check to detect. It was not: the sampler was achieving
   ~48 Hz against a requested 100 Hz, and `1/sample_hz` was the wrong
   dt. Now integrates against measured dt; gap fell to ±0.5%. **The
   cross-check works, and it caught our own instrumentation first.**
2. **The sampler does not always hit `--sample-hz`.** With DRAM warps
   resident it dropped to 34.6 Hz — NVML calls contend with memory
   traffic. Summaries now carry `sample_hz_achieved` alongside
   `sample_hz_requested`; transient claims must be judged against the
   achieved rate.
3. Sync-per-launch cost ~20% of achievable power (48% → 74% of TDP
   once launches were batched). The GPU was idling on host round-trips.

## Reproduce

```
sudo ./build/gpu-power-runner --probe
sudo ./build/gpu-power-runner --op powervirus --mix-tensor 0 --mix-fma 1 --mix-dram 0 \
  --warmup-sec 4 --steady-sec 15 --sample-hz 100 --raise-power-limit \
  --out-summary /tmp/r.json --out-metrics /tmp/r.ndjson
```
