# vLLM configuration for the R2 inference measurement

Device: NVIDIA B300 SXM6 AC, 275,040 MiB HBM, 1100 W enforced limit.
Model: Qwen2.5-72B-Instruct, bf16, 136 GB of weights, downloaded locally.
Stack: vLLM 0.27.1 on CPython 3.12.14 in a dedicated venv.

## Why Python 3.12 rather than the system interpreter

vLLM 0.27 depends on flashinfer, which uses `array.array[int]` subscript
syntax. Ubuntu 22.04's Python 3.10 cannot parse it and the engine dies during
initialisation with `TypeError: 'type' object is not subscriptable`. The
traceback points at `vllm/compilation`, not at the interpreter version, so it
is easy to misdiagnose as a vLLM or CUDA problem.

`r2-setup.sh` installs a standalone CPython 3.12 via `uv` rather than adding a
PPA, so the setup reproduces on a fresh box with no extra apt sources.

## Measured KV capacity

KV bytes per token, computed from the model config rather than taken from
vLLM's own report: `2 (K and V) × 80 layers × 8 KV heads × 128 head dim ×
dtype bytes`.

| KV dtype | Per token | Predicted per 100 GB |
|---|---|---|
| bf16 | 320 KiB | 0.31 M tokens |
| fp8 | 160 KiB | 0.61 M tokens |

Measured, at `--gpu-memory-utilization 0.95`:

| Configuration | GPU memory used | KV cache | Full-length seqs |
|---|---|---|---|
| 32k ctx, bf16 KV | 261,380 MiB (95.0%) | 370,640 tok | 11 @ 32k |
| 32k ctx, fp8 KV | 263,598 MiB (95.8%) | 755,264 tok | 23 @ 32k |
| 16k ctx, fp8 KV, chunked prefill | 261,406 MiB (95.0%) | 758,464 tok | 46 @ 16k |

fp8 KV doubles capacity, matching the arithmetic. The card is genuinely full
in all three: ~95% allocated, weights plus KV, not allocated-but-idle.

## Context ceiling

32,768 tokens. `max_position_embeddings` is 32768 and `rope_scaling` is null,
so anything longer requires `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and, per vLLM's
own warning, produces NaNs on a RoPE model. Not used.

## Chosen configuration

```
--gpu-memory-utilization 0.95
--max-model-len 32768
--kv-cache-dtype fp8
--max-num-seqs 256
--enable-chunked-prefill
--max-num-batched-tokens 8192
```

### Why fp8 KV, given the goal is to load the memory system

There is an apparent tension: bf16 KV moves twice the bytes per token, which
should mean more memory traffic. It does not follow, because of where decode
traffic actually goes.

In decode, generating one token for one sequence requires reading the entire
weight set — 136 GB — plus that sequence's KV. Weight traffic dominates by
orders of magnitude. The way to raise decode power is therefore not to make
each KV read larger, but to **amortise the weight read across more concurrent
sequences**: with a batch of N, one pass over the weights produces N tokens,
which raises arithmetic intensity and moves the phase away from being purely
bandwidth-starved.

fp8 KV doubles how many sequences fit, so it raises achievable concurrency,
which is the lever that matters. It is also what a production deployment of
this size would run.

Both configurations are worth measuring if time allows; the bf16-KV variant is
the control that tests this reasoning rather than assuming it.

## Phases

Prefill and decode are measured separately because they load the device in
opposite ways and averaging them hides the result.

| Phase | Input | Output | Concurrency | Expected |
|---|---|---|---|---|
| prefill | 8192 | 8 | 32 | compute-bound; should approach the training figures |
| decode | 128 | 2048 | 96 | bandwidth-bound; the phase the 72B model is required for |
| balanced | 1024 | 256 | 64 | reference serving mix |

`ignore_eos` is set on every request so the decode phase generates the full
configured length instead of stopping early and silently becoming shorter than
specified.

An idle-with-model-resident reading is taken before the phases. That is the
correct floor for a serving deployment and is distinct from the 183.5 W
empty-device idle measured earlier.

## Not yet established

Whether concurrency is actually the binding constraint on decode power, or
whether the device stays bandwidth-bound regardless. The 96-sequence decode
phase against a 256-sequence limit will show whether the scheduler is
saturating; if it is not, the phase should be re-run with more load before any
decode figure is treated as a ceiling.
