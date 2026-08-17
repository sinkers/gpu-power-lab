# gpu-power-lab

Low-level GPU power characterization harness.

Two questions drive the project:

- **O1 — can we hit 100%?** Can synthetic code drive a GPU to sustained
  draw at 100% of its enforced power limit — genuinely power-limited,
  not thermally or compute limited?
- **O2 — how spiky can we make it?** Can we then modulate that load into
  extreme transients — large, fast, repeatable idle↔max swings, with
  instantaneous excursions that overshoot the rated limit before the
  GPU's own power controller clamps them?

To answer them it runs a controlled ramp of compute workloads on a single
GPU while sampling NVML (and optionally DCGM) at high rate, capturing
power draw, thermal behaviour, throttling, and delivered TFLOPS across a
matrix of operations, precisions, sizes, and concurrency levels.

The working hypothesis is that **O1's answer is no** — that no workload
we can write holds a datacenter GPU at its full rated power. If so, the
useful result is the breakdown: how much power each class of work
contributes (FP32 vs each tensor precision vs memory traffic vs
fabric), where each one runs out of headroom, and what the residual gap
is made of.

Per-rung outputs include the standard efficiency trio — **EDP** (E×D),
**EDPp** (E×D²) and **%TDP** — so results are comparable to published
work and can locate operating points, not just ceilings. The headline
efficiency output is EDP and EDPp swept against the enforced power
limit: the cap that minimizes EDP is the energy-performance sweet spot,
and capping below it is where demand response starts genuinely trading
performance for grid service.

- `DESIGN.md` — success criteria, the O1 power-attribution ladder, the
  `powervirus` workload design, the efficiency metrics, and the
  instrumentation constraints that decide whether O2 is answerable at
  all.
- `docs/blackwell-cuda-notes.md` — Blackwell from a CUDA programming
  perspective: SM units and where the watts live, `tcgen05` and tensor
  memory, asynchrony, precision formats, DVFS behaviour near the cap.
  Living document; claims are marked verified or unverified.
- `TESTPLAN.md` — how the code gets written and validated on cheap
  hardware before it runs on a bare-metal 8×B300 node.
- `docs/real-workload-plan.md` — what comes next: a training waterfall and
  a maximum-inference sweep on one B300, then the 8×B300 node where the
  fabric contributes and eight GPUs can be made to swing in phase.

## Results

- `results/t2-b300/` — **B300 SXM6**: 90.4% of 1100 W sustained, an 881 W
  swing in 200 ms, and the 528 W gap between two ways of driving the tensor
  cores. Report: `report.html` (`python3 scripts/generate_report.py
  results/t2-b300`).
- `results/t1-a10g/` — A10G: first end-to-end run, and the Ampere result
  that B300 later inverted.

## Architecture

Two layers, one contract.

- **`runner/`** — a static C binary. One invocation runs **one rung**
  (one workload configuration) end-to-end: warmup → steady-state
  measurement → summary. Emits NDJSON telemetry + a summary JSON.
  Has no cloud dependencies. Deterministic timing, pinned sampler
  thread, direct NVML/cuBLAS calls.

- **`orchestrator/`** — Python. Expands YAML test plans into rung
  invocations, calls the C binary as a subprocess per rung, waits for
  thermal cooldown between rungs, then collates results and uploads
  to S3 / Timestream.

The **contract between them** is defined by
`schema/rung-summary.schema.json`. Both sides validate against it.

## Repo layout

```
gpu-power-lab/
├── DESIGN.md                    # The full design doc
├── TESTPLAN.md                  # Dev + test workflow, tier by tier
├── docs/
│   └── blackwell-cuda-notes.md  # Blackwell for CUDA programmers
├── schema/
│   └── rung-summary.schema.json # C ↔ Python contract
├── runner/                      # C binary
│   ├── src/
│   ├── CMakeLists.txt
│   └── README.md
├── orchestrator/                # Python
│   ├── plan.py
│   ├── campaign.py
│   ├── cooldown.py
│   ├── plans/
│   │   └── example.yaml
│   └── requirements.txt
├── terraform/                   # AWS bootstrap (TBD)
└── analysis/                    # Notebooks (TBD)
```

## Build the runner

Requires CUDA Toolkit ≥ 12.0, CMake ≥ 3.20, a C11 compiler.

```
cd runner
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gpu-power-runner --help
```

## Run one rung by hand

```
./build/gpu-power-runner \
  --op sgemm \
  --precision fp16 \
  --size 8192 \
  --streams 4 \
  --warmup-sec 5 \
  --steady-sec 30 \
  --sample-hz 100 \
  --device 0 \
  --out-metrics /tmp/rung.ndjson \
  --out-summary /tmp/rung.json
```

## Run a campaign

```
cd orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python campaign.py --plan plans/example.yaml --out-dir /tmp/campaign-01
```

To upload results automatically when the campaign finishes:

```
python campaign.py \
  --plan plans/example.yaml \
  --out-dir /tmp/campaign-01 \
  --upload-bucket my-results-bucket \
  --upload-prefix campaigns \
  --timestream-db gpu-lab \
  --timestream-metrics-table metrics \
  --timestream-summaries-table summaries
```

Upload failures are non-fatal — campaign data is always preserved locally.

## Testing

The Python test suite lives in `orchestrator/tests/` and covers plan expansion,
S3 upload (via moto), schema validation, and Timestream record building.
No GPU or AWS credentials are needed.

```bash
cd orchestrator
pip install -r requirements.txt   # adds pytest + moto if not already present
pytest -q
```

Run with verbose output:

```bash
pytest -v tests/
```

To validate a specific summary file against the schema:

```bash
python schema_validate.py /tmp/out/my-campaign/rung-0000/summary.json
```

## Status

**Scaffold + first working slice; scope revised 2026-08-17.** GEMM,
memstream and FFT workloads, NVML sampling, local NDJSON + JSON output.

Not yet built, and needed for the two objectives above:

- `powervirus` mixed-unit kernel and the power-limit-raising setup (O1)
- `--duty` modulation and the high-bandwidth power sampling paths (O2)

See `DESIGN.md` for the full plan and what's next.
