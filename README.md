# gpu-power-lab

Low-level GPU power characterization harness.

Runs a controlled ramp of compute workloads on a single GPU while sampling
NVML (and optionally DCGM) at high rate to characterize power draw,
thermal behaviour, throttling, and delivered TFLOPS across a matrix of
operations, precisions, sizes, and concurrency levels.

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

**Scaffold + first working slice.** GEMM workload only, NVML sampling,
local NDJSON + JSON output. See `DESIGN.md` for the full plan and
what's next.
