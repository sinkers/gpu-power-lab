# runner

C11 binary. Runs one rung.

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requirements:
- CUDA Toolkit ≥ 12.0 (provides `cudart`, `cublas`, and the NVML stub)
- CMake ≥ 3.20
- C11 compiler

## Run

```
./build/gpu-power-runner --help
```

Minimal invocation:

```
./build/gpu-power-runner \
  --op sgemm --precision fp16 --size 8192 --streams 4 \
  --warmup-sec 5 --steady-sec 30 --sample-hz 100 \
  --device 0 \
  --out-metrics /tmp/rung.ndjson --out-summary /tmp/rung.json
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | `--help` |
| 2 | bad arguments |
| 3 | NVML initialization failure |
| 4 | CUDA runtime init failure |
| 5 | workload error (CUDA / cuBLAS) |
| 6 | throttled and `--stop-on-throttle` was set |

## Output

Two files:

- `--out-metrics` — one NDJSON record per NVML sample, tagged with phase.
  Suitable for streaming into Timestream / InfluxDB / Grafana.
- `--out-summary` — a single JSON object matching
  `../schema/rung-summary.schema.json`.

## Notes

- FP8 is not implemented in v0.1 (needs `cublasLtMatmul`).
- FFT and memstream workloads are not implemented in v0.1.
- The sampler thread attempts `SCHED_RR` priority — this silently falls
  back to `SCHED_OTHER` if the process lacks the capability.
- `cudaMemset(0x3c)` pattern in the input matrices is intentionally a
  half-precision ~1.0 bit pattern. Values are never checked; this is a
  power/thermal harness, not a correctness harness.
