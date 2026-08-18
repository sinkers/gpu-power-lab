#!/usr/bin/env python3
"""
R2 — power draw of an inference server, by phase.

Drives a running vLLM OpenAI-compatible endpoint at fixed concurrency and
reports throughput, while `gpu-power-runner --op observe` samples power
alongside. Signals readiness through a file so the sampler measures the
steady window rather than ramp-up.

The point of the phase split: prefill and decode load the GPU in completely
different ways, and averaging them hides the result.

    prefill-heavy   long input, short output. Compute-bound; every prompt
                    token is processed in parallel, so this behaves like a
                    large GEMM and should draw near the training figures.
    decode-heavy    short input, long output. Memory-bound; one token at a
                    time, and every token requires reading the full weight
                    set from HBM. Throughput is bounded by bandwidth, not
                    arithmetic.
    balanced        a plausible serving mix, for reference.

Decode is the phase where a large model matters. Weight traffic per token
scales with parameter count, so a small model understates decode power on a
275 GB card no matter how much KV cache is allocated.

    python3 r2_inference.py --phase decode --concurrency 64 --seconds 60
"""

import argparse
import asyncio
import json
import random
import time

import aiohttp

# Deterministic filler so prompt length is controlled rather than incidental.
WORDS = ("power draw measurement instrument telemetry sample window enforced "
         "limit sustained transient throttle attribution precision cache "
         "bandwidth kernel tensor pipeline scheduler").split()


def make_prompt(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    # ~0.75 words per token is a reasonable approximation for this filler.
    n = max(8, int(tokens * 0.75))
    return " ".join(rng.choice(WORDS) for _ in range(n))


async def one_request(session, url, model, prompt, max_tokens, stats):
    t0 = time.monotonic()
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        # Force the full output length: without this the model may stop early
        # and the decode phase silently becomes shorter than configured.
        "ignore_eos": True,
    }
    try:
        async with session.post(url, json=body) as r:
            d = await r.json()
            if "usage" in d:
                stats["prompt_tokens"] += d["usage"].get("prompt_tokens", 0)
                stats["completion_tokens"] += d["usage"].get("completion_tokens", 0)
            stats["requests"] += 1
            stats["latency"].append(time.monotonic() - t0)
    except Exception as e:
        stats["errors"] += 1
        stats["last_error"] = str(e)[:200]


async def worker(session, url, model, args, stats, stop_at, seed):
    i = 0
    while time.monotonic() < stop_at:
        prompt = make_prompt(args.input_tokens, seed * 1000 + i)
        await one_request(session, url, model, prompt, args.output_tokens, stats)
        i += 1


async def run(args):
    base = args.endpoint.rstrip("/")
    url = f"{base}/v1/completions"

    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)) as session:
        # Resolve the served model name rather than requiring it as an argument.
        async with session.get(f"{base}/v1/models") as r:
            model = (await r.json())["data"][0]["id"]

        stats = {"requests": 0, "errors": 0, "prompt_tokens": 0,
                 "completion_tokens": 0, "latency": [], "last_error": ""}

        # Warmup: fill the scheduler and let the KV cache reach a realistic
        # occupancy before anything is measured.
        warm_stop = time.monotonic() + args.warmup
        await asyncio.gather(*[
            worker(session, url, model, args, dict(stats, latency=[]), warm_stop, s)
            for s in range(args.concurrency)])

        if args.ready_file:
            with open(args.ready_file, "w") as f:
                f.write(str(time.time()))

        t0 = time.monotonic()
        stop_at = t0 + args.seconds
        await asyncio.gather(*[
            worker(session, url, model, args, stats, stop_at, 10_000 + s)
            for s in range(args.concurrency)])
        elapsed = time.monotonic() - t0

        lat = sorted(stats["latency"])
        res = {
            "phase": args.phase, "model": model,
            "concurrency": args.concurrency,
            "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
            "seconds": round(elapsed, 2),
            "requests": stats["requests"], "errors": stats["errors"],
            "req_per_s": round(stats["requests"] / elapsed, 2),
            "prompt_tok_per_s": round(stats["prompt_tokens"] / elapsed, 1),
            "output_tok_per_s": round(stats["completion_tokens"] / elapsed, 1),
            "total_tok_per_s": round(
                (stats["prompt_tokens"] + stats["completion_tokens"]) / elapsed, 1),
            "latency_p50_s": round(lat[len(lat) // 2], 3) if lat else None,
            "latency_p95_s": round(lat[int(len(lat) * 0.95)], 3) if lat else None,
        }
        if stats["errors"]:
            res["last_error"] = stats["last_error"]
        print(json.dumps(res))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="balanced",
                   choices=["prefill", "decode", "balanced"])
    p.add_argument("--endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--input-tokens", type=int, default=None)
    p.add_argument("--output-tokens", type=int, default=None)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--warmup", type=float, default=20.0)
    p.add_argument("--ready-file", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    # Phase presets, if not overridden.
    if a.input_tokens is None or a.output_tokens is None:
        preset = {"prefill":  (8192, 8),
                  "decode":   (128, 2048),
                  "balanced": (1024, 256)}[a.phase]
        a.input_tokens = a.input_tokens or preset[0]
        a.output_tokens = a.output_tokens or preset[1]

    asyncio.run(run(a))


if __name__ == "__main__":
    main()
