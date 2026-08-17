#!/usr/bin/env python3
"""
R1 — what a training step actually draws, layer by layer.

Runs one phase of a transformer training step in a loop for a fixed duration,
so the runner's `--op observe` can sample power alongside it. Each phase adds
one layer of overhead to the one before, and the difference between
consecutive phases is what that layer costs in watts:

    gemm      just the GEMMs of one layer, at the real shapes
    forward   full forward pass, no autograd
    fwdbwd    add the backward pass
    optim     add the AdamW step
    full      add a dataloader that actually moves data from host

Deliberately a synthetic transformer with random data rather than a
downloaded checkpoint. A real model would add a 16 GB gated download and a
licence click to a measurement that does not depend on the weights at all —
the arithmetic and the memory traffic are set by the shapes, not the values.
The shapes here are Llama-3-8B's: 32 layers, hidden 4096, FFN 14336, 32
heads.

    python3 r1_training.py --phase fwdbwd --seconds 45 --batch 4 --seqlen 4096
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block, Llama-shaped: RMSNorm, GQA-less MHA, SwiGLU."""

    def __init__(self, d_model=4096, n_heads=32, d_ff=14336):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.n1 = nn.RMSNorm(d_model)
        self.n2 = nn.RMSNorm(d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, S, D)
        x = x + self.o(a)
        h = self.n2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


def build(layers, d_model, n_heads, d_ff, dtype, dev):
    return nn.Sequential(*[Block(d_model, n_heads, d_ff) for _ in range(layers)]).to(dev, dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["gemm", "forward", "fwdbwd", "optim", "full"])
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--warmup", type=float, default=10.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=14336)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out", default=None)
    # The power sampler is a separate process. It must not start until warmup
    # is done, or the measured window includes allocator churn, autotuning and
    # a cold cache - which is exactly the transient the soak result showed
    # matters. The workload signals readiness by touching this file.
    p.add_argument("--ready-file", default=None)
    a = p.parse_args()

    dev = "cuda"
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True

    # cuDNN's fused attention has no execution plan for sm_103 in this build
    # ("No valid execution plans built"), so steer SDPA to the flash /
    # mem-efficient kernels. Worth noting rather than hiding: on brand-new
    # silicon the vendor stack is not uniformly ready, and a benchmark that
    # silently falls back would be measuring a different kernel than intended.
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception as e:
        print(f"# sdpa backend selection: {e}", file=sys.stderr)

    B, S, D = a.batch, a.seqlen, a.d_model

    if a.phase == "gemm":
        # The GEMMs of one block, at the real shapes, with nothing around
        # them. This is the bridge between the synthetic ladder and the model:
        # if it lands near the powervirus ceiling, the shapes are not the
        # reason a training step draws less.
        x = torch.randn(B * S, D, device=dev, dtype=dt)
        wq = torch.randn(D, D, device=dev, dtype=dt)
        wg = torch.randn(D, a.d_ff, device=dev, dtype=dt)
        wd = torch.randn(a.d_ff, D, device=dev, dtype=dt)
        def step():
            h = x @ wq
            g = h @ wg
            _ = g @ wd
    else:
        model = build(a.layers, D, a.n_heads, a.d_ff, dt, dev)
        if a.compile:
            model = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True) \
            if a.phase in ("optim", "full") else None
        # Resident batch for every phase except `full`, which pays the host
        # transfer that a real dataloader does.
        resident = torch.randn(B, S, D, device=dev, dtype=dt)
        host_batch = torch.randn(B, S, D, dtype=dt).pin_memory()

        def step():
            if a.phase == "full":
                x = host_batch.to(dev, non_blocking=True)
            else:
                x = resident
            if a.phase == "forward":
                with torch.no_grad():
                    model(x)
                return
            out = model(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)

    # --- warmup ---
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.warmup:
        step()
    torch.cuda.synchronize()

    if a.ready_file:
        with open(a.ready_file, "w") as f:
            f.write(str(time.time()))

    # --- measured ---
    iters = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds:
        step()
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    # Model FLOPs, the standard 6*N*T for a training step (2 for forward,
    # 4 more for backward). Reported so power can be read per unit of real
    # work rather than per wall-second.
    n_params = a.layers * (4 * D * D + 3 * D * a.d_ff)
    tokens = iters * B * S
    mult = {"gemm": 0, "forward": 2, "fwdbwd": 6, "optim": 6, "full": 6}[a.phase]
    tflops = (mult * n_params * tokens / elapsed / 1e12) if mult else 0.0

    res = {
        "phase": a.phase, "iters": iters, "seconds": round(elapsed, 3),
        "tokens": tokens, "tokens_per_s": round(tokens / elapsed, 1),
        "model_tflops": round(tflops, 1),
        "params_b": round(n_params / 1e9, 2),
        "batch": B, "seqlen": S, "dtype": a.dtype,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
