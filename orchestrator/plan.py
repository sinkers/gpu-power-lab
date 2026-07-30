"""
plan.py — expand a YAML plan file into a list of concrete rung specs.

Plan file schema (informal):

    campaign: h100-power-sweep
    device: 0
    defaults:
      warmup_sec: 5
      steady_sec: 30
      sample_hz: 100
      stop_on_throttle: false
    rungs:
      - op: sgemm
        precision: [fp32, tf32, fp16, bf16]
        size: [2048, 4096, 8192, 16384]
        streams: [1, 2, 4]
      - op: sgemm
        precision: fp16
        size: 8192
        streams: [1, 2, 4, 8, 16]
        steady_sec: 60

Any field that is a list at rung-scope is expanded via cartesian product.
Scalar fields are used as-is. Rung-scope values override `defaults`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RungSpec:
    op: str
    precision: str
    size: int
    streams: int
    warmup_sec: float
    steady_sec: float
    sample_hz: int
    device: int
    stop_on_throttle: bool
    rung_id: str

    def to_cli_args(self, out_metrics: Path, out_summary: Path) -> list[str]:
        args = [
            "--op",           self.op,
            "--precision",    self.precision,
            "--size",         str(self.size),
            "--streams",      str(self.streams),
            "--warmup-sec",   str(self.warmup_sec),
            "--steady-sec",   str(self.steady_sec),
            "--sample-hz",    str(self.sample_hz),
            "--device",       str(self.device),
            "--rung-id",      self.rung_id,
            "--out-metrics",  str(out_metrics),
            "--out-summary",  str(out_summary),
        ]
        if self.stop_on_throttle:
            args.append("--stop-on-throttle")
        return args


@dataclass
class Plan:
    campaign: str
    device: int
    rungs: list[RungSpec] = field(default_factory=list)


_EXPANDABLE_KEYS = ("op", "precision", "size", "streams",
                    "warmup_sec", "steady_sec", "sample_hz",
                    "stop_on_throttle")


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _expand_rung_block(block: dict, defaults: dict, device: int) -> list[dict]:
    """Return a list of concrete rung dicts from a single rung-block."""
    merged = {**defaults, **block}
    axes = {k: _as_list(merged[k]) for k in _EXPANDABLE_KEYS if k in merged}

    # Any keys with a single value collapse trivially through product.
    keys = list(axes.keys())
    values = [axes[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        d = dict(zip(keys, combo))
        d.setdefault("device", device)
        out.append(d)
    return out


def _rung_id(campaign: str, i: int, r: dict) -> str:
    return (f"{campaign}-{i:04d}-"
            f"{r['op']}-{r['precision']}-{r['size']}-s{r['streams']}")


def load_plan(path: str | Path) -> Plan:
    p = Path(path)
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"plan file is not a mapping: {p}")

    campaign = str(doc.get("campaign") or p.stem)
    device = int(doc.get("device", 0))
    defaults = dict(doc.get("defaults") or {})
    defaults.setdefault("warmup_sec", 5.0)
    defaults.setdefault("steady_sec", 30.0)
    defaults.setdefault("sample_hz", 100)
    defaults.setdefault("stop_on_throttle", False)

    raw_rungs = doc.get("rungs") or []
    if not isinstance(raw_rungs, list):
        raise ValueError("'rungs' must be a list")

    concrete: list[dict] = []
    for block in raw_rungs:
        if not isinstance(block, dict):
            raise ValueError(f"rung block is not a mapping: {block!r}")
        concrete.extend(_expand_rung_block(block, defaults, device))

    rungs: list[RungSpec] = []
    for i, r in enumerate(concrete):
        rungs.append(RungSpec(
            op=str(r["op"]),
            precision=str(r["precision"]),
            size=int(r["size"]),
            streams=int(r["streams"]),
            warmup_sec=float(r["warmup_sec"]),
            steady_sec=float(r["steady_sec"]),
            sample_hz=int(r["sample_hz"]),
            device=int(r["device"]),
            stop_on_throttle=bool(r["stop_on_throttle"]),
            rung_id=_rung_id(campaign, i, r),
        ))

    return Plan(campaign=campaign, device=device, rungs=rungs)


if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser(description="Expand a plan and print the rungs.")
    ap.add_argument("plan")
    ns = ap.parse_args()
    plan = load_plan(ns.plan)
    print(f"campaign: {plan.campaign}")
    print(f"device:   {plan.device}")
    print(f"rungs:    {len(plan.rungs)}")
    for r in plan.rungs:
        print(json.dumps(r.__dict__))
