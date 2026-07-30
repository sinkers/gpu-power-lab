"""Tiny sanity test — plan expansion only, no GPU required."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan import load_plan

def test_example_expands():
    plan = load_plan(Path(__file__).parent / "plans" / "example.yaml")
    assert plan.campaign == "example-smoke"
    assert plan.device == 0
    # 4 precisions * 2 sizes * 1 stream = 8, plus 4 stream counts = 12
    assert len(plan.rungs) == 12, f"got {len(plan.rungs)}"
    ids = [r.rung_id for r in plan.rungs]
    assert len(set(ids)) == len(ids), "duplicate rung ids"
    # First rung should be fp32/4096/streams=1
    r0 = plan.rungs[0]
    assert r0.op == "sgemm"
    assert r0.precision == "fp32"
    assert r0.size == 4096
    assert r0.streams == 1
    print("ok, first rung:", json.dumps(r0.__dict__))
    print("total rungs:", len(plan.rungs))

if __name__ == "__main__":
    test_example_expands()
    print("PASS")
