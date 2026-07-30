"""
test_schema.py — validate rung summary JSON fixtures against
``schema/rung-summary.schema.json`` using the :mod:`schema_validate` helper.

These tests run without any GPU or AWS access.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import jsonschema
import pytest

from schema_validate import validate_summary

# Shared fixture path (created by test_upload.py setup, but defined independently).
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_SUMMARY = FIXTURES_DIR / "sample_summary.json"


# ---------------------------------------------------------------------------
# Inline representative fixture — mirrors exactly what summary.c emits.
# Fields are listed in the same order as the C fprintf calls so any
# structural divergence between C and schema is immediately obvious.
# ---------------------------------------------------------------------------

_VALID_SUMMARY: dict = {
    "schema_version": 1,
    "rung_id": "h100-power-sweep-0003-sgemm-fp16-8192-s4",
    "invocation": {
        "op":         "sgemm",
        "precision":  "fp16",
        "size":       8192,
        "streams":    4,
        "warmup_sec": 5.000,
        "steady_sec": 30.000,
        "sample_hz":  100,
        "device":     0,
    },
    "device": {
        "uuid":          "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "name":          "NVIDIA H100 80GB HBM3",
        "driver":        "535.104.12",
        "cuda_runtime":  "12.2",
        "vbios":         "96.00.74.00.01",
        "pci_bus_id":    "0000:00:1E.0",
        "power_limit_w": 700.00,
    },
    "timing": {
        "start_utc": "2026-07-28T10:00:00.000Z",
        "warmup_ms": 5001.23,
        "steady_ms": 30002.45,
        "wall_ms":   35100.00,
    },
    "compute": {
        "iterations":         312500,
        "tflops_measured":    989.1234,
        "tflops_theoretical": None,
        "efficiency":         None,
    },
    "power": {
        "avg_w":    695.320,
        "peak_w":   700.000,
        "p50_w":    695.100,
        "p95_w":    699.200,
        "p99_w":    699.850,
        "energy_j": 20859.600,
    },
    "thermal": {
        "avg_c":       72.50,
        "peak_c":      78.00,
        "throttled_sec": 0.000,
    },
    "throttle": {
        "mask":          0,
        "any_throttled": False,
        "reasons":       [],
    },
    "utilization": {
        "sm_avg_pct":  99.10,
        "mem_avg_pct": 88.40,
    },
    "clocks": {
        "sm_avg_mhz":       1980.0,
        "sm_min_mhz":       1980.0,
        "sm_max_mhz":       1980.0,
        "mem_avg_mhz":      2619.0,
        "sm_boost_dropped": False,
    },
    "samples_written": 3000,
    "result": "ok",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mutated(**overrides) -> dict:
    """Return a deep copy of _VALID_SUMMARY with *overrides* applied at top level."""
    doc = copy.deepcopy(_VALID_SUMMARY)
    doc.update(overrides)
    return doc


def _nested_set(d: dict, keys: list[str], value) -> dict:
    """Return a deep copy of *d* with the nested key path set to *value*."""
    doc = copy.deepcopy(d)
    target = doc
    for k in keys[:-1]:
        target = target[k]
    target[keys[-1]] = value
    return doc


# ---------------------------------------------------------------------------
# Positive tests — valid documents
# ---------------------------------------------------------------------------


def test_inline_fixture_is_valid(tmp_path: Path):
    """The inline representative summary validates cleanly."""
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(_VALID_SUMMARY), encoding="utf-8")
    validate_summary(p)  # must not raise


def test_file_fixture_is_valid():
    """The on-disk sample_summary.json fixture validates cleanly."""
    validate_summary(SAMPLE_SUMMARY)  # must not raise


def test_with_error_message_is_valid(tmp_path: Path):
    """Adding the optional 'error_message' field does not break validation."""
    doc = _mutated(result="cuda_error", error_message="cublas init failed")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    validate_summary(p)


def test_throttle_hit_result_is_valid(tmp_path: Path):
    doc = _nested_set(_VALID_SUMMARY, ["result"], "throttle_hit")
    doc = _nested_set(doc, ["throttle", "any_throttled"], True)
    doc = _nested_set(doc, ["throttle", "mask"], 4)
    doc = _nested_set(doc, ["throttle", "reasons"], ["sw_power_cap"])
    doc = _nested_set(doc, ["thermal", "throttled_sec"], 2.5)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    validate_summary(p)


def test_all_op_values_valid(tmp_path: Path):
    for op in ("sgemm", "fft", "memstream"):
        doc = _nested_set(_VALID_SUMMARY, ["invocation", "op"], op)
        p = tmp_path / f"summary_{op}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        validate_summary(p)  # must not raise


def test_all_precision_values_valid(tmp_path: Path):
    for prec in ("fp32", "tf32", "fp16", "bf16", "fp8"):
        doc = _nested_set(_VALID_SUMMARY, ["invocation", "precision"], prec)
        p = tmp_path / f"summary_{prec}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        validate_summary(p)


def test_all_result_values_valid(tmp_path: Path):
    for result in ("ok", "throttle_hit", "oom", "cuda_error", "nvml_error", "aborted"):
        doc = _mutated(result=result)
        p = tmp_path / f"summary_{result}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        validate_summary(p)


# ---------------------------------------------------------------------------
# Negative tests — invalid documents must raise ValidationError
# ---------------------------------------------------------------------------


def test_missing_schema_version_invalid(tmp_path: Path):
    doc = copy.deepcopy(_VALID_SUMMARY)
    del doc["schema_version"]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_wrong_schema_version_invalid(tmp_path: Path):
    doc = _mutated(schema_version=2)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_missing_rung_id_invalid(tmp_path: Path):
    doc = copy.deepcopy(_VALID_SUMMARY)
    del doc["rung_id"]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_invalid_op_invalid(tmp_path: Path):
    doc = _nested_set(_VALID_SUMMARY, ["invocation", "op"], "unknown_op")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_invalid_precision_invalid(tmp_path: Path):
    doc = _nested_set(_VALID_SUMMARY, ["invocation", "precision"], "int8")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_negative_avg_w_invalid(tmp_path: Path):
    doc = _nested_set(_VALID_SUMMARY, ["power", "avg_w"], -1.0)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_missing_power_field_invalid(tmp_path: Path):
    doc = copy.deepcopy(_VALID_SUMMARY)
    del doc["power"]["p50_w"]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_additional_top_level_property_invalid(tmp_path: Path):
    """The schema uses additionalProperties: false."""
    doc = _mutated(unexpected_field="should_fail")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_invalid_result_value_invalid(tmp_path: Path):
    doc = _mutated(result="great_success")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_sm_avg_pct_over_100_invalid(tmp_path: Path):
    doc = _nested_set(_VALID_SUMMARY, ["utilization", "sm_avg_pct"], 101.0)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(jsonschema.ValidationError):
        validate_summary(p)


def test_not_found_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_summary(tmp_path / "no_such_file.json")
