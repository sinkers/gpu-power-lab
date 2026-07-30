"""
test_upload.py — pytest tests for upload.py.

S3 tests use moto to mock the AWS API; no real credentials needed.
Timestream is NOT mocked (moto support is limited) — only the pure
record-building helpers (_build_metric_records, _build_summary_record)
are tested directly without any AWS client.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from upload import (
    _add_double_measure,
    _build_metric_records,
    _build_summary_record,
    _content_type_for,
    _find_rung_dirs,
    _iso_to_ms,
    upload_campaign_to_s3,
)

# ---------------------------------------------------------------------------
# Paths to shared fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_NDJSON = FIXTURES_DIR / "sample_metrics.ndjson"
SAMPLE_SUMMARY = FIXTURES_DIR / "sample_summary.json"

_BUCKET = "test-gpu-results"
_REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> boto3.Session:
    """Return a boto3 session pointed at the mock AWS environment."""
    return boto3.Session(region_name=_REGION)


def _make_campaign_dir(base: Path, name: str = "my-campaign-20260728T120000Z") -> Path:
    """
    Create a minimal two-rung campaign directory under *base* and return it.

    Structure::

        <name>/
          manifest.json
          summaries.json
          rung-0000-sgemm-fp16-4096-s4/
            summary.json   (non-empty)
            metrics.ndjson (non-empty)
            runner.stderr  (non-empty)
          rung-0001-sgemm-fp32-4096-s4/
            summary.json
            metrics.ndjson
            runner.stderr
    """
    campaign_dir = base / name
    campaign_dir.mkdir(parents=True)

    (campaign_dir / "manifest.json").write_text(
        json.dumps({"campaign": "my-campaign", "device": 0, "rungs": []}),
        encoding="utf-8",
    )
    (campaign_dir / "summaries.json").write_text("[]", encoding="utf-8")

    for rung_name, op_precision in [
        ("rung-0000-sgemm-fp16-4096-s4", "fp16"),
        ("rung-0001-sgemm-fp32-4096-s4", "fp32"),
    ]:
        rung_dir = campaign_dir / rung_name
        rung_dir.mkdir()
        (rung_dir / "summary.json").write_text(
            json.dumps({"rung_id": rung_name, "result": "ok"}), encoding="utf-8"
        )
        (rung_dir / "metrics.ndjson").write_text(
            '{"ts":"2026-07-28T10:00:00.000Z","rung_id":"' + rung_name + '",'
            '"phase":"steady","power_w":300.0,"temp_c":65.0,'
            '"sm_mhz":1599,"mem_mhz":2619,"sm_util":99,"mem_util":45}\n',
            encoding="utf-8",
        )
        (rung_dir / "runner.stderr").write_text(
            f"runner for {op_precision} finished ok\n", encoding="utf-8"
        )

    return campaign_dir


# ---------------------------------------------------------------------------
# _content_type_for
# ---------------------------------------------------------------------------


def test_content_type_json():
    assert _content_type_for(Path("x.json")) == "application/json"


def test_content_type_ndjson():
    assert _content_type_for(Path("x.ndjson")) == "application/x-ndjson"


def test_content_type_unknown():
    assert _content_type_for(Path("x.xyz")) == "application/octet-stream"


# ---------------------------------------------------------------------------
# _iso_to_ms
# ---------------------------------------------------------------------------


def test_iso_to_ms_basic():
    # 2026-07-28T00:00:00.000Z
    ms = int(_iso_to_ms("2026-07-28T00:00:00.000Z"))
    assert ms == 1785196800000


def test_iso_to_ms_with_millis():
    ms1 = int(_iso_to_ms("2026-07-28T10:00:05.120Z"))
    ms2 = int(_iso_to_ms("2026-07-28T10:00:05.130Z"))
    assert ms2 - ms1 == 10  # 10 ms apart


def test_iso_to_ms_no_millis():
    ms = int(_iso_to_ms("2026-07-28T10:00:05Z"))
    assert ms > 0


# ---------------------------------------------------------------------------
# _add_double_measure
# ---------------------------------------------------------------------------


def test_add_double_measure_present():
    measures: list[dict] = []
    _add_double_measure(measures, "power_w", {"power_w": 312.456})
    assert len(measures) == 1
    assert measures[0] == {"Name": "power_w", "Value": "312.456", "Type": "DOUBLE"}


def test_add_double_measure_absent():
    measures: list[dict] = []
    _add_double_measure(measures, "power_w", {})
    assert measures == []


def test_add_double_measure_bad_value():
    measures: list[dict] = []
    _add_double_measure(measures, "power_w", {"power_w": "not-a-number"})
    assert measures == []


def test_add_double_measure_integer_input():
    measures: list[dict] = []
    _add_double_measure(measures, "sm_mhz", {"sm_mhz": 1599})
    assert measures[0]["Value"] == "1599.0"


def test_add_double_measure_custom_key():
    measures: list[dict] = []
    _add_double_measure(measures, "temp_c", {"temperature": 68.0}, sample_key="temperature")
    assert measures[0]["Name"] == "temp_c"
    assert measures[0]["Value"] == "68.0"


# ---------------------------------------------------------------------------
# _build_metric_records — pure, no AWS client
# ---------------------------------------------------------------------------


def test_build_metric_records_fixture(tmp_path: Path):
    """Read the sample fixture and verify record shapes."""
    records = _build_metric_records(
        SAMPLE_NDJSON,
        rung_id="test-0000-sgemm-fp16-4096-s4",
        campaign="test-campaign",
        device_uuid="GPU-00000000-0000-0000-0000-000000000001",
    )
    # 5 lines, all have the expected measures
    assert len(records) == 5

    for rec in records:
        # Every record must be a MULTI record with required keys.
        assert rec["MeasureName"] == "gpu_metrics"
        assert rec["MeasureValueType"] == "MULTI"
        assert rec["TimeUnit"] == "MILLISECONDS"
        assert isinstance(rec["Time"], str) and rec["Time"].isdigit()

        dim_names = {d["Name"] for d in rec["Dimensions"]}
        assert dim_names == {"rung_id", "phase", "campaign", "device_uuid"}

        measure_names = {m["Name"] for m in rec["MeasureValues"]}
        # All 6 measures should be present in the fixture lines.
        assert measure_names == {"power_w", "temp_c", "sm_util", "mem_util", "sm_mhz", "mem_mhz"}

        # All measure types must be DOUBLE.
        for mv in rec["MeasureValues"]:
            assert mv["Type"] == "DOUBLE"
            float(mv["Value"])  # must be parseable


def test_build_metric_records_phases():
    """First 2 records are warmup, last 3 are steady."""
    records = _build_metric_records(SAMPLE_NDJSON)
    phases = [
        next(d["Value"] for d in r["Dimensions"] if d["Name"] == "phase")
        for r in records
    ]
    assert phases[:2] == ["warmup", "warmup"]
    assert phases[2:] == ["steady", "steady", "steady"]


def test_build_metric_records_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.ndjson"
    empty.write_text("")
    assert _build_metric_records(empty) == []


def test_build_metric_records_missing_file(tmp_path: Path):
    assert _build_metric_records(tmp_path / "nonexistent.ndjson") == []


def test_build_metric_records_bad_line(tmp_path: Path):
    """Lines with parse errors are skipped; valid lines are still returned."""
    ndjson = tmp_path / "mixed.ndjson"
    ndjson.write_text(
        'not json\n'
        '{"ts":"2026-07-28T10:00:00.000Z","power_w":300.0,"temp_c":65.0,'
        '"sm_mhz":1599,"mem_mhz":2619,"sm_util":99,"mem_util":45}\n',
        encoding="utf-8",
    )
    records = _build_metric_records(ndjson)
    assert len(records) == 1


def test_build_metric_records_missing_ts(tmp_path: Path):
    ndjson = tmp_path / "no_ts.ndjson"
    ndjson.write_text(
        '{"rung_id":"x","phase":"steady","power_w":300.0}\n',
        encoding="utf-8",
    )
    assert _build_metric_records(ndjson) == []


def test_build_metric_records_uses_fallback_rung_id(tmp_path: Path):
    ndjson = tmp_path / "no_rung.ndjson"
    ndjson.write_text(
        '{"ts":"2026-07-28T10:00:00.000Z","phase":"steady",'
        '"power_w":300.0,"temp_c":65.0,'
        '"sm_mhz":1599,"mem_mhz":2619,"sm_util":99,"mem_util":45}\n',
        encoding="utf-8",
    )
    records = _build_metric_records(ndjson, rung_id="fallback-id")
    assert len(records) == 1
    rung_dim = next(d for d in records[0]["Dimensions"] if d["Name"] == "rung_id")
    assert rung_dim["Value"] == "fallback-id"


def test_build_metric_records_no_measures(tmp_path: Path):
    """A line with only metadata fields (no measure fields) is skipped."""
    ndjson = tmp_path / "meta_only.ndjson"
    ndjson.write_text(
        '{"ts":"2026-07-28T10:00:00.000Z","rung_id":"x","phase":"steady",'
        '"energy_mj":12345,"throttle_mask":0,"fan_pct":40}\n',
        encoding="utf-8",
    )
    assert _build_metric_records(ndjson) == []


# ---------------------------------------------------------------------------
# _build_summary_record — pure, no AWS client
# ---------------------------------------------------------------------------


def test_build_summary_record_fixture():
    summary = json.loads(SAMPLE_SUMMARY.read_text())
    rec = _build_summary_record(summary, campaign="test-campaign")
    assert rec is not None

    assert rec["MeasureName"] == "rung_summary"
    assert rec["MeasureValueType"] == "MULTI"
    assert rec["TimeUnit"] == "MILLISECONDS"
    assert rec["Time"].isdigit()

    dim_names = {d["Name"] for d in rec["Dimensions"]}
    assert dim_names == {
        "rung_id", "campaign", "device_uuid", "device_name",
        "op", "precision", "size", "streams",
    }

    # Verify a few dimension values.
    dim = {d["Name"]: d["Value"] for d in rec["Dimensions"]}
    assert dim["op"] == "sgemm"
    assert dim["precision"] == "fp16"
    assert dim["size"] == "4096"
    assert dim["streams"] == "4"
    assert dim["campaign"] == "test-campaign"

    measure_names = {m["Name"] for m in rec["MeasureValues"]}
    expected_measures = {
        "tflops_measured", "avg_w", "peak_w", "p95_w", "p99_w",
        "energy_j", "avg_c", "peak_c", "throttled_sec", "iterations",
        "sm_avg_pct", "mem_avg_pct",
    }
    assert expected_measures.issubset(measure_names)

    # iterations must be BIGINT.
    iter_mv = next(m for m in rec["MeasureValues"] if m["Name"] == "iterations")
    assert iter_mv["Type"] == "BIGINT"
    assert int(iter_mv["Value"]) == 87350

    # tflops_measured should be close to fixture value.
    tflops_mv = next(m for m in rec["MeasureValues"] if m["Name"] == "tflops_measured")
    assert abs(float(tflops_mv["Value"]) - 95.1234) < 1e-3


def test_build_summary_record_missing_key():
    """A summary with a required top-level key missing returns None."""
    rec = _build_summary_record({"rung_id": "x"})  # missing timing, device, etc.
    assert rec is None


def test_build_summary_record_bad_timestamp():
    base = json.loads(SAMPLE_SUMMARY.read_text())
    base["timing"]["start_utc"] = "not-a-date"
    rec = _build_summary_record(base)
    assert rec is None


# ---------------------------------------------------------------------------
# upload_campaign_to_s3 — moto S3 mock
# ---------------------------------------------------------------------------


@mock_aws
def test_upload_campaign_to_s3(tmp_path: Path):
    """All non-empty files in a two-rung campaign land at expected S3 keys."""
    session = _make_session()
    session.client("s3").create_bucket(Bucket=_BUCKET)

    campaign_dir = _make_campaign_dir(tmp_path)

    result = upload_campaign_to_s3(campaign_dir, _BUCKET, "campaigns", session)

    assert result["s3_base"] == f"s3://{_BUCKET}/campaigns/my-campaign-20260728T120000Z"
    assert result["uploaded_files"] == 8  # manifest + summaries + 2*(summary + ndjson + stderr)
    assert result["total_bytes"] > 0

    # Verify specific keys exist.
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = {
        obj["Key"]
        for page in paginator.paginate(Bucket=_BUCKET, Prefix="campaigns/")
        for obj in page.get("Contents", [])
    }

    assert "campaigns/my-campaign-20260728T120000Z/manifest.json" in keys
    assert "campaigns/my-campaign-20260728T120000Z/summaries.json" in keys
    assert (
        "campaigns/my-campaign-20260728T120000Z/rung-0000-sgemm-fp16-4096-s4/summary.json"
        in keys
    )
    assert (
        "campaigns/my-campaign-20260728T120000Z/rung-0001-sgemm-fp32-4096-s4/metrics.ndjson"
        in keys
    )
    assert (
        "campaigns/my-campaign-20260728T120000Z/rung-0000-sgemm-fp16-4096-s4/runner.stderr"
        in keys
    )


@mock_aws
def test_upload_campaign_to_s3_no_prefix(tmp_path: Path):
    """With an empty prefix the key starts with the campaign dir name."""
    session = _make_session()
    session.client("s3").create_bucket(Bucket=_BUCKET)
    campaign_dir = _make_campaign_dir(tmp_path)

    result = upload_campaign_to_s3(campaign_dir, _BUCKET, "", session)

    assert result["s3_base"] == f"s3://{_BUCKET}/my-campaign-20260728T120000Z"
    s3 = session.client("s3")
    objs = s3.list_objects_v2(Bucket=_BUCKET, Prefix="my-campaign-20260728T120000Z/")
    assert objs.get("KeyCount", 0) > 0


@mock_aws
def test_upload_content_types(tmp_path: Path):
    """Verify JSON files are uploaded with application/json content-type."""
    session = _make_session()
    session.client("s3").create_bucket(Bucket=_BUCKET)
    campaign_dir = _make_campaign_dir(tmp_path)

    upload_campaign_to_s3(campaign_dir, _BUCKET, "p", session)

    s3 = session.client("s3")
    head = s3.head_object(
        Bucket=_BUCKET,
        Key="p/my-campaign-20260728T120000Z/manifest.json",
    )
    assert head["ContentType"] == "application/json"


@mock_aws
def test_upload_skips_empty(tmp_path: Path):
    """Empty files are skipped and do NOT appear as S3 objects."""
    session = _make_session()
    session.client("s3").create_bucket(Bucket=_BUCKET)

    campaign_dir = tmp_path / "camp-20260728T000000Z"
    campaign_dir.mkdir()
    (campaign_dir / "manifest.json").write_text('{"campaign":"c"}', encoding="utf-8")
    # Create an empty file — must be skipped.
    (campaign_dir / "empty_file.txt").write_text("", encoding="utf-8")
    # Create a non-empty file — must be uploaded.
    (campaign_dir / "data.ndjson").write_text(
        '{"ts":"2026-07-28T00:00:00.000Z"}\n', encoding="utf-8"
    )

    result = upload_campaign_to_s3(campaign_dir, _BUCKET, "x", session)

    # Only 2 non-empty files: manifest.json + data.ndjson.
    assert result["uploaded_files"] == 2

    s3 = session.client("s3")
    keys = {
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=_BUCKET, Prefix="x/")
        for obj in page.get("Contents", [])
    }
    assert "x/camp-20260728T000000Z/empty_file.txt" not in keys
    assert "x/camp-20260728T000000Z/data.ndjson" in keys
    assert "x/camp-20260728T000000Z/manifest.json" in keys


@mock_aws
def test_upload_returns_correct_byte_count(tmp_path: Path):
    """total_bytes in the return dict equals the sum of uploaded file sizes."""
    session = _make_session()
    session.client("s3").create_bucket(Bucket=_BUCKET)

    campaign_dir = tmp_path / "c"
    campaign_dir.mkdir()
    content = b"hello world\n"
    (campaign_dir / "file.txt").write_bytes(content)

    result = upload_campaign_to_s3(campaign_dir, _BUCKET, "", session)
    assert result["total_bytes"] == len(content)
    assert result["uploaded_files"] == 1


# ---------------------------------------------------------------------------
# _find_rung_dirs
# ---------------------------------------------------------------------------


def test_find_rung_dirs(tmp_path: Path):
    campaign_dir = _make_campaign_dir(tmp_path)
    dirs = _find_rung_dirs(campaign_dir)
    names = [d.name for d in dirs]
    assert "rung-0000-sgemm-fp16-4096-s4" in names
    assert "rung-0001-sgemm-fp32-4096-s4" in names
    # top-level files / dirs without summary.json are excluded
    assert len(dirs) == 2
