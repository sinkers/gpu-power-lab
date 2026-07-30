"""
upload.py — ship campaign results to S3 and (optionally) Amazon Timestream.

S3
--
Copies the entire campaign directory tree to::

    s3://<bucket>/<prefix>/<campaign_dir_name>/...

preserving the local structure.  Empty files are skipped.

Timestream
----------
Two tables:

* ``table_metrics``   — one MULTI-measure record per NDJSON sample line.
* ``table_summaries`` — one MULTI-measure record per rung summary.

See :func:`_build_metric_records` and :func:`_build_summary_record` for
the exact dimensions/measures schema.

CLI usage::

    python upload.py \\
        --campaign-dir /tmp/out/h100-20260728T120000Z \\
        --bucket my-results-bucket \\
        [--prefix campaigns] \\
        [--timestream-db gpu-lab \\
         --timestream-metrics-table metrics \\
         --timestream-summaries-table summaries]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-type map
# ---------------------------------------------------------------------------

_CONTENT_TYPES: dict[str, str] = {
    ".json":    "application/json",
    ".ndjson":  "application/x-ndjson",
    ".stderr":  "text/plain",
    ".txt":     "text/plain",
    ".log":     "text/plain",
    ".csv":     "text/csv",
    ".parquet": "application/octet-stream",
}


def _content_type_for(path: Path) -> str:
    """Return a MIME Content-Type for *path* based on its suffix."""
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _iter_files(base: Path) -> Iterator[Path]:
    """
    Yield all non-empty regular files under *base*, depth-first, sorted.

    Empty files are logged at WARNING level and skipped so we don't
    produce zero-byte S3 objects that confuse downstream consumers.
    """
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if p.stat().st_size == 0:
            log.warning("skipping empty file: %s", p.relative_to(base))
            continue
        yield p


def upload_campaign_to_s3(
    campaign_dir: Path,
    bucket: str,
    prefix: str,
    boto3_session=None,
) -> dict:
    """
    Upload the entire campaign directory tree to S3 preserving structure.

    S3 keys are formed as::

        <prefix>/<campaign_dir_name>/<relative_path>

    If *prefix* is empty the key starts directly with the campaign dir name.

    Parameters
    ----------
    campaign_dir:
        Local campaign output directory (e.g. ``/tmp/out/h100-20260728T120000Z``).
    bucket:
        S3 bucket name.
    prefix:
        S3 key prefix.  Leading/trailing slashes are normalised.  May be empty.
    boto3_session:
        Optional pre-configured :class:`boto3.Session`.  If *None*, a fresh
        session is created (honours env-var credentials / instance profile).

    Returns
    -------
    dict
        ``{"uploaded_files": int, "total_bytes": int, "s3_base": "s3://..."}``
    """
    try:
        import boto3 as _boto3  # imported lazily so the module works without boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 upload; add it to requirements.txt") from exc

    session = boto3_session if boto3_session is not None else _boto3.Session()
    s3 = session.client("s3")

    campaign_name = campaign_dir.name
    stripped_prefix = prefix.strip("/")
    key_prefix = f"{stripped_prefix}/{campaign_name}" if stripped_prefix else campaign_name

    uploaded = 0
    total_bytes = 0

    for local_path in _iter_files(campaign_dir):
        rel = local_path.relative_to(campaign_dir)
        key = f"{key_prefix}/{rel.as_posix()}"
        size = local_path.stat().st_size
        content_type = _content_type_for(local_path)

        log.debug(
            "upload %s → s3://%s/%s  (%d B, %s)",
            rel, bucket, key, size, content_type,
        )
        s3.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        uploaded += 1
        total_bytes += size

    s3_base = f"s3://{bucket}/{key_prefix}"
    log.info(
        "S3 upload complete: %d files, %d bytes → %s",
        uploaded, total_bytes, s3_base,
    )
    return {"uploaded_files": uploaded, "total_bytes": total_bytes, "s3_base": s3_base}


# ---------------------------------------------------------------------------
# Timestream helpers — pure functions, no AWS client
# ---------------------------------------------------------------------------


def _iso_to_ms(ts_str: str) -> str:
    """
    Convert an ISO 8601 UTC timestamp string to milliseconds-since-epoch (str).

    Accepts the format emitted by the C runner: ``"2026-07-28T12:34:56.789Z"``.
    Python 3.11+ :meth:`datetime.fromisoformat` handles the trailing Z natively.
    Naive timestamps (no timezone) are assumed UTC.
    """
    parsed = dt.datetime.fromisoformat(ts_str)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return str(int(parsed.timestamp() * 1000))


def _add_double_measure(
    measures: list[dict],
    name: str,
    sample: dict,
    sample_key: str | None = None,
) -> None:
    """
    Append a DOUBLE MeasureValue for *name* to *measures* from *sample*.

    Silently skips if the key is absent or the value cannot be cast to float.
    Keeping this as a standalone helper makes it easy to test the conversion
    logic in isolation.
    """
    key = sample_key or name
    raw = sample.get(key)
    if raw is None:
        return
    try:
        measures.append({"Name": name, "Value": str(float(raw)), "Type": "DOUBLE"})
    except (TypeError, ValueError):
        pass


def _build_metric_records(
    ndjson_path: Path,
    rung_id: str = "",
    campaign: str = "",
    device_uuid: str = "",
) -> list[dict]:
    """
    Parse *ndjson_path* (``metrics.ndjson``) and return Timestream
    ``WriteRecords``-compatible record dicts with ``MeasureValueType="MULTI"``.

    One Timestream record is produced per NDJSON line.  Lines that cannot be
    parsed, lack a ``ts`` field, or carry no recognisable measures are skipped
    with a warning log.

    NDJSON line shape (from C runner telemetry.c)::

        {
          "ts": "2026-07-28T10:00:05.120Z",
          "rung_id": "...", "phase": "steady",
          "power_w": 312.456, "temp_c": 68.0,
          "sm_mhz": 1599, "mem_mhz": 2619,
          "sm_util": 99, "mem_util": 47,
          "energy_mj": 100012578, "throttle_mask": 4, "fan_pct": 55
        }

    Parameters
    ----------
    ndjson_path:
        Path to ``metrics.ndjson``.
    rung_id:
        Fallback rung identifier used when the NDJSON line omits ``rung_id``.
    campaign:
        Campaign name added as a Timestream Dimension.
    device_uuid:
        Device UUID added as a Timestream Dimension.

    Returns
    -------
    list[dict]
        Ready-to-pass records for ``client.write_records(Records=...)``.
    """
    records: list[dict] = []
    if not ndjson_path.exists() or ndjson_path.stat().st_size == 0:
        return records

    with ndjson_path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample: dict = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("metrics.ndjson line %d: JSON parse error — %s", lineno, exc)
                continue

            ts_str = sample.get("ts", "")
            if not ts_str:
                log.warning("metrics.ndjson line %d: missing 'ts', skipping", lineno)
                continue
            try:
                time_ms = _iso_to_ms(str(ts_str))
            except (ValueError, TypeError) as exc:
                log.warning(
                    "metrics.ndjson line %d: bad ts %r — %s, skipping", lineno, ts_str, exc
                )
                continue

            measure_values: list[dict] = []
            _add_double_measure(measure_values, "power_w", sample)
            _add_double_measure(measure_values, "temp_c",  sample)
            _add_double_measure(measure_values, "sm_util", sample)
            _add_double_measure(measure_values, "mem_util", sample)
            _add_double_measure(measure_values, "sm_mhz",  sample)
            _add_double_measure(measure_values, "mem_mhz", sample)

            if not measure_values:
                log.debug(
                    "metrics.ndjson line %d: no recognisable measures, skipping", lineno
                )
                continue

            rec_rung_id = str(sample.get("rung_id") or rung_id)
            phase = str(sample.get("phase", "unknown"))

            records.append(
                {
                    "Dimensions": [
                        {"Name": "rung_id",     "Value": rec_rung_id},
                        {"Name": "phase",       "Value": phase},
                        {"Name": "campaign",    "Value": campaign},
                        {"Name": "device_uuid", "Value": device_uuid},
                    ],
                    "MeasureName": "gpu_metrics",
                    "MeasureValueType": "MULTI",
                    "MeasureValues": measure_values,
                    "Time": time_ms,
                    "TimeUnit": "MILLISECONDS",
                }
            )

    return records


def _build_summary_record(summary: dict, campaign: str = "") -> dict | None:
    """
    Build a single Timestream ``WriteRecords``-compatible MULTI record from a
    rung summary dict (already parsed from ``summary.json``).

    Returns *None* if required fields are absent (logs a WARNING).

    Dimensions vs. Measures design decision
    ----------------------------------------
    **Dimensions** — low-cardinality string identifiers; good for WHERE / GROUP BY;
    stored indexed by Timestream:

        ``rung_id``, ``campaign``, ``device_uuid``, ``device_name``,
        ``op``, ``precision``, ``size``, ``streams``.

        ``size`` and ``streams`` are dimensions (not measures) because they are
        categorical experiment parameters used for slicing, not quantities to
        aggregate.  Both have small, bounded cardinality per campaign (e.g.
        four matrix sizes × five stream counts).

    **Measures** — numeric quantities being tracked over time:

        ``tflops_measured`` (DOUBLE), ``avg_w`` (DOUBLE), ``peak_w`` (DOUBLE),
        ``p95_w`` (DOUBLE), ``p99_w`` (DOUBLE), ``energy_j`` (DOUBLE),
        ``avg_c`` (DOUBLE), ``peak_c`` (DOUBLE), ``throttled_sec`` (DOUBLE),
        ``sm_avg_pct`` (DOUBLE), ``mem_avg_pct`` (DOUBLE),
        ``iterations`` (BIGINT).
    """
    try:
        rung_id   = str(summary["rung_id"])
        start_utc = str(summary["timing"]["start_utc"])
        device    = summary["device"]
        inv       = summary["invocation"]
        power     = summary["power"]
        thermal   = summary["thermal"]
        compute   = summary["compute"]
    except KeyError as exc:
        log.warning("summary missing required key %s — skipping Timestream record", exc)
        return None

    try:
        time_ms = _iso_to_ms(start_utc)
    except (ValueError, TypeError) as exc:
        log.warning("summary start_utc %r unparseable: %s — skipping record", start_utc, exc)
        return None

    util = summary.get("utilization", {})

    def _dbl(name: str, value) -> dict:
        return {"Name": name, "Value": str(float(value)), "Type": "DOUBLE"}

    def _bigint(name: str, value) -> dict:
        return {"Name": name, "Value": str(int(value)), "Type": "BIGINT"}

    measure_values = [
        _dbl("tflops_measured", compute.get("tflops_measured",  0.0)),
        _dbl("avg_w",           power.get("avg_w",              0.0)),
        _dbl("peak_w",          power.get("peak_w",             0.0)),
        _dbl("p95_w",           power.get("p95_w",              0.0)),
        _dbl("p99_w",           power.get("p99_w",              0.0)),
        _dbl("energy_j",        power.get("energy_j",           0.0)),
        _dbl("avg_c",           thermal.get("avg_c",            0.0)),
        _dbl("peak_c",          thermal.get("peak_c",           0.0)),
        _dbl("throttled_sec",   thermal.get("throttled_sec",    0.0)),
        _bigint("iterations",   compute.get("iterations",       0)),
        _dbl("sm_avg_pct",      util.get("sm_avg_pct",          0.0)),
        _dbl("mem_avg_pct",     util.get("mem_avg_pct",         0.0)),
    ]

    return {
        "Dimensions": [
            {"Name": "rung_id",     "Value": rung_id},
            {"Name": "campaign",    "Value": campaign},
            {"Name": "device_uuid", "Value": str(device.get("uuid",  ""))},
            {"Name": "device_name", "Value": str(device.get("name",  ""))},
            {"Name": "op",          "Value": str(inv.get("op",        ""))},
            {"Name": "precision",   "Value": str(inv.get("precision", ""))},
            {"Name": "size",        "Value": str(inv.get("size",      ""))},
            {"Name": "streams",     "Value": str(inv.get("streams",   ""))},
        ],
        "MeasureName": "rung_summary",
        "MeasureValueType": "MULTI",
        "MeasureValues": measure_values,
        "Time": time_ms,
        "TimeUnit": "MILLISECONDS",
    }


# ---------------------------------------------------------------------------
# Timestream write — AWS client calls
# ---------------------------------------------------------------------------


def _write_records_with_retry(
    ts_client,
    database: str,
    table: str,
    records: list[dict],
) -> int:
    """
    Write *records* to Timestream with resilience against ``RejectedRecordsException``.

    Behaviour
    ---------
    * **RejectedRecordsException**: Timestream has already written the valid
      records from the batch; only the rejected ones failed.  We log the
      rejected record indices and reasons, then return
      ``len(records) - len(rejected)``.  We do **not** retry rejected records
      because they typically fail for structural reasons (duplicate timestamp,
      bad value type) that will persist.
    * **Other transient errors**: up to 3 attempts with exponential backoff
      (1 s, 2 s, 4 s).

    Returns
    -------
    int
        Number of records accepted by Timestream.
    """
    if not records:
        return 0

    for attempt in range(3):
        try:
            ts_client.write_records(
                DatabaseName=database,
                TableName=table,
                Records=records,
            )
            return len(records)

        except Exception as exc:  # noqa: BLE001
            resp = getattr(exc, "response", None) or {}
            err_code = (
                resp.get("Error", {}).get("Code", "")
                if isinstance(resp, dict)
                else ""
            )
            exc_name = type(exc).__name__

            is_rejected = (
                exc_name == "RejectedRecordsException"
                or err_code == "RejectedRecordsException"
                or ("RejectedRecords" in exc_name and "Exception" in exc_name)
            )

            if is_rejected:
                # Valid records ARE already written.  Drop rejected, log.
                rejected_list: list[dict] = (
                    resp.get("RejectedRecords", []) if isinstance(resp, dict) else []
                )
                rejected_indices = {int(r.get("RecordIndex", -1)) for r in rejected_list}
                reasons = [r.get("Reason", "") for r in rejected_list]
                log.warning(
                    "Timestream table %s: %d/%d records rejected "
                    "(indices=%s reasons=%s) — dropping and continuing",
                    table,
                    len(rejected_indices),
                    len(records),
                    sorted(rejected_indices),
                    reasons,
                )
                return len(records) - len(rejected_indices)

            # Transient error — retry with backoff.
            log.warning(
                "Timestream table %s write error (attempt %d/3): %s",
                table, attempt + 1, exc,
            )
            if attempt == 2:
                raise
            time.sleep(2.0 ** attempt)

    return 0  # unreachable; satisfies type checker


def _write_in_batches(
    ts_client,
    database: str,
    table: str,
    records: list[dict],
    batch_size: int = 100,
) -> int:
    """
    Write *records* to *table* in batches of at most *batch_size*.

    Returns total number of records accepted.
    """
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        total += _write_records_with_retry(ts_client, database, table, batch)
    return total


def _find_rung_dirs(campaign_dir: Path) -> list[Path]:
    """Return subdirs of *campaign_dir* that contain a ``summary.json``, sorted."""
    return sorted(
        d for d in campaign_dir.iterdir()
        if d.is_dir() and (d / "summary.json").exists()
    )


def write_campaign_to_timestream(
    campaign_dir: Path,
    database: str,
    table_metrics: str,
    table_summaries: str,
    boto3_session=None,
) -> dict:
    """
    Write per-sample telemetry and per-rung summaries to Amazon Timestream.

    Parameters
    ----------
    campaign_dir:
        Local campaign output directory.
    database:
        Timestream database name.
    table_metrics:
        Timestream table for per-sample NDJSON records (one record per line).
    table_summaries:
        Timestream table for per-rung summary records (one record per rung).
    boto3_session:
        Optional pre-configured :class:`boto3.Session`.

    Returns
    -------
    dict
        ``{"metric_records_written": int, "summary_records_written": int}``
    """
    try:
        import boto3 as _boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for Timestream upload; add it to requirements.txt") from exc

    session = boto3_session if boto3_session is not None else _boto3.Session()
    ts = session.client("timestream-write")

    # Derive campaign name from manifest.json when present.
    campaign_name = campaign_dir.name
    manifest_path = campaign_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            campaign_name = str(manifest.get("campaign", campaign_name))
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read manifest.json (%s) — using dir name as campaign", exc)

    rung_dirs = _find_rung_dirs(campaign_dir)
    log.info(
        "writing %d rungs to Timestream database=%s "
        "metrics_table=%s summaries_table=%s",
        len(rung_dirs), database, table_metrics, table_summaries,
    )

    total_metrics = 0
    total_summaries = 0

    for rung_dir in rung_dirs:
        summary_path = rung_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read %s (%s) — skipping rung", summary_path, exc)
            continue

        rung_id = str(summary.get("rung_id", rung_dir.name))
        device_uuid = str(summary.get("device", {}).get("uuid", ""))

        # --- per-sample metrics ---
        ndjson_path = rung_dir / "metrics.ndjson"
        metric_records = _build_metric_records(
            ndjson_path, rung_id, campaign_name, device_uuid
        )
        if metric_records:
            written = _write_in_batches(ts, database, table_metrics, metric_records)
            total_metrics += written
            log.info("  rung %-52s metrics=%d", rung_id, written)
        else:
            log.debug("  rung %s: no metric records (empty or missing ndjson)", rung_id)

        # --- per-rung summary ---
        rec = _build_summary_record(summary, campaign_name)
        if rec:
            written = _write_records_with_retry(ts, database, table_summaries, [rec])
            total_summaries += written
        else:
            log.warning("  rung %s: could not build summary record — skipping", rung_id)

    log.info(
        "Timestream write complete: %d metric records, %d summary records",
        total_metrics, total_summaries,
    )
    return {
        "metric_records_written": total_metrics,
        "summary_records_written": total_summaries,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Upload a gpu-power-lab campaign to S3 and/or Amazon Timestream."
    )
    ap.add_argument(
        "--campaign-dir", required=True, metavar="DIR",
        help="Path to the campaign output directory.",
    )
    ap.add_argument("--bucket", required=True, help="S3 bucket name.")
    ap.add_argument(
        "--prefix", default="",
        help="S3 key prefix (default: empty — keys start with campaign dir name).",
    )
    ap.add_argument("--timestream-db", metavar="DB",
                    help="Timestream database name.")
    ap.add_argument("--timestream-metrics-table", metavar="TABLE",
                    help="Timestream table for per-sample metrics.")
    ap.add_argument("--timestream-summaries-table", metavar="TABLE",
                    help="Timestream table for per-rung summaries.")

    ns = ap.parse_args()

    campaign_dir = Path(ns.campaign_dir)
    if not campaign_dir.is_dir():
        log.error(
            "--campaign-dir does not exist or is not a directory: %s", campaign_dir
        )
        return 2

    # S3 upload
    try:
        result = upload_campaign_to_s3(campaign_dir, ns.bucket, ns.prefix)
        print(
            f"S3: {result['uploaded_files']} files, {result['total_bytes']} bytes"
            f" → {result['s3_base']}"
        )
    except Exception as exc:  # noqa: BLE001
        log.error("S3 upload failed: %s", exc)
        return 1

    # Timestream (optional)
    if ns.timestream_db:
        if not (ns.timestream_metrics_table and ns.timestream_summaries_table):
            log.error(
                "--timestream-metrics-table and --timestream-summaries-table are "
                "both required when --timestream-db is set"
            )
            return 2
        try:
            ts_result = write_campaign_to_timestream(
                campaign_dir,
                ns.timestream_db,
                ns.timestream_metrics_table,
                ns.timestream_summaries_table,
            )
            print(
                f"Timestream: {ts_result['metric_records_written']} metric records, "
                f"{ts_result['summary_records_written']} summary records"
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Timestream write failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
