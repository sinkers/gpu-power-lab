"""
campaign.py — run a full campaign end-to-end.

For each rung:
  1. Invoke the C runner as a subprocess.
  2. Capture its two output files into the campaign directory.
  3. Wait for thermal cooldown before the next rung.

At the end, collate all summary JSONs into a Parquet + manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from cooldown import CooldownConfig, capture_baseline, wait as cooldown_wait
from plan import Plan, RungSpec, load_plan

# upload is an optional dependency — imported only when needed.
# This keeps campaign.py functional even without boto3 installed.
try:
    import upload as _upload
    _UPLOAD_AVAILABLE = True
except ImportError:  # pragma: no cover
    _UPLOAD_AVAILABLE = False

log = logging.getLogger("campaign")

DEFAULT_RUNNER = str(Path(__file__).resolve().parent.parent /
                     "runner" / "build" / "gpu-power-runner")


def _run_one(runner: str, rung: RungSpec, rung_dir: Path) -> dict:
    rung_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = rung_dir / "metrics.ndjson"
    summary_path = rung_dir / "summary.json"
    stderr_path  = rung_dir / "runner.stderr"

    cli = [runner] + rung.to_cli_args(metrics_path, summary_path)
    log.info("→ %s", " ".join(cli))
    t0 = time.monotonic()
    with stderr_path.open("wb") as errf:
        rc = subprocess.call(cli, stderr=errf, stdout=errf)
    elapsed = time.monotonic() - t0

    entry = {
        "rung_id": rung.rung_id,
        "exit_code": rc,
        "wall_sec": elapsed,
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "stderr_path": str(stderr_path),
    }
    if summary_path.exists():
        try:
            entry["summary"] = json.loads(summary_path.read_text())
        except Exception as e:  # noqa: BLE001
            entry["summary_load_error"] = str(e)
    else:
        entry["summary_missing"] = True

    if rc != 0:
        log.warning("rung %s exited %d (see %s)", rung.rung_id, rc, stderr_path)
    else:
        s = entry.get("summary", {})
        power = s.get("power", {})
        thermal = s.get("thermal", {})
        comp = s.get("compute", {})
        log.info("   ok  tflops=%.2f  avg_w=%.1f peak_w=%.1f  peak_c=%.1f  throttled_s=%.2f",
                 comp.get("tflops_measured", 0.0),
                 power.get("avg_w", 0.0),
                 power.get("peak_w", 0.0),
                 thermal.get("peak_c", 0.0),
                 thermal.get("throttled_sec", 0.0))
    return entry


def run_campaign(
    plan: Plan,
    out_dir: Path,
    runner: str,
    skip_cooldown: bool,
    upload_bucket: str | None = None,
    upload_prefix: str = "",
    timestream_db: str | None = None,
    timestream_metrics_table: str | None = None,
    timestream_summaries_table: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = out_dir / f"{plan.campaign}-{now}"
    campaign_dir.mkdir()

    manifest_path = campaign_dir / "manifest.json"

    if not skip_cooldown:
        log.info("capturing thermal baseline (60s idle window)...")
        baseline = capture_baseline(plan.device, sample_sec=60.0)
        log.info("baseline temp: %.1fC", baseline)
    else:
        baseline = 40.0
        log.info("skipping baseline capture (cooldown disabled)")

    manifest = {
        "campaign": plan.campaign,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": plan.device,
        "baseline_c": baseline,
        "rungs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    cooldown_cfg = CooldownConfig(baseline_c=baseline)

    for i, rung in enumerate(plan.rungs):
        log.info("─── rung %d/%d : %s ───", i + 1, len(plan.rungs), rung.rung_id)
        rung_dir = campaign_dir / rung.rung_id
        entry = _run_one(runner, rung, rung_dir)
        manifest["rungs"].append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2))

        if not skip_cooldown and i < len(plan.rungs) - 1:
            cr = cooldown_wait(plan.device, cooldown_cfg)
            log.info("cooldown: waited %.1fs, temp=%.1fC, target_reached=%s",
                     cr.waited_sec, cr.final_temp_c, cr.reached_target)

    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Collate summaries as a small JSON side-table for convenience.
    all_summaries = [r["summary"] for r in manifest["rungs"] if "summary" in r]
    (campaign_dir / "summaries.json").write_text(json.dumps(all_summaries, indent=2))

    log.info("campaign complete: %s", campaign_dir)

    # ------------------------------------------------------------------
    # Optional upload — non-fatal: log ERROR but still return the path.
    # ------------------------------------------------------------------
    if upload_bucket:
        if not _UPLOAD_AVAILABLE:
            log.error(
                "upload requested but 'upload' module could not be imported "
                "(is boto3 installed?); skipping upload"
            )
        else:
            # S3
            try:
                s3_result = _upload.upload_campaign_to_s3(
                    campaign_dir, upload_bucket, upload_prefix
                )
                log.info(
                    "S3 upload: %d files, %d bytes → %s",
                    s3_result["uploaded_files"],
                    s3_result["total_bytes"],
                    s3_result["s3_base"],
                )
            except Exception as exc:  # noqa: BLE001
                log.error("S3 upload failed (campaign data is safe locally): %s", exc)

            # Timestream
            if timestream_db and timestream_metrics_table and timestream_summaries_table:
                try:
                    ts_result = _upload.write_campaign_to_timestream(
                        campaign_dir,
                        timestream_db,
                        timestream_metrics_table,
                        timestream_summaries_table,
                    )
                    log.info(
                        "Timestream write: %d metric records, %d summary records",
                        ts_result["metric_records_written"],
                        ts_result["summary_records_written"],
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("Timestream write failed (campaign data is safe locally): %s", exc)

    return campaign_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run a gpu-power-lab campaign.")
    ap.add_argument("--plan", required=True, help="Path to YAML plan file.")
    ap.add_argument("--out-dir", required=True, help="Directory for campaign outputs.")
    ap.add_argument("--runner", default=DEFAULT_RUNNER,
                    help="Path to gpu-power-runner binary.")
    ap.add_argument("--no-cooldown", action="store_true",
                    help="Skip thermal cooldown between rungs (debug only).")
    # Upload options (all optional — absent = no upload)
    ap.add_argument("--upload-bucket", default=None, metavar="BUCKET",
                    help="S3 bucket to upload campaign results to after the run.")
    ap.add_argument("--upload-prefix", default="", metavar="PREFIX",
                    help="S3 key prefix for uploaded files (default: empty).")
    ap.add_argument("--timestream-db", default=None, metavar="DB",
                    help="Amazon Timestream database name for telemetry upload.")
    ap.add_argument("--timestream-metrics-table", default=None, metavar="TABLE",
                    help="Timestream table for per-sample metrics (requires --timestream-db).")
    ap.add_argument("--timestream-summaries-table", default=None, metavar="TABLE",
                    help="Timestream table for per-rung summaries (requires --timestream-db).")
    ns = ap.parse_args()

    if not os.path.exists(ns.runner):
        log.error("runner not found at %s (build it first: cmake --build runner/build -j)",
                  ns.runner)
        return 2

    plan = load_plan(ns.plan)
    log.info("plan: campaign=%s device=%d rungs=%d",
             plan.campaign, plan.device, len(plan.rungs))
    run_campaign(
        plan,
        Path(ns.out_dir),
        ns.runner,
        ns.no_cooldown,
        upload_bucket=ns.upload_bucket,
        upload_prefix=ns.upload_prefix,
        timestream_db=ns.timestream_db,
        timestream_metrics_table=ns.timestream_metrics_table,
        timestream_summaries_table=ns.timestream_summaries_table,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
