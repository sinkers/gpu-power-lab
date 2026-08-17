#!/usr/bin/env bash
#
# Resumable campaign runner.
#
#   campaign.sh <plan-file> [state-dir]
#
# Runs a list of rungs, one per line, and is safe to kill and restart. This
# exists because a spot instance vanished mid-batch and took thirty minutes of
# B300 time with it: the rungs had all completed, the results were all written,
# and every one of them was in /tmp on a machine that no longer existed.
#
# Three independent protections, because any one of them can fail:
#
#   1. Results are written to a PERSISTENT VOLUME, not /tmp. Survives the
#      instance if the volume can be reattached.
#   2. A per-rung .done marker means a restart skips finished work. Resuming
#      costs only the rung that was interrupted.
#   3. A drain loop copies each completed rung off the box as it lands (see
#      scripts/drain.sh). This is the one that actually saves you, because it
#      does not assume the volume survives either.
#
# Plan file format - one rung per line, blank lines and # comments ignored:
#
#     <rung-name> <runner arguments...>
#
# e.g.  cap-900w  --op powervirus --mix-tensor 1 --tensor-backend cublas \
#                 --precision bf16 --power-limit 900 --iters 4000

set -u

PLAN="${1:?usage: campaign.sh <plan-file> [state-dir]}"
STATE="${2:-${GPL_WORK:-/mnt/data/gpu-power-lab}}"
RUNNER="${GPL_RUNNER:-$HOME/gpu-power-lab/runner/build/gpu-power-runner}"

mkdir -p "$STATE/results" || {
    echo "cannot write to $STATE — is the persistent volume mounted?" >&2
    echo "override with: GPL_WORK=/some/path $0 $PLAN" >&2
    exit 1
}

LOG="$STATE/campaign.log"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- interruption handling ------------------------------------------------
# Spot reclamation arrives as SIGTERM. We cannot finish the rung in the time
# available, so record that it was cut short rather than leaving a truncated
# NDJSON that looks like a complete measurement.
INTERRUPTED=0
CURRENT=""
on_term() {
    INTERRUPTED=1
    say "SIGTERM — instance is going away"
    if [ -n "$CURRENT" ]; then
        echo "interrupted at $(date -u +%FT%TZ)" > "$STATE/results/$CURRENT.interrupted"
        say "  rung '$CURRENT' marked interrupted; it will re-run on resume"
    fi
    sync
    exit 143
}
trap on_term TERM INT

# --- spot termination notice ---------------------------------------------
# AWS publishes a two-minute warning on the instance metadata service. Other
# providers vary; where there is no notice this simply never fires and the
# SIGTERM path above is the only protection. Polled in the background so it
# does not slow the measurement loop.
watch_for_reclaim() {
    while true; do
        code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' \
               http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)
        if [ "$code" = "200" ]; then
            say "SPOT RECLAMATION NOTICE — roughly two minutes left"
            touch "$STATE/RECLAIM_IMMINENT"
            # Let the drain loop know it should hurry.
            break
        fi
        sleep 5
    done
}
watch_for_reclaim &
WATCH_PID=$!

# --- the loop -------------------------------------------------------------
total=$(grep -cvE '^\s*(#|$)' "$PLAN")
n=0
say "campaign start: $total rungs, state in $STATE"
[ -f "$STATE/RESUMED" ] && say "(this is a resume)"

while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    n=$((n + 1))
    name="${line%% *}"
    args="${line#* }"

    if [ -f "$STATE/results/$name.done" ]; then
        say "[$n/$total] $name — already done, skipping"
        continue
    fi

    # A reclamation notice mid-campaign: stop cleanly rather than starting a
    # rung we cannot finish. A half-measured rung is worse than a missing one.
    if [ -f "$STATE/RECLAIM_IMMINENT" ]; then
        say "reclamation imminent — stopping before '$name' rather than starting it"
        break
    fi

    CURRENT="$name"
    say "[$n/$total] $name"
    rm -f "$STATE/results/$name.interrupted"

    # shellcheck disable=SC2086
    sudo "$RUNNER" $args \
        --out-summary "$STATE/results/rung-$name.json" \
        --out-metrics "$STATE/results/rung-$name.ndjson" \
        >>"$STATE/results/$name.stderr" 2>&1
    rc=$?

    if [ "$rc" = "0" ] && [ -s "$STATE/results/rung-$name.json" ]; then
        # Marker written last and only on success, so a partially written
        # result is never mistaken for a finished one on resume.
        sync
        date -u +%FT%TZ > "$STATE/results/$name.done"
        python3 - "$STATE/results/rung-$name.json" <<'PY' | tee -a "$LOG"
import json,sys
d=json.load(open(sys.argv[1])); p=d['power']; e=d.get('efficiency',{}); t=d.get('thermal',{})
print('    avg %7.1fW  %%lim %5.1f  %3.0fC  edp %.4g  %s'%(
 p['avg_w'], p['pct_of_enforced_limit'], t.get('peak_c',0),
 e.get('edp_j_s',0), ','.join(d['throttle']['reasons']) or '-'))
PY
    else
        say "    FAILED (exit $rc) — see $name.stderr; will retry on resume"
    fi
    CURRENT=""
done < "$PLAN"

kill "$WATCH_PID" 2>/dev/null
touch "$STATE/RESUMED"

done_n=$(ls "$STATE/results"/*.done 2>/dev/null | wc -l | tr -d ' ')
say "campaign end: $done_n/$total rungs complete"
[ "$done_n" -lt "$total" ] && say "resume by re-running this exact command"
exit 0
