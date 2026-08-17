#!/usr/bin/env bash
#
# Continuously copy finished rungs off a remote box, while the campaign runs.
#
#   ./scripts/drain.sh ubuntu@<ip> results/<campaign> [-i ~/.ssh/key]
#
# Run this on the laptop, in parallel with the campaign. It is the protection
# that actually works, because unlike a persistent volume it does not assume
# anything about the remote machine surviving.
#
# The lesson it encodes: batch 2's results were collected at the end and
# survived; batch 3's were also written per-rung, also complete, and were lost
# entirely because collection only happened at the end and the spot instance
# went away first. Same code path, different luck. Continuous drain removes
# the luck.
#
# Only files with a matching .done marker are pulled, so a rung that is
# half-written when the box disappears never lands locally looking complete.

set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 user@host <local-dir> [ssh args...]" >&2; exit 2
fi
HOST="$1"; shift
DEST="$1"; shift
SSHA=("$@")
REMOTE="${GPL_WORK:-/mnt/data/gpu-power-lab}/results"
INTERVAL="${GPL_DRAIN_INTERVAL:-45}"

mkdir -p "$DEST"
echo "==> draining $HOST:$REMOTE -> $DEST every ${INTERVAL}s"
echo "    (ctrl-c to stop; safe to stop and restart at any time)"

pulled_total=0
while true; do
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${SSHA[@]}" "$HOST" true 2>/dev/null; then
        echo "[$(date -u +%H:%M:%S)] host unreachable — retrying"
        sleep "$INTERVAL"
        continue
    fi

    # Which rungs have a .done marker and are not yet local?
    remote_done=$(ssh "${SSHA[@]}" "$HOST" \
        "ls $REMOTE/*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.done\$//'" 2>/dev/null)

    # Fallback for campaigns that predate campaign.sh and write no markers:
    # treat a summary JSON untouched for 20 s as finished. Cruder than a
    # marker, but the alternative is pulling nothing at all, which is exactly
    # how batch 3 was lost.
    if [ -z "$remote_done" ]; then
        remote_done=$(ssh "${SSHA[@]}" "$HOST" \
            "find $REMOTE -name 'rung-*.json' -mmin +0.34 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/^rung-//; s/\.json\$//'" 2>/dev/null)
    fi

    new=0
    for name in $remote_done; do
        if [ ! -f "$DEST/rung-$name.json" ]; then
            if scp -q "${SSHA[@]}" \
                 "$HOST:$REMOTE/rung-$name.json" "$HOST:$REMOTE/rung-$name.ndjson" \
                 "$DEST/" 2>/dev/null; then
                new=$((new + 1)); pulled_total=$((pulled_total + 1))
                echo "[$(date -u +%H:%M:%S)] pulled $name"
            fi
        fi
    done

    # The campaign log and any stderr are small and worth having every pass.
    scp -q "${SSHA[@]}" "$HOST:${GPL_WORK:-/mnt/data/gpu-power-lab}/campaign.log" \
        "$DEST/" 2>/dev/null || true

    if ssh "${SSHA[@]}" "$HOST" "[ -f ${GPL_WORK:-/mnt/data/gpu-power-lab}/RECLAIM_IMMINENT ]" 2>/dev/null; then
        echo "[$(date -u +%H:%M:%S)] RECLAMATION NOTICE on the box — draining hard"
        scp -q "${SSHA[@]}" "$HOST:$REMOTE/*" "$DEST/" 2>/dev/null || true
        echo "    grabbed everything available. Instance is going away."
        break
    fi

    [ "$new" = "0" ] && printf "." || echo "    ($pulled_total rungs local, $(du -sh "$DEST" 2>/dev/null | cut -f1))"
    sleep "$INTERVAL"
done

echo
echo "==> $pulled_total rungs in $DEST"
echo "    python3 scripts/generate_report.py $DEST"
