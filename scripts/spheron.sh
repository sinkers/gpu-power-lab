#!/usr/bin/env bash
#
# Spheron deployment automation — the load-shave aws_boot.py equivalent.
#
#   export SPHERON_API_KEY=...
#   ./scripts/spheron.sh offers b300        # what is available, and at what price
#   ./scripts/spheron.sh deploy <offerId> <gpuType> <region> [gpuCount]
#   ./scripts/spheron.sh status             # active deployments + ssh command
#   ./scripts/spheron.sh ssh                # ssh into the deployment we launched
#   ./scripts/spheron.sh terminate [id]     # stop the billing
#
# API: https://docs.spheron.ai/api-reference   Base: https://app.spheron.ai
#
# Deployments are billed by the hour and this script can start them, so it
# prints the hourly rate and asks before deploying. `terminate` is the one
# you must not forget — see the trap note in `deploy`.

set -euo pipefail

BASE="https://app.spheron.ai"
STATE="${HOME}/.gpu-power-lab/spheron-state.json"
KEY_PATH="${SPHERON_SSH_KEY:-$HOME/.ssh/gpu-power-lab}"

: "${SPHERON_API_KEY:?set SPHERON_API_KEY (generate one in the Spheron dashboard)}"

api() {  # method path [json-body]
    local method="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -fsS -X "$method" "$BASE$path" \
            -H "Authorization: Bearer $SPHERON_API_KEY" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -fsS -X "$method" "$BASE$path" \
            -H "Authorization: Bearer $SPHERON_API_KEY"
    fi
}

cmd_offers() {
    local search="${1:-}"
    echo "==> GPU offers${search:+ matching '$search'}"
    api GET "/api/gpu-offers?limit=100${search:+&search=$search}" | python3 -c '
import json,sys
d=json.load(sys.stdin)
rows=d.get("data",d) if isinstance(d,dict) else d
if not rows: print("  (none)"); sys.exit()
print("  %-26s %-12s %-16s %-10s %8s  %s"%("offerId","gpuType","region","type","$/hr","provider"))
for o in rows:
    print("  %-26s %-12s %-16s %-10s %8s  %s"%(
        str(o.get("offerId") or o.get("id"))[:26],
        str(o.get("gpuType"))[:12],
        str(o.get("region") or o.get("regions"))[:16],
        str(o.get("instanceType"))[:10],
        o.get("hourlyRate", o.get("price","?")),
        o.get("providerId") or o.get("provider","")))'
}

cmd_deploy() {
    local offer="${1:?offerId}" gputype="${2:?gpuType}" region="${3:?region}" count="${4:-1}"
    [ -f "${KEY_PATH}.pub" ] || { echo "no public key at ${KEY_PATH}.pub" >&2; exit 1; }
    local pub; pub="$(cat "${KEY_PATH}.pub")"

    # cloud-init installs the driver and toolchain during provisioning, so the
    # box is measurement-ready when SSH answers rather than 8-10 minutes after.
    # startup-script.sh is shipped via writeFiles so there is one source of
    # truth shared with hand-launched boxes.
    local startup
    startup="$(cat "$(dirname "$0")/startup-script.sh")"

    local body
    body=$(python3 - "$offer" "$gputype" "$region" "$count" "$pub" "$startup" <<'PY'
import json,sys
offer,gputype,region,count,pub,startup = sys.argv[1:7]
print(json.dumps({
  "provider": "spheron-ai",
  "offerId": offer,
  "gpuType": gputype,
  "gpuCount": int(count),
  "region": region,
  "operatingSystem": "ubuntu-22.04",
  "instanceType": "DEDICATED",   # never SPOT for a measurement run
  "ssh_public_key": pub,
  "name": "gpu-power-lab",
  "cloudInit": {
    "packages": ["wget", "curl", "git", "cmake", "build-essential",
                 "python3", "rsync"],
    "writeFiles": [{
      "path": "/opt/gpu-power-lab-startup.sh",
      "content": startup,
      "owner": "root:root",
      "permissions": "0755"
    }],
    "runcmd": ["bash /opt/gpu-power-lab-startup.sh"]
  }
}))
PY
)
    echo "==> deploying $count x $gputype ($offer) in $region"
    echo "    DEDICATED, billed hourly. Terminate when done:  $0 terminate"
    read -r -p "    proceed? [y/N] " ok
    [ "$ok" = "y" ] || { echo "aborted"; exit 1; }

    mkdir -p "$(dirname "$STATE")"
    api POST "/api/deployments" "$body" | tee "$STATE" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("    id=%s status=%s $%.2f/hr"%(d.get("id"),d.get("status"),d.get("hourlyRate") or 0))'
    echo "==> poll with: $0 status"
}

cmd_status() {
    api GET "/api/deployments?status=active" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
if not rows: print("  no active deployments"); sys.exit()
for d in rows:
    print("  %s  %s  %s x%s  %s  $%.2f/hr"%(
        d.get("id"), d.get("status"), d.get("gpuType"), d.get("gpuCount"),
        d.get("ipAddress") or "(no ip yet)", d.get("hourlyRate") or 0))
    if d.get("sshCommand"): print("    ssh: %s"%d["sshCommand"])'
}

cmd_ssh() {
    local ip
    ip=$(api GET "/api/deployments?status=active" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
ips=[r.get("ipAddress") for r in rows if r.get("ipAddress")]
print(ips[0] if ips else "")')
    [ -n "$ip" ] || { echo "no deployment with an IP yet" >&2; exit 1; }
    echo "==> ssh ubuntu@$ip"
    exec ssh -i "$KEY_PATH" -o StrictHostKeyChecking=accept-new "ubuntu@$ip"
}

cmd_terminate() {
    local id="${1:-}"
    if [ -z "$id" ]; then
        id=$(api GET "/api/deployments?status=active" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
print(rows[0]["id"] if rows else "")')
    fi
    [ -n "$id" ] || { echo "nothing active to terminate"; exit 0; }
    # Providers enforce a minimum runtime; asking first turns a rejected
    # request into a clear message rather than a silently-still-billing box.
    api GET "/api/deployments/$id/can-terminate" || true
    echo
    echo "==> terminating $id"
    api DELETE "/api/deployments/$id" | python3 -c '
import json,sys; d=json.load(sys.stdin)
print("   ", d.get("message"), d.get("deployment",{}).get("status"))'
}

case "${1:-}" in
    offers)    shift; cmd_offers "$@" ;;
    deploy)    shift; cmd_deploy "$@" ;;
    status)    shift; cmd_status "$@" ;;
    ssh)       shift; cmd_ssh "$@" ;;
    terminate) shift; cmd_terminate "$@" ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
