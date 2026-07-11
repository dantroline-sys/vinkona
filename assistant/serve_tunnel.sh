#!/bin/bash
# SSH tunnel to the Mac tool host (Tier-2 tools).
#
# The tool host stays bound to 127.0.0.1:8765 ON THE MAC — never exposed on the LAN.
# This forwards Vinkona's localhost:<local_port> → the Mac's <remote_host>:<remote_port>
# over SSH, so config tools.url (http://127.0.0.1:8765) reaches it securely.
# Reads config/config.json (tools.tunnel); auto-reconnects.
#
# One-time setup:
#   ssh-keygen -t ed25519 -f ~/.ssh/vinkona_tunnel -N ''        # private key (+ .pub)
#   ssh-copy-id -i ~/.ssh/vinkona_tunnel.pub  USER@MAC_IP       # or append .pub to the
#                                                             # Mac's ~/.ssh/authorized_keys
# Then set tools.tunnel.{host,user} in config and tools.tunnel.enabled = true.
#
#   ./serve_tunnel.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Line 1: connection params (tab-separated).  Lines 2+: one "-L" forward spec each
# (the main tool-host forward, then any tools.tunnel.extra_forwards, e.g. SearXNG).
mapfile -t TUNNEL_LINES < <(python3 - <<'PY'
import importlib.util
s = importlib.util.spec_from_file_location("config", "config.py")
c = importlib.util.module_from_spec(s); s.loader.exec_module(c)
t = c.load_config("config/config.json")["tools"]["tunnel"]
print("\t".join(str(v) for v in [
    t.get("host", ""), t.get("user", ""), t.get("port", 22),
    t.get("identity", "~/.ssh/vinkona_tunnel"), "1" if t.get("enabled") else "0"]))
fwds = [(t.get("local_port", 8765), t.get("remote_host", "127.0.0.1"), t.get("remote_port", 8765))]
for f in (t.get("extra_forwards") or []):
    fwds.append((f.get("local_port"), f.get("remote_host", "127.0.0.1"), f.get("remote_port")))
for lp, rh, rp in fwds:
    if lp and rp:
        print(f"{lp}:{rh}:{rp}")
PY
)
IFS=$'\t' read -r HOST MUSER SSHPORT IDENT ENABLED <<< "${TUNNEL_LINES[0]}"
FWDS=("${TUNNEL_LINES[@]:1}")

if [ "$ENABLED" != "1" ]; then
    echo "tools.tunnel.enabled is false in config — nothing to do."; exit 0
fi
if [ -z "$HOST" ] || [ -z "$MUSER" ]; then
    echo "Set tools.tunnel.host (the Mac's IP) and tools.tunnel.user in config."; exit 1
fi
IDENT="${IDENT/#\~/$HOME}"                       # expand a leading ~
if [ ! -f "$IDENT" ]; then
    echo "Private key not found: $IDENT"
    echo "Generate it and put its .pub on the Mac (see the header of this script)."; exit 1
fi

if [ "${#FWDS[@]}" -eq 0 ]; then
    echo "No forwards configured (tools.tunnel.local_port/remote_port)."; exit 1
fi
echo "Tunnel: ${MUSER}@${HOST}:${SSHPORT}  (key ${IDENT})"
FWD_OPTS=()
for f in "${FWDS[@]}"; do
    FWD_OPTS+=(-L "$f")
    echo "  forward localhost:${f%%:*} -> ${f#*:}"
done

OPTS=(-i "$IDENT" -p "$SSHPORT" -N "${FWD_OPTS[@]}"
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3
      -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new
      "${MUSER}@${HOST}")

# autossh reconnects on its own; otherwise loop plain ssh.
if command -v autossh >/dev/null 2>&1; then
    exec autossh -M 0 "${OPTS[@]}"
fi
while true; do
    ssh "${OPTS[@]}"
    echo "tunnel dropped — reconnecting in 5s"; sleep 5
done
