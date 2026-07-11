#!/bin/bash
# One-time migration: rename the core virtualenv  personaplex_env -> vinkona_env.
# (Vinkona is a local cascade now, not PersonaPlex — the old name was misleading.)
#
# A venv bakes its own ABSOLUTE PATH into bin/activate* and the console-script shebangs
# (pip, etc.), so a plain `mv` isn't enough — we move it and rewrite those paths in place.
# Idempotent and safe to re-run.  If there's no old venv, a fresh ./install.sh now creates
# vinkona_env directly, so nothing to do.
#
#   ./rename_env.sh           # then:  ./vinkona.sh restart
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD="$DIR/personaplex_env"
NEW="$DIR/vinkona_env"

if [ -d "$NEW" ]; then
  echo "vinkona_env already exists — nothing to do."
  exit 0
fi
if [ ! -d "$OLD" ]; then
  echo "No personaplex_env found — a fresh ./install.sh now creates vinkona_env directly."
  exit 0
fi

echo "Renaming $OLD -> $NEW and fixing the paths baked into the venv…"
mv "$OLD" "$NEW"
# Rewrite the venv's own absolute-path references (activate scripts + bin/ shebangs).  Only
# the 'personaplex_env' path segment changes; the rest of the absolute path is already
# correct for this machine (the venv was created here).
while IFS= read -r f; do
  sed -i "s|personaplex_env|vinkona_env|g" "$f"
done < <(grep -rlI "personaplex_env" "$NEW/bin" 2>/dev/null || true)
[ -f "$NEW/pyvenv.cfg" ] && sed -i "s|personaplex_env|vinkona_env|g" "$NEW/pyvenv.cfg" || true

echo "Done.  Restart the stack with:  ./vinkona.sh restart"
