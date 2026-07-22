#!/bin/bash
# AMIGA-OPS-01 §4 — Vinkona's gates, one entry point.  Every check prints PASS,
# FAIL, or SKIPPED(tool absent); a SKIP is loud, never a silent pass (B-21).
# Exit 0 only when nothing FAILED and nothing REQUIRED was skipped.
#
#   scripts/gates.sh              # everything available
#   GATES_ALLOW_SKIP=1 …          # tolerate missing G-1..G-4 tools (sandboxes)
#
# G-1 ruff format --check   G-2 ruff check   G-3 deptry   (install: uv sync --group dev)
# Always-on stdlib gates: compile sweep of the broker + touched modules, the
# egress-broker battery, the posture battery, and the G-8 broker size cap.
set -u
cd "$(dirname "$0")/.."
FAIL=0; SKIP=0

note() { printf '  %-26s %s\n' "$1" "$2"; }
run()  {  # name, required-tool, cmd...
    local name="$1" tool="$2"; shift 2
    if [ -n "$tool" ] && ! command -v "$tool" >/dev/null 2>&1; then
        note "$name" "SKIPPED ($tool not installed — uv sync --group dev)"
        SKIP=$((SKIP+1)); return
    fi
    if "$@" >/tmp/vkgate.$$ 2>&1; then note "$name" "PASS"
    else note "$name" "FAIL"; sed 's/^/      /' /tmp/vkgate.$$ | head -30; FAIL=$((FAIL+1)); fi
    rm -f /tmp/vkgate.$$
}

PY="python3"
echo "gates ($(git rev-parse --short HEAD 2>/dev/null || echo '?')):"
run "G-1 format"        ruff   ruff format --check assistant/amiga_net assistant/posture.py assistant/netadmin.py
run "G-2 lint"          ruff   ruff check assistant/amiga_net assistant/posture.py assistant/netadmin.py

# ── always-on, stdlib, no excuses ────────────────────────────────────────────
run "compile sweep"     ""     "$PY" -W error::SyntaxWarning -m py_compile \
                                   assistant/amiga_net/*.py assistant/posture.py \
                                   assistant/netadmin.py assistant/test_amiga_net.py \
                                   assistant/test_posture.py
run "broker battery"    ""     "$PY" assistant/test_amiga_net.py
run "posture battery"   ""     "$PY" assistant/test_posture.py
run "G-8 broker size"   ""     "$PY" - <<'EOF'
import pathlib, sys
n = sum(len(p.read_text().splitlines())
        for p in pathlib.Path("assistant/amiga_net").glob("*.py"))
print(f"amiga_net: {n} lines (cap 1000)")
sys.exit(0 if n < 1000 else 1)
EOF

echo
if [ "$FAIL" -gt 0 ]; then echo "gates: $FAIL FAILED"; exit 1; fi
if [ "$SKIP" -gt 0 ] && [ "${GATES_ALLOW_SKIP:-0}" != 1 ]; then
    echo "gates: $SKIP skipped and GATES_ALLOW_SKIP is not set — a skipped gate"
    echo "is not a passed gate.  Install the tools: uv sync --group dev"
    exit 1
fi
echo "gates: all green${SKIP:+ ($SKIP skipped, allowed)}"
