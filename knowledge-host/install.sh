#!/usr/bin/env bash
# Vinkona Knowledge Host — production installer.
#
# Builds a self-contained Python virtualenv (.venv) and installs exactly the
# dependencies your deployment needs.  The query service and the default
# `sqlite` backend require NOTHING beyond the Python stdlib (FTS5 ships with
# sqlite3); numpy is the recommended dense fast-path; every document parser and
# the Wikipedia-scale `lance` backend are optional and lazy-imported, so a host
# only installs what its corpus actually contains.
#
# Usage:
#   ./install.sh                 # venv + numpy  (sqlite backend, query service)
#   ./install.sh --all           # + every parser (pdf/epub/html/zim) + lance
#   ./install.sh --pdf --epub    # pick formats your collection has
#   ./install.sh --wikipedia     # libzim, for a Kiwix Wikipedia ZIM
#   ./install.sh --lance         # LanceDB IVF-PQ backend (Wikipedia scale)
#   ./install.sh --minimal       # stdlib only, not even numpy
#   ./install.sh --no-venv       # install into the active/system interpreter
#   ./install.sh --python python3.12 --venv /opt/kb/venv
#   ./install.sh --no-test       # skip the post-install smoke verification
#
# Re-running is safe and incremental: an existing venv is reused and pip just
# adds anything new.  config.toml is seeded once and never overwritten.
#
# Maintenance commands:
#   ./install.sh status          # what's installed and how big the data is
#   ./install.sh uninstall       # remove the venv, caches and reranker model
#                 --purge        #   ALSO delete the knowledge base (var/) and
#                                #   config.toml — asks for confirmation first
#
# Everything this script and the host write lives INSIDE this folder (see
# env.sh): .venv, var/ (indexes, kb.db, caches, tmp), models/, config.toml.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh          # in-tree var/ caches + tmp — see env.sh

# ── maintenance commands (before flag parsing) ──────────────────────────────
if [ "${1:-}" = "status" ]; then
  echo "Knowledge host @ $(pwd)"
  [ -d .venv ]  && echo "  venv     .venv ($(du -sh .venv 2>/dev/null | cut -f1))" || echo "  venv     not installed"
  [ -d var ]    && echo "  data     var/ ($(du -sh var 2>/dev/null | cut -f1)) — indexes, kb.db, caches" || echo "  data     none yet"
  [ -d models ] && echo "  models   models/ ($(du -sh models 2>/dev/null | cut -f1))"
  [ -f config.toml ] && echo "  config   config.toml (yours, kept on uninstall)" || echo "  config   not seeded yet"
  echo "  All of the above lives inside this folder — nothing is written elsewhere."
  exit 0
fi
if [ "${1:-}" = "uninstall" ]; then
  purge=0; [ "${2:-}" = "--purge" ] && purge=1
  rm -rf .venv models var/cache var/tmp
  find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  echo "removed: .venv, models/, var/cache, var/tmp (software + caches)"
  if [ "$purge" -eq 1 ]; then
    echo ""
    echo "WARNING: --purge deletes the KNOWLEDGE BASE itself (var/: index.db, kb.db,"
    echo "the distilled cards — potentially days of ingestion) and config.toml."
    printf "Type 'purge' to confirm: "
    read -r answer
    if [ "$answer" = "purge" ]; then
      rm -rf var config.toml
      echo "knowledge base and config purged"
    else
      echo "skipped purge — var/ and config.toml kept"
    fi
  else
    echo "kept: var/ (your knowledge base) and config.toml — './install.sh uninstall --purge' removes them too"
  fi
  exit 0
fi

# ── defaults ────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"
USE_VENV=1
WANT_NUMPY=1
RUN_TEST=1
WITH_PDF=0; WITH_EPUB=0; WITH_HTML=0; WITH_WIKI=0; WITH_LANCE=0

# Print the leading comment block (everything after the shebang up to the first
# non-comment line), with the leading "# " stripped.
usage() { sed -n '2,/^[^#]/p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --all)       WITH_PDF=1; WITH_EPUB=1; WITH_HTML=1; WITH_WIKI=1; WITH_LANCE=1 ;;
    --pdf)       WITH_PDF=1 ;;
    --epub)      WITH_EPUB=1 ;;
    --html)      WITH_HTML=1 ;;
    --wikipedia|--zim) WITH_WIKI=1 ;;
    --lance)     WITH_LANCE=1 ;;
    --minimal|--no-numpy) WANT_NUMPY=0 ;;
    --no-venv)   USE_VENV=0 ;;
    --venv)      VENV_DIR="${2:?--venv needs a path}"; shift ;;
    --python)    PYTHON="${2:?--python needs an interpreter}"; shift ;;
    --no-test)   RUN_TEST=0 ;;
    -h|--help)   usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. interpreter + version (tomllib needs >= 3.11) ────────────────────────
command -v "$PYTHON" >/dev/null 2>&1 || die "interpreter not found: $PYTHON (try --python python3.11)"
"$PYTHON" - <<'PY' || die "Python >= 3.11 is required (config.py uses the stdlib tomllib)."
import sys
sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)
PY
say "using $("$PYTHON" -V) at $(command -v "$PYTHON")"

# ── 2. virtualenv ───────────────────────────────────────────────────────────
if [ "$USE_VENV" -eq 1 ]; then
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    say "creating virtualenv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR" \
      || die "venv creation failed — install the venv module (e.g. 'dnf install python3-virtualenv' / 'apt install python3-venv')"
  else
    say "reusing virtualenv at $VENV_DIR"
  fi
  PY="$VENV_DIR/bin/python"
else
  warn "installing into the active interpreter (no venv): $(command -v "$PYTHON")"
  PY="$PYTHON"
fi

say "upgrading pip"
"$PY" -m pip install --quiet --upgrade pip

# ── 3. dependency set (only what was requested) ─────────────────────────────
PKGS=()
[ "$WANT_NUMPY" -eq 1 ] && PKGS+=("numpy" "usearch")   # usearch: optional ANN (build-ann)
[ "$WITH_PDF"   -eq 1 ] && PKGS+=("pymupdf")
[ "$WITH_EPUB"  -eq 1 ] && PKGS+=("ebooklib")
[ "$WITH_HTML"  -eq 1 ] && PKGS+=("trafilatura")
[ "$WITH_WIKI"  -eq 1 ] && PKGS+=("libzim")
[ "$WITH_LANCE" -eq 1 ] && PKGS+=("lancedb" "pyarrow")

if [ "${#PKGS[@]}" -gt 0 ]; then
  say "installing: ${PKGS[*]}"
  "$PY" -m pip install --upgrade "${PKGS[@]}"
else
  say "stdlib-only install (no pip packages requested)"
fi

# ── 4. system binaries we cannot pip-install (offer, never block) ───────────
if [ "$WITH_PDF" -eq 1 ]; then
  vk_require_tools "tesseract:tesseract|tesseract-ocr" ocrmypdf || {
    warn "OCR fallback unavailable — scanned PDFs with no text layer will be skipped."
    warn "(PDFs with a real text layer ingest fine without it; re-run once installed.)"
  }
fi

# ── 5. seed config.toml once (never clobber an existing one) ────────────────
if [ ! -f config.toml ] && [ -f config.example.toml ]; then
  cp config.example.toml config.toml
  say "seeded config.toml from config.example.toml — edit 'sources', 'embed_url', etc."
else
  say "config.toml left as-is"
fi

# ── 6. post-install verification (offline; no embed endpoint needed) ────────
if [ "$RUN_TEST" -eq 1 ]; then
  say "verifying the install (offline smoke test)"
  FIX="$(mktemp -d)"
  trap 'rm -rf "$FIX"' EXIT
  bash tests/make_fixtures.sh "$FIX" >/dev/null
  if "$PY" tests/smoke.py "$FIX"; then
    say "smoke test passed"
  else
    die "smoke test failed — see output above"
  fi
fi

# ── done ────────────────────────────────────────────────────────────────────
cat <<EOF

$(say "install complete")
  Next:
    1. edit ./config.toml      (sources, embed_url, zim_path, port, auth_token)
    2. ./ingest.sh             (crawl documents; add --wikipedia for a ZIM)
    3. ./run.sh                (start the query service)

  run.sh / ingest.sh auto-use $VENV_DIR when present.
EOF
