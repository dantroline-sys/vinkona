# Knowledge-host filesystem confinement — source this from every script here.
#
# THE GUARANTEE: everything the knowledge host writes stays inside this folder:
#   var/         indexes + databases (index.db, kb.db, library.db, lance/) — the
#                config defaults point here; absolute paths in config.toml are
#                honoured if you deliberately choose somewhere else
#   var/cache/   third-party caches (XDG, HF)
#   var/tmp/     temp files (TMPDIR — OCR scratch, mktemp, pip build isolation)
#   models/      the reranker GGUF fetched by run-reranker.sh
#   .venv/       the virtualenv built by install.sh
#   config.toml  seeded once by install.sh, then yours
#
# Reads are unrestricted — `sources`, `zim_path`, `library_sources` can point
# anywhere. Process-scoped: affects these scripts only, not your shell.

KH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$KH_ROOT/var/cache"
export HF_HOME="$KH_ROOT/var/cache/huggingface"
export TMPDIR="$KH_ROOT/var/tmp"
mkdir -p "$KH_ROOT/var/cache" "$KH_ROOT/var/tmp"
