#!/bin/bash
# Install the Orpheus-on-llama.cpp TTS path (engine "orpheus_gguf").
#
# No new venv, no torch: the Orpheus 3B backbone runs as a GGUF on a
# plain llama-server (the tts_lm tier), and the SNAC vocoder decodes on the CPU
# via onnxruntime inside vinkona_env.  Total footprint: one ~3.4 GB GGUF, one
# ~50 MB ONNX file, one pip wheel.  Works on any Python vinkona_env runs on.
#
# What this does:
#   1. pip install onnxruntime into vinkona_env (+ import verify)
#   2. get an Orpheus GGUF into Models/ — your choice: download the default
#      (pick a quant with ORPHEUS_GGUF_QUANT=F16|Q8_0|Q4_K_M ...), select a
#      .gguf you already have in Models/, or skip the download entirely
#   3. fetch the SNAC decoder ONNX into Models/
#   4. point config tts_lm.model + tts.orpheus_gguf.snac_path at them
#   5. decode a real SNAC window as a smoke test
#   6. offer to make orpheus_gguf the active engine (interactive only)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
cd "$SCRIPT_DIR"

GGUF_REPO="${ORPHEUS_GGUF_REPO:-unsloth/orpheus-3b-0.1-ft-GGUF}"
GGUF_QUANT="${ORPHEUS_GGUF_QUANT:-Q8_0}"          # Q8 ≈ lossless; Q4 dulls the emotion tags
GGUF_FILE="orpheus-3b-0.1-ft-${GGUF_QUANT}.gguf"
SNAC_REPO="onnx-community/snac_24khz-ONNX"
SNAC_FILE="onnx/decoder_model.onnx"
SNAC_OUT="Models/snac_24khz_decoder.onnx"

[ -f vinkona_env/bin/activate ] \
    || { echo "ERROR: vinkona_env is missing — run './install.sh core' first."; exit 1; }

echo "== onnxruntime into vinkona_env (SNAC vocoder, CPU) =="
./vinkona_env/bin/pip install --quiet onnxruntime
./vinkona_env/bin/python -c "import onnxruntime" \
    || { echo "ERROR: onnxruntime did not import after install."; exit 1; }

echo "== Orpheus GGUF =="
mkdir -p Models
existing="$(find -L Models -maxdepth 2 -iname '*orpheus*.gguf' -print -quit 2>/dev/null)"
tty=0; { [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; } && tty=1

_download_gguf() {
    echo "downloading $GGUF_REPO :: $GGUF_FILE (~3.4 GB for Q8_0) ..."
    vk_hf_download "$GGUF_REPO" "$GGUF_FILE" Models    # python API — see env.sh
}

if [ "$tty" -ne 1 ]; then
    # Non-interactive: deterministic — an Orpheus-named GGUF already in Models/
    # wins, otherwise download the default.
    if [ -n "$existing" ]; then
        GGUF_FILE="$(basename "$existing")"
        echo "found $existing — using it"
    else
        _download_gguf
    fi
else
    echo "The tts_lm tier needs the Orpheus 3B GGUF (the TTS voice backbone)."
    [ -n "$existing" ] && echo "Found one already in Models/: $(basename "$existing")"
    echo "  1) download $GGUF_REPO :: $GGUF_FILE"
    echo "  2) use a .gguf already in Models/ (pick from a list — symlinks welcome)"
    echo "  s) skip — keep whatever config tts_lm.model already points at"
    def=1; [ -n "$existing" ] && def=2
    printf "choice [%s]: " "$def"
    read -r c; c="${c:-$def}"
    case "$c" in
        1) _download_gguf ;;
        2)  files=()
            while IFS= read -r f; do [ -n "$f" ] && files+=("$f"); done \
                <<<"$(find -L Models -maxdepth 2 -name '*.gguf' 2>/dev/null | sort)"
            if [ "${#files[@]}" -eq 0 ]; then
                echo "no .gguf files in Models/ — copy/symlink one in and re-run, or pick the download option."
                exit 1
            fi
            i=1; for f in "${files[@]}"; do echo "    $i) $(basename "$f")"; i=$((i+1)); done
            def=""
            if [ -n "$existing" ]; then
                i=1; for f in "${files[@]}"; do [ "$f" = "$existing" ] && def="$i"; i=$((i+1)); done
            fi
            printf "which one%s: " "${def:+ [$def]}"
            read -r n; n="${n:-$def}"
            case "$n" in *[!0-9]*|"") echo "not a number — aborting, nothing changed."; exit 1 ;; esac
            { [ "$n" -ge 1 ] && [ "$n" -le "${#files[@]}" ]; } \
                || { echo "out of range — aborting, nothing changed."; exit 1; }
            GGUF_FILE="$(basename "${files[$((n-1))]}")"
            case "$GGUF_FILE" in
                *[Oo]rpheus*) ;;
                *) echo "note: '$GGUF_FILE' doesn't look like an Orpheus model — a plain chat GGUF"
                   echo "      produces no audio tokens, so the voice would come out silent." ;;
            esac ;;
        s|S) GGUF_FILE=""
             echo "skipped — config tts_lm.model stays as it is." ;;
        *) echo "unknown choice: $c"; exit 1 ;;
    esac
fi

echo "== SNAC vocoder decoder (ONNX, ~50 MB) =="
if [ -f "$SNAC_OUT" ]; then
    echo "found $SNAC_OUT — keeping it"
else
    tmp="Models/.snac_dl"
    vk_hf_download "$SNAC_REPO" "$SNAC_FILE" "$tmp"
    mv "$tmp/$SNAC_FILE" "$SNAC_OUT"
    rm -rf "$tmp"
fi

echo "== Pointing config at the files =="
mkdir -p config
[ -f config/config.json ] || cp config/config.example.json config/config.json
./vinkona_env/bin/python - "$GGUF_FILE" "$SNAC_OUT" <<'PY'
import json, sys
from pathlib import Path
gguf, snac = sys.argv[1], sys.argv[2]
path = "config/config.json"
cfg = json.load(open(path))
if gguf:
    cfg.setdefault("tts_lm", {})["model"] = gguf
    print(f"  config: tts_lm.model = {gguf}")
cfg.setdefault("tts", {}).setdefault("orpheus_gguf", {})["snac_path"] = snac
json.dump(cfg, open(path, "w"), indent=2)
print(f"  config: tts.orpheus_gguf.snac_path = {snac}")
# Whether just set or kept from before (the skip option), the model the tts_lm
# server will try to load should actually exist — say so now, not at startup.
m = (cfg.get("tts_lm") or {}).get("model")
if m:
    p = Path(m)
    if not p.is_absolute():
        p = Path(cfg.get("models_dir") or "Models") / m
    if not p.exists():
        print(f"  warning: config tts_lm.model = {m} but {p} does not exist —")
        print(f"           the tts_lm server won't start until it does (re-run this task to fix)")
else:
    print("  warning: config tts_lm.model is not set — the tts_lm server won't start")
    print("           until it is (re-run this task and pick or download a GGUF)")
PY

echo "== Verifying (decoding a real SNAC window on the CPU) =="
./vinkona_env/bin/python - "$SNAC_OUT" <<'PY'
import sys
import numpy as np
import onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(sys.argv[1], sess_options=so, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
assert len(names) == 3, f"unexpected decoder inputs: {names}"
rng = np.random.default_rng(0)
feed = {names[0]: rng.integers(0, 4096, (1, 4), dtype=np.int64),
        names[1]: rng.integers(0, 4096, (1, 8), dtype=np.int64),
        names[2]: rng.integers(0, 4096, (1, 16), dtype=np.int64)}
out = sess.run(None, feed)[0]
assert out.shape == (1, 1, 8192), f"unexpected output shape {out.shape}"
print(f"  SNAC decode ok: 4 frames -> {out.shape[2]} samples @ 24 kHz")
PY

echo ""
echo "Done — Orpheus (llama.cpp) is installed. It needs no extra venv."
current="$(./vinkona_env/bin/python -c "import json;print(json.load(open('config/config.json')).get('tts',{}).get('engine') or 'orpheus')" 2>/dev/null || echo orpheus)"
if [ "$current" = "orpheus_gguf" ]; then
    echo "config tts.engine is already 'orpheus_gguf' — a restart picks everything up."
elif [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
    printf "Make orpheus_gguf the active TTS engine now (currently '%s')? [Y/n]: " "$current"
    read -r a
    case "$a" in
        n*|N*) echo "kept '$current' — switch later by setting tts.engine to \"orpheus_gguf\" in the config UI." ;;
        *) ./vinkona_env/bin/python - <<'PY'
import json
path = "config/config.json"
cfg = json.load(open(path))
cfg.setdefault("tts", {})["engine"] = "orpheus_gguf"
json.dump(cfg, open(path, "w"), indent=2)
print("  config: tts.engine = orpheus_gguf")
PY
           echo "switched — './vinkona.sh restart' starts the tts_lm llama-server + the new engine." ;;
    esac
elif [ "$current" = "orpheus" ]; then
    # Non-interactive, and the configured engine is the retired vLLM one —
    # switching can only fix things, never break a working setup.
    ./vinkona_env/bin/python - <<'PY'
import json
path = "config/config.json"
cfg = json.load(open(path))
cfg.setdefault("tts", {})["engine"] = "orpheus_gguf"
json.dump(cfg, open(path, "w"), indent=2)
print("  config: tts.engine = orpheus_gguf (was 'orpheus', which isn't installed)")
PY
else
    echo "To use it, set tts.engine to \"orpheus_gguf\" (config UI or config/config.json), then restart."
fi
