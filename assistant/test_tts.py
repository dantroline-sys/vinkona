#!/usr/bin/env python
"""
Isolated TTS smoke test for either engine — run BEFORE wiring TTS into the server
to confirm voice quality and latency / real-time factor on your hardware.

The engine wrappers each live in their own venv, so we lazy-import only the one
selected by --engine (run this inside that engine's venv):

  # NeuTTS (in neutts_env): clone a voice from a reference clip
  source neutts_env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python test_tts.py --engine neutts \
      --ref voices/vinkona.wav --text "Hello, I'm Vinkona." --out /tmp/neutts.wav

  # Chatterbox (in chatterbox_env): built-in voice, or clone with --ref
  source chatterbox_env/bin/activate
  python test_tts.py --engine chatterbox --text "Hello." --out /tmp/cb.wav

  # Orpheus on llama.cpp has its own end-to-end test: test_tts_orpheus_gguf.py
"""

import argparse
import time

import numpy as np
import soundfile as sf


def build_engine(args):
    """Lazy-import and construct the selected engine (only its venv has its deps)."""
    if args.engine == "chatterbox":
        from tts_chatterbox import ChatterboxEngine
        eng = ChatterboxEngine(device=args.device)
        eng.register_voice("test", ref_wav=args.ref)   # None = the built-in voice
        return eng, "test"
    if args.engine == "neutts":
        from tts_neutts import NeuTTSEngine
        eng = NeuTTSEngine(backbone_repo=args.backbone or "neuphonic/neutts-air",
                           device=args.device)
        if not args.ref:
            raise SystemExit("--ref <voice.wav> is required for --engine neutts")
        eng.register_voice("test", args.ref, ref_text=args.ref_text)
        return eng, "test"
    raise SystemExit(f"unknown engine {args.engine}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["neutts", "chatterbox"], default="neutts")
    ap.add_argument("--text", default="Hello, this is a test of the text to speech voice.")
    ap.add_argument("--out", default="/tmp/tts_test.wav")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), or explicit (neutts)")
    ap.add_argument("--backbone", default=None, help="override the model repo/name")
    # NeuTTS-specific
    ap.add_argument("--ref", default=None, help="[neutts] reference voice WAV (~3-10 s)")
    ap.add_argument("--ref-text", default=None, help="[neutts] transcript of --ref (else sibling .txt)")
    args = ap.parse_args()

    print(f"Loading engine '{args.engine}' ...")
    t0 = time.monotonic()
    eng, voice = build_engine(args)
    print(f"  ready in {time.monotonic()-t0:.1f}s  (voice={voice})")

    print(f"Synthesizing: {args.text!r}")
    t0 = time.monotonic()
    pcm = eng.synthesize(args.text, voice=voice)
    dt = time.monotonic() - t0
    dur = len(pcm) / eng.sample_rate if len(pcm) else 0.0
    if dur > 0:
        print(f"  {dur:.2f}s of audio in {dt:.2f}s  (RTF {dt/dur:.3f}, {dur/dt:.0f}x real-time)")
    else:
        print(f"  WARNING: produced no audio (in {dt:.2f}s)")

    sf.write(args.out, pcm, eng.sample_rate)
    print(f"Wrote {args.out}  (peak {np.abs(pcm).max():.3f})" if len(pcm) else "No audio written.")


if __name__ == "__main__":
    main()
