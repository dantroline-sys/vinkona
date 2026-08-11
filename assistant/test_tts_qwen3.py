"""Tests for the qwen3 TTS engine (tts_qwen3.py), which wraps the official
`qwen-tts` package.

The heavy deps (torch + qwen-tts + a GPU) aren't in the sandbox, so we inject a
FAKE qwen_tts + torch and pin the engine's own logic against the documented API
shape:
  • contract: sample_rate / voices;
  • natural-language voice control: a voice NAME → its style DESCRIPTION as the
    `instruct` arg (unknown name → literal; default falls back to a real style);
  • synthesize(): calls generate_voice_design(text, language, instruct=…), takes
    wavs[0], returns float32 in [-1,1], and adopts the returned sample rate;
  • synthesize_stream(): synth-then-chunk framing to 16-bit PCM byte chunks.

Needs numpy (the engine returns np arrays) — skips cleanly without it.

Run inside qwen3_env:  python test_tts_qwen3.py
"""
import sys
import types

try:
    import numpy as np
except Exception as e:
    print(f"  skip qwen3 engine tests (needs numpy): {e}")
    raise SystemExit(0)


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


# ── fake torch + qwen_tts, injected before importing the engine ───────────────
class _FakeModel:
    last = None                       # records the last generate_voice_design call

    @classmethod
    def from_pretrained(cls, repo, **kw):
        m = cls(); m.repo = repo; m.kw = kw
        return m

    def generate_voice_design(self, text, language, instruct, **gen):
        _FakeModel.last = {"text": text, "language": language,
                           "instruct": instruct, "gen": gen}
        # 0.1 s of quiet-ish audio at 24 kHz, batch of one, as numpy (the model
        # may return torch or numpy; the engine must handle numpy at least).
        wav = (np.linspace(-0.5, 0.5, 2400, dtype=np.float32))
        return [wav], 24000


def _install_fakes():
    torch = types.ModuleType("torch")
    torch.float32 = "float32"; torch.bfloat16 = "bfloat16"; torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch
    qt = types.ModuleType("qwen_tts")
    qt.Qwen3TTSModel = _FakeModel
    sys.modules["qwen_tts"] = qt


_install_fakes()
import tts_qwen3 as q3     # noqa: E402 — must follow the fake-module install


def _engine(default_voice="vinkona"):
    return q3.Qwen3TTSEngine(
        model_repo="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device="auto",
        default_voice=default_voice,
        voices={"vinkona": "A warm, calm assistant.",
                "narrator": "A slow storyteller with a faint British accent."})


def test_contract_and_voice_style():
    eng = _engine()
    check("voices are the configured names", set(eng.voices) == {"vinkona", "narrator"})
    check("a known voice resolves to its description",
          eng.resolve_style("narrator").startswith("A slow storyteller"))
    check("an unknown voice is used as a literal style",
          eng.resolve_style("A brisk newsreader.") == "A brisk newsreader.")
    check("no voice → the default's description",
          eng.resolve_style(None) == "A warm, calm assistant.")


def test_default_voice_falls_back_to_a_real_style():
    # "tara" (the Orpheus default) isn't a qwen3 style → must fall back, not be
    # sent as a literal instruct.
    eng = _engine(default_voice="tara")
    check("unknown default_voice falls back to a configured style",
          eng.default_voice in ("vinkona", "narrator"))


def test_synthesize_calls_api_and_returns_float32():
    eng = _engine()
    pcm = eng.synthesize("hello world", "narrator")
    call = _FakeModel.last
    check("generate_voice_design got the resolved instruct",
          call["instruct"].startswith("A slow storyteller"))
    check("…and the text + language", call["text"] == "hello world" and call["language"] == "English")
    check("returns float32 in [-1,1]",
          pcm.dtype == np.float32 and float(np.abs(pcm).max()) <= 1.0)
    check("adopts the model's returned sample rate", eng.sample_rate == 24000)
    check("sample count matches the returned wav", pcm.shape == (2400,))


def test_synthesize_stream_frames_pcm():
    eng = _engine()
    chunks = list(eng.synthesize_stream("hi", "vinkona"))
    total = sum(len(c) for c in chunks)
    check("stream yields 16-bit PCM bytes for the whole utterance", total == 2400 * 2)
    step = int(eng.sample_rate * eng._chunk_ms / 1000)
    check("chunks are sized by chunk_ms", len(chunks) == (2400 + step - 1) // step)
    check("every chunk is non-empty", all(len(c) > 0 for c in chunks))


def main():
    test_contract_and_voice_style()
    test_default_voice_falls_back_to_a_real_style()
    test_synthesize_calls_api_and_returns_float32()
    test_synthesize_stream_frames_pcm()
    print(f"\n{'ALL OK' if not check.failed else str(check.failed) + ' FAILED'}")
    raise SystemExit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
