"""
Orpheus TTS engine — expressive speech with inline emotion tags.

Unlike NeuTTS (clone-from-clip, plain prosody), Orpheus performs inline tags in
the text — <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>,
<gasp> — for human texture.  It runs on a Llama-3B backbone via vLLM, so it lives
in its OWN venv (orpheus_env); its deps conflict with neutts_env and
vinkona_env.  Output is 16-bit PCM @ 24 kHz mono, matching the client pipeline.

Same synthesize() contract as NeuTTSEngine so the cascade can switch engines by a
flag.  "voice" here is a preset name rather than a cloned clip.

Usage:
    eng = OrpheusEngine()                       # raises if orpheus-speech missing
    pcm = eng.synthesize("Well <laugh> hello there.", voice="tara")  # float32 @ 24 kHz

synthesize() is blocking; call it from a worker thread, not the event loop.
"""

import asyncio
import queue as _queue
import threading
import typing as tp

import numpy as np

SAMPLE_RATE = 24000
# Preset voices (the installed finetune-prod model reports these; "tara" also
# works as a prompt prefix and is our default).
PRESET_VOICES = ["tara", "leah", "julia", "jess", "leo", "mia", "zac", "zoe"]


class OrpheusEngine:
    def __init__(
        self,
        model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod",
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.4,
        enforce_eager: bool = False,
        default_voice: str = "tara",
        max_tokens: tp.Optional[int] = None,
        dtype: tp.Any = None,
    ):
        # We subclass OrpheusModel for two reasons:
        #
        # 1. Memory: it hardcodes AsyncEngineArgs(model, dtype), so vLLM defaults to
        #    0.9 GPU util and the model's full 131072-token context — gigabytes of
        #    KV cache it never needs.  We synthesize one sentence at a time, so we
        #    cap max_model_len + gpu_memory_utilization (fits beside the fast LM).
        #
        # 2. Event loop: its generate_tokens_sync runs asyncio.run() in a fresh
        #    thread PER request, which creates and then CLOSES an event loop each
        #    call.  vLLM's AsyncLLMEngine binds its output handler to the first
        #    loop it runs on; once that loop closes (after request #1), request #2
        #    hits a dead engine → EngineDeadError.  We override it to drive the
        #    engine from ONE persistent loop for the server's whole life.
        from orpheus_tts import OrpheusModel  # raises ImportError if not installed
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        import torch

        _dtype = dtype if dtype is not None else torch.bfloat16
        # Per-call audio-token budget.  The old hardcoded 1200 (~13–14 s) truncated long
        # sentences mid-word; default to nearly the whole context (leaving room for the
        # text prompt) so a sentence isn't cut short.  The cascade also chunks long
        # sentences, so this is headroom, not the primary guard.
        _gen_max_tokens = int(max_tokens) if max_tokens else max(1024, max_model_len - 256)

        # Persistent event loop in a daemon thread — all engine.generate() calls
        # run here, so the engine's async machinery never gets torn down.
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        loop = self._loop

        class _TunedOrpheusModel(OrpheusModel):
            def _setup_engine(self_inner):
                engine_args = AsyncEngineArgs(
                    model=self_inner.model_name,
                    dtype=self_inner.dtype,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                )
                return AsyncLLMEngine.from_engine_args(engine_args)

            def generate_tokens_sync(self_inner, prompt, voice=None, request_id="req-001",
                                     temperature=0.6, top_p=0.8, max_tokens=None,
                                     stop_token_ids=(49158,), repetition_penalty=1.3):
                prompt_string = self_inner._format_prompt(prompt, voice)
                sp = SamplingParams(
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens or _gen_max_tokens,
                    stop_token_ids=list(stop_token_ids), repetition_penalty=repetition_penalty,
                )
                tq: _queue.Queue = _queue.Queue()

                async def producer():
                    try:
                        async for result in self_inner.engine.generate(
                            prompt=prompt_string, sampling_params=sp, request_id=request_id):
                            tq.put(result.outputs[0].text)
                    finally:
                        tq.put(None)

                fut = asyncio.run_coroutine_threadsafe(producer(), loop)
                while True:
                    tok = tq.get()
                    if tok is None:
                        break
                    yield tok
                fut.result()  # surface any exception from the producer

        self._model = _TunedOrpheusModel(model_name, dtype=_dtype)
        self.sample_rate = SAMPLE_RATE
        self.default_voice = default_voice
        self._req = 0

    @property
    def voices(self) -> list[str]:
        return list(PRESET_VOICES)

    def synthesize(self, text: str, voice: tp.Optional[str] = None) -> np.ndarray:
        """
        Synthesize one utterance (inline <tag>s allowed).  Returns float32 PCM at
        self.sample_rate.  BLOCKING — run in a worker thread, not the event loop.
        """
        chunks = [np.frombuffer(c, dtype=np.int16) for c in self._raw_stream(text, voice)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        pcm16 = np.concatenate(chunks)
        return np.ascontiguousarray(pcm16.astype(np.float32) / 32768.0)

    def synthesize_stream(self, text: str, voice: tp.Optional[str] = None):
        """Yield 16-bit PCM byte chunks as Orpheus produces them (low first-audio
        latency).  Same int16 @ 24 kHz framing as NeuTTSEngine.synthesize_stream."""
        yield from self._raw_stream(text, voice)

    def _raw_stream(self, text: str, voice: tp.Optional[str]):
        voice = voice or self.default_voice
        self._req += 1
        yield from self._model.generate_speech(
            prompt=text, voice=voice, request_id=f"req-{self._req}")
