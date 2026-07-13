"""
CPU speech-to-text for the user stream (faster-whisper / CTranslate2).

We run faster-whisper on the CPU over the VAD-segmented turns the RNNoise
front-end already produces — one transcription per turn, off the asyncio
event-loop thread, so the GPU stays free for the LMs and TTS and the audio
stream never stalls.

The resulting text drives two things: the 'You: …' dialogue line and the
user_turn_queue that feeds the two-tier LLM bridge.

Usage:
    asr = WhisperASR(model="base.en")        # raises if faster-whisper missing
    text = asr.transcribe(clip_f32, opts)     # clip at in_rate, returns a string

`opts` (from config "asr") controls noise rejection: Silero VAD on the clip
(vad_filter, the bundled Silero model), decode-confidence thresholds, and a
post-filter that drops hallucinated/low-confidence segments — so fan, keyboard
and clothing noise that slips past the live gate transcribes to "" instead of a
phantom user turn.

transcribe() is blocking (CPU inference) and is meant to be called via
loop.run_in_executor(), never directly on the event loop.
"""

import numpy as np

_ASR_RATE = 16000  # Whisper operates at 16 kHz.

try:
    import soxr
except ImportError as e:  # pragma: no cover - environment dependent
    soxr = None
    _SOXR_IMPORT_ERROR = e
else:
    _SOXR_IMPORT_ERROR = None


class WhisperASR:
    def __init__(self, model: str = "base.en", in_rate: int = 24000,
                 device: str = "cpu", compute_type: str = "int8"):
        if soxr is None:
            raise RuntimeError(
                f"soxr is required for resampling but failed to import: {_SOXR_IMPORT_ERROR}. "
                "Install with: pip install soxr"
            )
        from faster_whisper import WhisperModel  # raises ImportError if not installed
        self.in_rate = in_rate
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, clip: np.ndarray, opts: dict | None = None,
                   prompt: str | None = None) -> tuple[str, float | None]:
        """
        Transcribe one user turn.  clip: float32 [-1, 1] at self.in_rate.
        `opts`: the config "asr" dict (decode + VAD knobs); falls back to defaults.
        `prompt`: an optional initial_prompt that biases the decoder toward known
        vocabulary (e.g. the names in the people store) — the single best fix for the
        proper-noun mishearings that poison memory.
        Returns (text, confidence): the joined transcript (empty for silence/noise) and a
        length-weighted mean of the kept segments' avg_logprob (closer to 0 = surer; None
        if nothing was kept), so the caller can ask the user to repeat when Whisper was
        genuinely unsure rather than feed it a garbled turn.
        BLOCKING — call from a worker thread, not the event loop.
        """
        o = opts or {}
        clip = np.ascontiguousarray(clip, dtype=np.float32)
        if self.in_rate != _ASR_RATE:
            clip = soxr.resample(clip, self.in_rate, _ASR_RATE).astype(np.float32)

        vad_filter = o.get("vad_filter", True)
        vad_params = ({"threshold": o.get("vad_threshold", 0.5),
                       "min_speech_duration_ms": o.get("min_speech_ms", 250),
                       "min_silence_duration_ms": o.get("min_silence_ms", 100)}
                      if vad_filter else None)
        no_speech_thr = o.get("no_speech_threshold", 0.6)
        logprob_thr = o.get("log_prob_threshold", -1.0)

        segments, _info = self.model.transcribe(
            clip,
            beam_size=o.get("beam_size", 1),
            condition_on_previous_text=o.get("condition_on_previous_text", False),
            initial_prompt=(prompt or None),  # bias toward known names/vocab
            no_speech_threshold=no_speech_thr,
            log_prob_threshold=logprob_thr,
            compression_ratio_threshold=o.get("compression_ratio_threshold", 2.4),
            vad_filter=vad_filter,          # Silero VAD on the clip (bundled)
            vad_parameters=vad_params,
        )
        # Post-filter: drop segments Whisper itself is unsure are speech (the same
        # no-speech/logprob gate it uses internally, applied explicitly here so a
        # low-confidence hallucination on noise becomes "" rather than a fake turn).
        # Of the kept segments, track a length-weighted mean avg_logprob as a confidence.
        kept: list[str] = []
        conf_num = conf_den = 0.0
        for seg in segments:
            if seg.no_speech_prob >= no_speech_thr and seg.avg_logprob < logprob_thr:
                continue
            text = seg.text.strip()
            if text:
                kept.append(text)
                w = max(1, len(text))
                conf_num += seg.avg_logprob * w
                conf_den += w
        confidence = (conf_num / conf_den) if conf_den else None
        return " ".join(kept).strip(), confidence


def should_clarify(text: str, confidence: float | None, opts: dict,
                   just_clarified: bool) -> bool:
    """Decide whether a transcribed turn is too unsure to act on — in which case the caller
    should ask the user to repeat rather than feed likely-garbled text to the LM (and into
    memory).  Pure (no model state) so it's easy to test.

    Fires when: clarification is enabled (opts['clarify_below'] is set), Whisper's
    confidence is below it, the turn is non-trivial (>= clarify_min_words), and we didn't
    JUST ask (so a bad mic can't trap the user in a clarify loop)."""
    thr = opts.get("clarify_below")
    if thr is None or confidence is None or just_clarified:
        return False
    if len((text or "").split()) < opts.get("clarify_min_words", 2):
        return False
    return confidence < thr
