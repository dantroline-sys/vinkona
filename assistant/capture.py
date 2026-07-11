"""
capture.py — durable, append-only capture of the 9B's orchestration traces, for the
(future) operational skill-LoRA loop (see vinkona_skill_lora_spec.md).

This is NOT the config UI's trace feed (a small ring buffer that truncates).  It's the
training-grade corpus: one record per orchestration turn, kept.

PRE-FREEZE it is for ANALYSIS ONLY.  The prompt-assembly format is still moving (live
guidance, working memory, the time-sense blocks…), so a LoRA trained on today's traces
would learn a layout we stop serving — the spec's §1 "quiet rot".  So every record is
stamped with `format_version` and the `base_model`: once the assembly contract is frozen,
curation selects the byte-compatible slice and discards the rest.  Capturing now (cheap,
rides on the turn we already run) banks the corpus so the day you freeze, training-grade
collection has already been happening.

Schema follows the spec's §3 (input_context / model_action / outcome).  Outcome fields
that need a later moment — teacher_trace, user_response, reconciliation_verdict — are left
absent here and filled by nightly curation downstream.

Best-effort throughout: a filesystem hiccup must never break a live turn, so every write
swallows OSError.  Append-only JSONL, rotated by day.
"""

import datetime
import json
import time
import typing as tp
import uuid
from pathlib import Path


class TraceCapture:
    def __init__(self, directory, *, format_version: str = "v0-unfrozen",
                 base_model: str = "", enabled: bool = True):
        self.dir = Path(directory)
        self.format_version = format_version
        self.base_model = base_model
        self.enabled = bool(enabled)

    def _path(self) -> Path:
        return self.dir / f"traces-{datetime.date.today().isoformat()}.jsonl"

    def record(self, *, input_context: dict, model_action: dict,
               outcome: tp.Optional[dict] = None,
               trace_id: tp.Optional[str] = None) -> tp.Optional[str]:
        """Append one orchestration-turn record; return its trace_id (or None if disabled
        or the write failed).  Never raises."""
        if not self.enabled:
            return None
        tid = trace_id or uuid.uuid4().hex
        rec = {
            "trace_id": tid,
            "ts": time.time(),
            "format_version": self.format_version,
            "base_model": self.base_model,
            "input_context": input_context,
            "model_action": model_action,
            "outcome": outcome or {},
        }
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self._path().open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            return None
        return tid
