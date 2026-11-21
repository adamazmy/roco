"""Per-turn latency: stages in ms from end of speech, one log line + one JSON line per turn.

Brain stages (its own monotonic clock, from ``turn_end`` arrival minus ``trailing_ms``):
``stt_done``, ``first_token``, ``first_sentence``, ``first_tts_done``,
``first_audio_sent``. Body stages come later in ``turn_metrics`` (``first_audio_played``,
``vad_trailing``, ...) and are merged in. A turn is written when its body metrics
arrive, or ``wait_body_s`` after ``turn_done`` without them (old body, interrupted).

Targets (docs/ROCO_SPEC.md §11): end of speech -> first spoken word < 1000 ms
(stretch), < 2500 ms hard limit.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BRAIN_STAGES = ("stt_done", "first_token", "first_sentence", "first_tts_done",
                "first_audio_sent")
TARGET_MS = 1000
HARD_LIMIT_MS = 2500


class TurnLatency:
    """Marks for one turn. ``mark`` keeps the first time each stage is reached."""

    def __init__(self, turn: int, t0: float, clock: Callable[[], float] = time.monotonic,
                 kind: str = "voice") -> None:
        self.turn = turn
        self.t0 = t0
        self.kind = kind
        self._clock = clock
        self._marks: dict[str, int] = {}
        self._lock = threading.Lock()
        self.info: dict[str, Any] = {}

    def mark(self, stage: str, at: float | None = None) -> None:
        at = self._clock() if at is None else at
        with self._lock:
            if stage not in self._marks:
                self._marks[stage] = max(0, round((at - self.t0) * 1000))

    def stages(self) -> dict[str, int]:
        with self._lock:
            return dict(self._marks)


class LatencyLog:
    def __init__(self, path: Path | None, wait_body_s: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.path = Path(path) if path else None
        self.wait_body_s = wait_body_s
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[TurnLatency, float]] = {}
        self.written: list[dict[str, Any]] = []  # last records, for tests and diagnostics

    def turn_done(self, latency: TurnLatency, **info: Any) -> None:
        latency.info.update(info)
        with self._lock:
            self._pending[latency.turn] = (latency, self._clock())

    def body_metrics(self, turn: int, stages: dict[str, Any]) -> bool:
        with self._lock:
            item = self._pending.pop(turn, None)
        if item is None:
            log.debug("turn_metrics for unknown/finished turn %d: %s", turn, stages)
            return False
        self._write(item[0], stages)
        return True

    def flush(self, force: bool = False) -> None:
        """Write turns whose body metrics are overdue (all of them with ``force``)."""
        now = self._clock()
        with self._lock:
            due = [t for t, (_, at) in self._pending.items()
                   if force or now - at >= self.wait_body_s]
            items = [self._pending.pop(t)[0] for t in due]
        for item in items:
            self._write(item, {})

    def _write(self, latency: TurnLatency, body: dict[str, Any]) -> None:
        stages = latency.stages()
        body = {k: int(v) for k, v in (body or {}).items() if isinstance(v, int | float)}
        parts = [f"{k}={stages[k]}" for k in BRAIN_STAGES if k in stages]
        parts += [f"{k}={v}" for k, v in stages.items() if k not in BRAIN_STAGES]
        parts += [f"body.{k}={v}" for k, v in body.items()]
        spoken = body.get("first_audio_played", stages.get("first_audio_sent"))
        verdict = ""
        if spoken is not None and latency.kind == "voice":
            verdict = ("ok" if spoken < TARGET_MS
                       else "slow" if spoken < HARD_LIMIT_MS else "TOO SLOW")
        info = latency.info
        flags = " interrupted" if info.get("interrupted") else ""
        log.info("latency turn=%d (%s, ms from end of speech) %s%s%s", latency.turn,
                 latency.kind, " ".join(parts) or "-", f" [{verdict}]" if verdict else "",
                 flags)
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "turn": latency.turn,
            "kind": latency.kind,
            "stages": stages,
            "body": body,
            **info,
        }
        self.written = [*self.written[-49:], record]
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Could not append to %s: %s", self.path, exc)
