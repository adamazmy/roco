"""The voice conversation loop used by ``roco run``.

    wake word -> record -> STT -> brain.respond(text, on_sentence=speaker.say)
              -> wait until spoken -> follow-up window (no wake word) -> ...

Announcements (reminders) arriving on the bus are spoken at once while ROCO is
idle, and held back during a conversation until the current reply has been spoken,
so they never cut into a sentence or into Adam's own speech.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from roco.brain.base import Brain
from roco.bus import EventBus, State, Topic
from roco.voice.base import Listener, Speaker

log = logging.getLogger(__name__)

APOLOGY = "Sorry, something went wrong on my side. Could you try that again?"


class AnnouncementQueue:
    """Speaks ``announce`` events now when idle, or defers them during a turn."""

    def __init__(self, speaker: Speaker) -> None:
        self.speaker = speaker
        self._lock = threading.Lock()
        self._busy = False
        self._deferred: deque[str] = deque()

    def on_announce(self, text: Any) -> None:  # bus handler: must stay quick
        text = str(text or "").strip()
        if not text:
            return
        with self._lock:
            if self._busy:
                log.info("Deferring announcement until the turn ends: %r", text)
                self._deferred.append(text)
                return
        log.info("Announcing: %r", text)
        self.speaker.say(text)

    def begin(self) -> None:
        with self._lock:
            self._busy = True

    def flush(self) -> int:
        """Speak deferred announcements now (does not change busy). Returns how many."""
        with self._lock:
            pending = list(self._deferred)
            self._deferred.clear()
        for text in pending:
            log.info("Announcing (deferred): %r", text)
            self.speaker.say(text)
        return len(pending)

    def end(self) -> int:
        with self._lock:
            self._busy = False
        return self.flush()

    @property
    def busy(self) -> bool:
        return self._busy


class LatencyTracker:
    """Logs end-of-speech -> stt_done / first_sentence / first_audio deltas (ms)."""

    def __init__(self) -> None:
        self.speech_end: float | None = None
        self.marks: dict[str, float] = {}

    def start(self, speech_end: float | None) -> None:
        self.speech_end = speech_end
        self.marks = {}

    def mark(self, name: str, at: float | None = None) -> None:
        if self.speech_end is None or name in self.marks:
            return
        self.marks[name] = (at if at is not None else time.monotonic())
        log.info("latency %s=+%d ms", name, (self.marks[name] - self.speech_end) * 1000)

    def summary(self) -> dict[str, int]:
        if self.speech_end is None:
            return {}
        return {k: int((v - self.speech_end) * 1000) for k, v in self.marks.items()}


class VoiceLoop:
    def __init__(
        self,
        listener: Listener,
        speaker: Speaker,
        brain: Brain,
        bus: EventBus,
        followup_s: float = 5.0,
        speak_timeout_s: float = 120.0,
    ) -> None:
        self.listener = listener
        self.speaker = speaker
        self.brain = brain
        self.bus = bus
        self.followup_s = followup_s
        self.speak_timeout_s = speak_timeout_s
        self.announcements = AnnouncementQueue(speaker)
        self.latency = LatencyTracker()
        self.turns = 0
        hook = getattr(speaker, "on_audio_start", None)
        if callable(hook):
            hook(lambda t: self.latency.mark("first_audio", t))

    # ------------------------------------------------------------------ running
    def run(self, stop: threading.Event) -> None:
        unsubscribe = self.bus.subscribe(Topic.ANNOUNCE, self.announcements.on_announce)
        watcher = threading.Thread(target=self._close_on_stop, args=(stop,),
                                   name="roco-voice-stop", daemon=True)
        watcher.start()
        self.bus.set_state(State.IDLE)
        log.info("Ready: say the wake word.")
        try:
            while not stop.is_set():
                text = self.listener.listen()
                if stop.is_set():
                    break
                if text is None:
                    continue
                self.conversation(text, stop)
        finally:
            unsubscribe()
            stop.set()

    def _close_on_stop(self, stop: threading.Event) -> None:
        stop.wait()
        self.speaker.stop()
        self.listener.close()

    def conversation(self, text: str, stop: threading.Event | None = None) -> None:
        """One wake-word-initiated exchange, including any follow-up turns."""
        self.announcements.begin()
        try:
            while text and not (stop and stop.is_set()):
                self.turn(text)
                self.announcements.flush()  # a reminder that came in during the reply
                self._wait_spoken()
                if stop and stop.is_set():
                    break
                text = self.listener.listen_followup(self.followup_s)
        finally:
            self.announcements.end()

    def turn(self, text: str) -> str:
        """Answer one utterance and wait until it has been spoken. Mic muted meanwhile."""
        self.turns += 1
        timing = getattr(self.listener, "last_timing", None)
        self.latency.start(timing.speech_end if timing else time.monotonic())
        if timing:
            self.latency.mark("stt_done", timing.stt_done)
        log.info("You: %s", text)

        def on_sentence(sentence: str) -> None:
            self.latency.mark("first_sentence")
            self.speaker.say(sentence)

        self.listener.set_muted(True)
        hold = getattr(self.speaker, "hold", None)
        reply = ""
        try:
            if callable(hold):
                with hold():
                    reply = self._respond(text, on_sentence)
            else:
                reply = self._respond(text, on_sentence)
            self._wait_spoken()
        finally:
            self.listener.set_muted(False)
        log.info("ROCO: %s", reply)
        if self.latency.marks:
            log.info("latency summary (ms from end of speech): %s", self.latency.summary())
        return reply

    def _respond(self, text: str, on_sentence: Any) -> str:
        try:
            return self.brain.respond(text, on_sentence=on_sentence)
        except Exception:  # the brain should not raise, but never kill the loop
            log.exception("Brain failed")
            on_sentence(APOLOGY)
            return APOLOGY

    def _wait_spoken(self) -> None:
        if not self.speaker.wait_done(self.speak_timeout_s):
            log.warning("Speech did not finish within %.0fs; cutting it", self.speak_timeout_s)
            self.speaker.stop()
