"""Speaker and Listener interfaces, plus null implementations for text mode and tests."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Speaker(ABC):
    """Speaks sentences in order on a background thread.

    Contract for implementations:

    - ``say`` only queues and returns immediately; sentences play in FIFO order.
    - While audio plays, publish ``State.SPEAKING`` once at the start and
      ``mouth`` amplitude (0..1) per playback block; publish ``mouth`` 0.0 and
      ``State.IDLE`` when the queue drains (unless something else changed state).
    - ``is_speaking`` is True from the first queued sentence until playback of
      the last one has finished. The listener uses it to mute the mic.
    """

    @abstractmethod
    def say(self, text: str) -> None:
        """Queue ``text`` for playback. Empty/whitespace text is ignored."""

    @abstractmethod
    def wait_done(self, timeout: float | None = None) -> bool:
        """Block until the queue is empty and playback finished. False on timeout."""

    @property
    @abstractmethod
    def is_speaking(self) -> bool: ...

    def stop(self) -> None:
        """Drop queued sentences and cut current playback (for barge-in later)."""

    def close(self) -> None:
        """Release audio resources. Safe to call twice."""


class Listener(ABC):
    """Hears Adam: wake word, then record until silence, then transcribe.

    Contract for implementations:

    - Audio is 16 kHz mono int16 in 80 ms frames (what openWakeWord expects).
    - While ``muted`` (ROCO speaking), captured frames are dropped and the wake
      word detector is reset on unmute, so ROCO never hears itself.
    - Publish ``State.LISTENING`` when recording starts and ``State.THINKING``
      once the transcript is ready (the brain will publish it again; harmless).
    """

    @abstractmethod
    def listen(self) -> str | None:
        """Block until the wake word fires, record the utterance and return its
        transcript. Returns None if nothing intelligible was said, or if
        :meth:`close` was called."""

    @abstractmethod
    def listen_followup(self, timeout: float) -> str | None:
        """Like :meth:`listen` but without the wake word: wait up to ``timeout``
        seconds for speech to start. None if Adam stays quiet."""

    @abstractmethod
    def set_muted(self, muted: bool) -> None: ...

    def close(self) -> None:
        """Stop capture and unblock any pending ``listen`` call."""


class NullSpeaker(Speaker):
    """Says nothing. Used by ``roco chat`` without ``--speak`` and in tests."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        if text and text.strip():
            self.spoken.append(text.strip())

    def wait_done(self, timeout: float | None = None) -> bool:
        return True

    @property
    def is_speaking(self) -> bool:
        return False
