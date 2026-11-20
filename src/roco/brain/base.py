"""Brain interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

SentenceCallback = Callable[[str], None]


class Brain(ABC):
    """Turns one user utterance into ROCO's reply.

    Contract for implementations:

    - Emotion: the model starts each reply with a tag like ``[happy]``. The brain
      strips it from the text and publishes it on the bus (``bus.set_emotion``)
      *before* the first sentence is delivered. The tag never reaches
      ``on_sentence`` or the returned string.
    - Streaming: the final answer is split into sentences as tokens arrive and
      each complete sentence is passed to ``on_sentence`` immediately, so TTS can
      start before generation finishes. Tool-call rounds produce no sentences.
    - State: the brain publishes ``State.THINKING`` when it starts. It does NOT
      publish ``SPEAKING``/``IDLE``; the speaker/voice loop owns those.
    - Errors: network/model failures must not raise to the caller. Deliver a short
      spoken-style apology through ``on_sentence`` and return it.
    - History: the user turn and the final assistant text (without tag) are appended
      to the history. An implementation may also keep, per turn, the emotion and a
      bounded trace of that turn's tool calls/results, and replay them to the model
      on later turns (small models otherwise stop tagging and start claiming actions
      they never called a tool for). Replayed results must be size-capped and marked
      with when they were fetched, so stale data is not mistaken for fresh.
    """

    @abstractmethod
    def respond(self, text: str, on_sentence: SentenceCallback | None = None) -> str:
        """Return the full reply text (tag stripped), calling ``on_sentence`` per sentence."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the conversation history (and its persisted copy)."""
