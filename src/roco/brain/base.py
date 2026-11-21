"""Brain interface."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

SentenceCallback = Callable[[str], None]


@dataclass(frozen=True)
class TurnHooks:
    """Optional progress callbacks for :meth:`Brain.respond_turn` (the brain server uses
    them for latency stages and filler lines). They run on the brain's thread: keep them
    quick and never raise.

    - ``on_token()``: a piece of reply text arrived from the model (called per piece).
    - ``on_tools(names)``: the model asked for these tools; they are about to run.
    """

    on_token: Callable[[], None] | None = None
    on_tools: Callable[[list[str]], None] | None = None


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

    def respond_turn(
        self,
        text: str,
        on_sentence: SentenceCallback | None = None,
        *,
        cancel: threading.Event | None = None,
        hooks: TurnHooks | None = None,
    ) -> str:
        """:meth:`respond` that can be stopped (barge-in) and reports progress.

        Once ``cancel`` is set, the brain stops generating as soon as it can (within one
        streamed chunk), runs no further tools, delivers no more sentences and returns
        the text delivered so far. The turn is still stored in the history (with any
        tool calls already made), so :meth:`amend_last_reply` can then replace the reply
        with what the user actually heard.

        The default ignores ``hooks`` and only honours ``cancel`` between sentences, so
        every existing :class:`Brain` works with the brain server unchanged.
        """
        if cancel is None:
            return self.respond(text, on_sentence)

        def guarded(sentence: str) -> None:
            if not cancel.is_set() and on_sentence is not None:
                on_sentence(sentence)

        return self.respond(text, guarded)

    def amend_last_reply(self, heard: str) -> None:
        """Replace the last stored reply with ``heard`` (barge-in: only what was played).

        Default: no-op, for brains without a history.
        """

    @abstractmethod
    def reset(self) -> None:
        """Forget the conversation history (and its persisted copy)."""
