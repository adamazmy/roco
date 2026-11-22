"""The fallback chain the body client uses while the PC brain is unreachable.

Spec §9: PC brain -> Claude API direct from the body (whisper tiny.en on the body) ->
"my brain's offline" canned line. The PC brain itself is the body client's WebSocket
link; this object is everything after it. Contract agreed with workstream B:

    fb = create_fallback(cfg)                     # light, never raises
    fb.name                                       # "claude:<model>" or "canned"
    fb.offline_line                               # spoken by the body's local TTS
    fb.offline_wav                                # same line pre-rendered (Path or None)
    fb.respond_audio(pcm, rate, on_sentence) -> str
    fb.respond_text(text, on_sentence) -> str
    fb.cancel()                                   # barge-in; safe from another thread
    fb.brain_restored_line() -> str | None        # when the PC link is back
    fb.last_emotion                               # set before the first sentence
    fb.close()

Sentences passed to ``on_sentence`` are clean spoken text: no emotion tag, no markdown.
The first reply after a switch starts with a sentence saying ROCO is on the cloud
brain (spec §10: says out loud when it uses the cloud).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from roco.brain.emotion import EmotionTagParser
from roco.brain.sentences import SentenceSplitter
from roco.bus import Emotion
from roco.config import Config
from roco.fallback.claude import (
    ClaudeClient,
    CloudError,
    CloudPermanentError,
)
from roco.fallback.keys import load_api_key
from roco.fallback.persona import fallback_system_prompt
from roco.fallback.stt import LazyStt

log = logging.getLogger(__name__)

SentenceCallback = Callable[[str], None]

OFFLINE_LINE = "Sorry, my brain's offline right now. Give me a minute to sort myself out."
CLOUD_SWITCH_LINE = (
    "Quick heads up: my main brain's offline, so I'm borrowing one in the cloud for now."
)
CLOUD_LOST_LINE = "Hang on, I've lost my backup brain as well."
BRAIN_RESTORED_LINE = "And I'm back on my own brain. Much better."
DIDNT_CATCH_LINE = "Sorry, I didn't catch that."


class FallbackChain:
    """Claude (if a key is configured and it keeps working), else the canned line."""

    def __init__(
        self,
        claude: ClaudeClient | None,
        stt: LazyStt | None = None,
        *,
        assistant_name: str = "ROCO",
        user_name: str = "Adam",
        history_turns: int = 6,
        now: Callable[[], datetime] | None = None,
        describe_now: Callable[[datetime], str] | None = None,
        offline_wav: Path | None = None,
    ) -> None:
        self._claude = claude
        self._stt = stt or LazyStt()
        self.assistant_name = assistant_name
        self.user_name = user_name
        self.history_turns = max(0, history_turns)
        self._now = now or datetime.now
        self._describe_now = describe_now
        self._history: list[dict[str, Any]] = []
        self._cancel = threading.Event()
        self._announced = False  # said "I'm on the cloud brain" since the last switch
        self._spoke = False  # the fallback answered something since the last restore
        self._claude_ok = claude is not None
        self.offline_line = OFFLINE_LINE
        # Pre-rendered WAV of offline_line in the body's voice (roco fallback-prepare), for
        # the body to play when its own TTS fails; None if not rendered for this voice.
        self.offline_wav = offline_wav
        self.last_emotion: Emotion | None = None

    # ------------------------------------------------------------------ state
    @property
    def name(self) -> str:
        if self._claude is not None and self._claude_ok:
            return f"claude:{self._claude.model}"
        return "canned"

    @property
    def uses_cloud(self) -> bool:
        return self.name != "canned"

    def cancel(self) -> None:
        """Stop the reply in progress: no further ``on_sentence`` calls for it."""
        self._cancel.set()

    def brain_restored_line(self) -> str | None:
        """The PC brain is back: the line to say if the fallback spoke since the switch."""
        spoke = self._spoke
        self._spoke = False
        self._announced = False
        self._history.clear()
        if self._claude is not None:
            self._claude_ok = True  # give Claude another chance next time
        return BRAIN_RESTORED_LINE if spoke else None

    def reset(self) -> None:
        self._history.clear()

    def close(self) -> None:
        self.cancel()

    # ------------------------------------------------------------------ turns
    def respond_audio(self, pcm: bytes, rate: int, on_sentence: SentenceCallback) -> str:
        """Transcribe ``pcm`` (PCM16 mono LE at ``rate``) on the body, then answer it."""
        self._cancel.clear()
        if not self.uses_cloud:
            return self._say(on_sentence, self.offline_line)
        try:
            text = self._stt.transcribe(pcm, rate)
        except Exception as exc:  # missing faster-whisper, model files, ...
            log.warning("Fallback speech-to-text failed: %s", exc)
            return self._say(on_sentence, self.offline_line)
        log.info("Fallback heard: %r", text)
        if not text:
            return ""
        return self.respond_text(text, on_sentence)

    def respond_text(self, text: str, on_sentence: SentenceCallback) -> str:
        """Answer one utterance through the best brain left. Never raises."""
        self._cancel.clear()
        text = (text or "").strip()
        if not text:
            return ""
        if not self.uses_cloud:
            return self._say(on_sentence, self.offline_line)

        spoken: list[str] = []

        def emit(sentence: str) -> None:
            if not self._cancel.is_set() and sentence:
                spoken.append(sentence)
                on_sentence(sentence)

        if not self._announced:
            self._announced = True
            emit(CLOUD_SWITCH_LINE)
        prefix = len(spoken)

        parser = EmotionTagParser()
        splitter = SentenceSplitter()
        self.last_emotion = None

        def on_text(chunk: str) -> None:
            out = parser.feed(chunk)
            if parser.decided and self.last_emotion is None:
                self.last_emotion = parser.emotion or Emotion.NEUTRAL
            for sentence in splitter.feed(out):
                emit(sentence)

        messages = self._history + [{"role": "user", "content": text}]
        assert self._claude is not None
        try:
            reply = self._claude.stream_reply(self._system_prompt(), messages, on_text,
                                              cancel=self._cancel)
        except CloudPermanentError as exc:
            log.error("Claude fallback disabled: %s", exc)
            self._claude_ok = False
            return self._finish_failed(spoken, prefix, emit)
        except CloudError as exc:
            log.warning("Claude fallback unavailable this turn: %s", exc)
            return self._finish_failed(spoken, prefix, emit)

        tail = parser.flush()
        if self.last_emotion is None:
            self.last_emotion = parser.emotion or Emotion.NEUTRAL
        for sentence in splitter.feed(tail):
            emit(sentence)
        if (rest := splitter.flush()) is not None:
            emit(rest)
        if not reply.cancelled and len(spoken) == prefix:
            emit(DIDNT_CATCH_LINE)  # e.g. a refusal with no text

        answer = " ".join(spoken[prefix:])
        self._remember(text, answer)
        self._spoke = True
        return " ".join(spoken)

    # ------------------------------------------------------------------ helpers
    def _finish_failed(self, spoken: list[str], prefix: int,
                       emit: SentenceCallback) -> str:
        emit(CLOUD_LOST_LINE if len(spoken) > prefix else self.offline_line)
        self._spoke = True
        return " ".join(spoken)

    def _say(self, on_sentence: SentenceCallback, line: str) -> str:
        self.last_emotion = Emotion.SAD
        on_sentence(line)
        self._spoke = True
        return line

    def _remember(self, user_text: str, answer: str) -> None:
        if not answer or self.history_turns == 0:
            return
        self._history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
        self._history = self._history[-2 * self.history_turns :]

    def _system_prompt(self) -> str:
        now = self._now()
        describe = self._describe_now(now) if self._describe_now else None
        return fallback_system_prompt(now, name=self.assistant_name, user=self.user_name,
                                      describe=describe)


def create_fallback(cfg: Config, *, client: Any = None) -> FallbackChain:
    """Build the chain from the config. Light (no model or SDK import) and never raises.

    ``client`` injects a stand-in for ``anthropic.Anthropic`` (tests).
    """
    from roco.fallback.offline_audio import find_offline_wav
    from roco.tools.clock import describe_now, local_timezone

    tz = local_timezone(cfg)
    claude: ClaudeClient | None = None
    if cfg.get_path("cloud.enabled", True):
        key = load_api_key(cfg)
        if key or client is not None:
            claude = ClaudeClient.from_config(cfg, key, client=client)
        else:
            log.info("No Anthropic API key: the fallback will only say the offline line")
    return FallbackChain(
        claude,
        LazyStt.from_config(cfg),
        assistant_name=str(cfg.get_path("assistant.name", "ROCO")),
        user_name=str(cfg.get_path("assistant.user_name", "Adam")),
        history_turns=int(cfg.get_path("cloud.history_turns", 6)),
        now=lambda: datetime.now(tz),
        describe_now=describe_now,
        offline_wav=find_offline_wav(cfg, OFFLINE_LINE),
    )
