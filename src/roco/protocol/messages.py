"""ROCO link protocol v1: the messages between the body client and the brain server.

Two kinds of WebSocket frame travel on one connection:

- **Control messages**: JSON text frames ``{"v": 1, "type": "...", ...fields}``.
  Each type is a frozen dataclass below; :func:`encode` / :func:`decode` convert.
- **Audio**: binary frames, see :mod:`roco.protocol.audio`.

Versioning: ``v`` is the protocol major version. A peer refuses a ``hello`` whose
``protocol`` differs from :data:`PROTOCOL_VERSION` (see :class:`Error`). Within a
version, new *optional* fields and new message types may be added: :func:`decode`
ignores unknown fields and returns :class:`Unknown` for unknown types, which
receivers log and skip. Removing or renaming a field needs a new major version.

Turn ids: the body numbers user turns 1, 2, 3, ... (per connection). Speech the brain
starts on its own (reminders, announcements) uses ids from :func:`brain_turn_id`,
which have the top bit set, so the two never collide. Every message and audio frame
that belongs to a turn carries its id; anything for a turn the receiver has already
cancelled (barge-in) is dropped.

Timing: clocks on the Pi and the PC are not synchronised, so no message carries an
absolute timestamp that the other side must compare with its own clock. Each side
measures durations on its own monotonic clock and reports them in milliseconds
(``stages`` in :class:`TurnDone` / :class:`TurnMetrics`). ``ping``/``pong`` echo the
sender's own timestamp back so the sender can compute the round trip.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
BRAIN_TURN_BIT = 0x8000_0000


class ProtocolError(ValueError):
    """A frame that is not valid protocol (bad JSON, missing field, wrong version)."""


def brain_turn_id(n: int) -> int:
    """Turn id for speech the brain starts itself (announcements). ``n`` >= 1."""
    return BRAIN_TURN_BIT | (n & 0x7FFF_FFFF)


def is_brain_turn(turn: int) -> bool:
    return bool(turn & BRAIN_TURN_BIT)


# --------------------------------------------------------------------------- base
_REGISTRY: dict[str, type[Message]] = {}


@dataclass(frozen=True)
class Message:
    """Base class. Subclasses set ``TYPE`` and declare their fields."""

    TYPE: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.TYPE:
            if cls.TYPE in _REGISTRY:
                raise TypeError(f"duplicate message type {cls.TYPE!r}")
            _REGISTRY[cls.TYPE] = cls

    def to_dict(self) -> dict[str, Any]:
        return {"v": PROTOCOL_VERSION, "type": self.TYPE, **dataclasses.asdict(self)}


@dataclass(frozen=True)
class Unknown(Message):
    """A message type this version doesn't know. Receivers log it and move on."""

    type: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - never sent
        return {"v": PROTOCOL_VERSION, "type": self.type, **self.data}


# --------------------------------------------------------------------------- both ways
@dataclass(frozen=True)
class Ping(Message):
    """Either side. The peer answers with :class:`Pong` echoing ``id`` and ``ts``."""

    TYPE: ClassVar[str] = "ping"
    id: int
    ts: float  # sender's monotonic clock, seconds; only the sender interprets it


@dataclass(frozen=True)
class Pong(Message):
    TYPE: ClassVar[str] = "pong"
    id: int
    ts: float  # copied from the ping


@dataclass(frozen=True)
class Error(Message):
    """Something went wrong. ``fatal`` errors are followed by the sender closing.

    Codes: ``version`` (protocol mismatch), ``auth`` (bad token), ``busy`` (another
    body is connected), ``bad_message``, ``internal``.
    """

    TYPE: ClassVar[str] = "error"
    code: str
    message: str = ""
    fatal: bool = False


@dataclass(frozen=True)
class Settings(Message):
    """Either side. Partial settings (e.g. ``{"personality": {"roast": 80}}``,
    ``{"muted": true}``). The brain owns the truth: it applies body requests and
    broadcasts the result back as a ``settings`` message."""

    TYPE: ClassVar[str] = "settings"
    values: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- body -> brain
@dataclass(frozen=True)
class Hello(Message):
    """First message from the body. The brain answers :class:`Welcome` or a fatal
    :class:`Error`. ``caps`` lists what the body can do, e.g.
    ``{"face": true, "servos": "simulated", "audio_rates": [22050, 24000]}``."""

    TYPE: ClassVar[str] = "hello"
    protocol: int
    body_id: str
    token: str
    mic_rate: int = 16000
    caps: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnStart(Message):
    """The body started recording a user turn. Mic audio for ``turn`` follows.

    ``trigger``: ``wake`` (wake word), ``followup`` (follow-up window), ``barge_in``
    (Adam spoke over ROCO; an :class:`Interrupt` for the old turn precedes it).
    """

    TYPE: ClassVar[str] = "turn_start"
    turn: int
    trigger: str = "wake"


@dataclass(frozen=True)
class TurnEnd(Message):
    """The body's VAD closed the turn: no more mic audio for ``turn``.

    ``reason``: ``silence`` (end of speech), ``max_length``, ``nothing`` (no speech
    at all; the brain discards the turn). ``trailing_ms`` is how much silence the
    body waited after the last speech, so the brain can report true
    end-of-speech latency.
    """

    TYPE: ClassVar[str] = "turn_end"
    turn: int
    reason: str = "silence"
    trailing_ms: int = 0


@dataclass(frozen=True)
class TextInput(Message):
    """A typed utterance instead of audio (tests, ``roco body --text``)."""

    TYPE: ClassVar[str] = "text_input"
    turn: int
    text: str


@dataclass(frozen=True)
class Interrupt(Message):
    """Stop ``turn`` now: the body has already cut playback. The brain cancels
    generation/TTS for it and drops anything still queued. ``reason``: ``barge_in``,
    ``user`` (e.g. "stop"), ``shutdown``."""

    TYPE: ClassVar[str] = "interrupt"
    turn: int
    reason: str = "barge_in"


@dataclass(frozen=True)
class PlaybackDone(Message):
    """The body finished (or cut) playing ``turn``. ``heard`` is the text of the
    sentences that actually played, so after a barge-in the brain's history holds
    what Adam heard rather than what was generated."""

    TYPE: ClassVar[str] = "playback_done"
    turn: int
    interrupted: bool = False
    heard: str = ""


@dataclass(frozen=True)
class TurnMetrics(Message):
    """Body-side timings for ``turn`` in ms, e.g. ``{"vad_trailing": 450,
    "first_audio_played": 830}`` measured from end of speech. The brain merges
    them with its own stages into the per-turn latency log line."""

    TYPE: ClassVar[str] = "turn_metrics"
    turn: int
    stages: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- brain -> body
@dataclass(frozen=True)
class Welcome(Message):
    """Handshake accepted. ``brain`` names which brain answers (``ollama:qwen2.5:7b``);
    ``tts_rate`` is the sample rate of TTS audio frames; ``settings`` is the
    current full settings snapshot."""

    TYPE: ClassVar[str] = "welcome"
    protocol: int
    brain: str
    tts_rate: int
    session: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateChange(Message):
    """``roco.bus.State`` value: idle | listening | thinking | speaking | sleeping."""

    TYPE: ClassVar[str] = "state"
    state: str
    turn: int = 0


@dataclass(frozen=True)
class EmotionChange(Message):
    """``roco.bus.Emotion`` value from the reply's leading tag."""

    TYPE: ClassVar[str] = "emotion"
    emotion: str
    turn: int = 0


@dataclass(frozen=True)
class Transcript(Message):
    """What the brain heard. ``final`` False = partial (optional, for display)."""

    TYPE: ClassVar[str] = "transcript"
    turn: int
    text: str
    final: bool = True


@dataclass(frozen=True)
class Sentence(Message):
    """Sentence ``index`` (0-based) of ``turn``'s reply. Its TTS audio follows as
    frames with the same turn and ``sentence`` index; the last frame of each
    sentence has the END flag. ``filler`` marks an instant line said while a slow
    tool runs ("Let me check."). Text always arrives before its audio."""

    TYPE: ClassVar[str] = "sentence"
    turn: int
    index: int
    text: str
    filler: bool = False


@dataclass(frozen=True)
class Announce(Message):
    """The brain starts speaking on its own (reminder due, briefing). ``turn`` is a
    :func:`brain_turn_id`. Sentences and audio for it follow like a reply."""

    TYPE: ClassVar[str] = "announce"
    turn: int
    text: str


@dataclass(frozen=True)
class TurnDone(Message):
    """All sentences and audio for ``turn`` have been sent. ``stages`` are the
    brain's timings in ms from end of speech (``stt_done``, ``first_token``,
    ``first_sentence``, ``first_audio_sent``, ...). ``brain`` is who answered."""

    TYPE: ClassVar[str] = "turn_done"
    turn: int
    text: str = ""
    brain: str = ""
    stages: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- codec
def encode(msg: Message) -> str:
    """Message -> JSON text frame."""
    if isinstance(msg, Unknown) or not msg.TYPE:
        raise ProtocolError("cannot encode an Unknown/base message")
    return json.dumps(msg.to_dict(), separators=(",", ":"), ensure_ascii=False)


def decode(text: str | bytes) -> Message:
    """JSON text frame -> Message. Unknown types give :class:`Unknown`.

    Raises :class:`ProtocolError` for invalid JSON, a wrong ``v``, or missing
    required fields. Unknown fields are ignored (forward compatibility).
    """
    try:
        data = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    version = data.pop("v", None)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version!r}")
    mtype = data.pop("type", None)
    if not isinstance(mtype, str):
        raise ProtocolError("message has no type")
    cls = _REGISTRY.get(mtype)
    if cls is None or cls is Unknown:
        return Unknown(type=mtype, data=data)
    names = {f.name for f in dataclasses.fields(cls)}
    extra = set(data) - names
    if extra:
        log.debug("ignoring unknown fields on %s: %s", mtype, sorted(extra))
    try:
        return cls(**{k: v for k, v in data.items() if k in names})
    except TypeError as exc:
        raise ProtocolError(f"bad {mtype} message: {exc}") from exc


def message_types() -> dict[str, type[Message]]:
    """All known message types by wire name (for docs and tests)."""
    return {k: v for k, v in _REGISTRY.items() if v is not Unknown}
