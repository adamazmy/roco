"""Binary audio frames of the ROCO link protocol v1.

Layout (network byte order), 18-byte header then the payload::

    0  2s  magic      b"RA"
    2  B   version    PROTOCOL_VERSION
    3  B   kind       AudioKind (1 = mic, body -> brain; 2 = tts, brain -> body)
    4  B   flags      AudioFlag bits (END = last frame of a sentence / of the turn)
    5  B   sentence   TTS: sentence index within the turn (mod 256); mic: 0
    6  I   turn       turn id (see roco.protocol.messages)
    10 I   seq        frame number within the turn, from 0, per kind
    14 I   rate       sample rate in Hz
    18 ... payload    PCM16 little-endian mono samples

Mic audio is 16 kHz, 80 ms per frame (2560 bytes). TTS frames are whatever the
brain's TTS produces (``Welcome.tts_rate``), sent in blocks of about 50-100 ms so
playback can start, and stop, quickly. Raw PCM keeps v1 simple: ~32 KB/s up and
~44-48 KB/s down, fine on a LAN and over Tailscale. A compressed codec would be a
new ``kind`` value, not a format change.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from roco.protocol.messages import PROTOCOL_VERSION, ProtocolError

MAGIC = b"RA"
HEADER = struct.Struct("!2sBBBBIII")
HEADER_SIZE = HEADER.size  # 18


class AudioKind(IntEnum):
    MIC = 1
    TTS = 2


class AudioFlag(IntFlag):
    NONE = 0
    END = 1  # TTS: last frame of this sentence. MIC: last frame of the turn.


@dataclass(frozen=True)
class AudioFrame:
    kind: AudioKind
    turn: int
    seq: int
    rate: int
    pcm: bytes
    sentence: int = 0
    flags: AudioFlag = AudioFlag.NONE

    @property
    def end(self) -> bool:
        return bool(self.flags & AudioFlag.END)

    @property
    def duration_ms(self) -> float:
        return 1000.0 * (len(self.pcm) // 2) / self.rate if self.rate else 0.0

    def to_bytes(self) -> bytes:
        return encode_audio(self)


def encode_audio(frame: AudioFrame) -> bytes:
    if len(frame.pcm) % 2:
        raise ProtocolError("PCM16 payload must have an even number of bytes")
    header = HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(frame.kind),
        int(frame.flags),
        frame.sentence & 0xFF,
        frame.turn & 0xFFFF_FFFF,
        frame.seq & 0xFFFF_FFFF,
        frame.rate,
    )
    return header + frame.pcm


def decode_audio(data: bytes) -> AudioFrame:
    if len(data) < HEADER_SIZE:
        raise ProtocolError(f"audio frame too short ({len(data)} bytes)")
    magic, version, kind, flags, sentence, turn, seq, rate = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError("not an audio frame (bad magic)")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported audio frame version {version}")
    try:
        akind = AudioKind(kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown audio kind {kind}") from exc
    pcm = bytes(data[HEADER_SIZE:])
    if len(pcm) % 2:
        raise ProtocolError("PCM16 payload must have an even number of bytes")
    return AudioFrame(
        kind=akind,
        turn=turn,
        seq=seq,
        rate=rate,
        pcm=pcm,
        sentence=sentence,
        flags=AudioFlag(flags),
    )
