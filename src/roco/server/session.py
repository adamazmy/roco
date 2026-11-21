"""One body session on the brain server: handshake, turns, speech out, barge-in.

Threads per session:

- the caller's thread (``serve_connection``) receives frames and dispatches them; it
  never blocks on the brain or on synthesis;
- a worker thread runs one job at a time (a user turn or an announcement): STT final,
  ``brain.respond_turn`` and the history amend after a barge-in;
- per reply, a speech thread turns the queue of sentences (and emotions, fillers) into
  ``sentence`` messages followed by their TTS audio frames, as soon as each sentence
  exists; it stops within one audio frame once the turn is cancelled;
- STT decodes run on the context's single STT thread while Adam is still talking.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from roco.brain.base import TurnHooks
from roco.bus import Topic
from roco.protocol import (
    PROTOCOL_VERSION,
    Announce,
    AudioFlag,
    AudioFrame,
    AudioKind,
    Connection,
    ConnectionClosed,
    EmotionChange,
    Error,
    Frame,
    Hello,
    Interrupt,
    Ping,
    PlaybackDone,
    Pong,
    ProtocolError,
    Sentence,
    Settings,
    StateChange,
    TextInput,
    Transcript,
    TurnDone,
    TurnEnd,
    TurnMetrics,
    TurnStart,
    Unknown,
    Welcome,
    brain_turn_id,
    token_matches,
)
from roco.server.context import BrainContext, FillerLine
from roco.server.latency import LatencyLog, TurnLatency
from roco.server.stt import StreamingSettings, StreamingSTT

log = logging.getLogger(__name__)

HANDSHAKE_TIMEOUT_S = 10.0
RECV_POLL_S = 0.1
PLAYBACK_WAIT_S = 30.0  # announcements wait at most this long for the last playback_done
APOLOGY = "Sorry, something went wrong on my side. Could you try that again?"

_END = object()


@dataclass
class _Turn:
    turn: int
    kind: str = "voice"  # voice | text | announce
    stt: StreamingSTT | None = None
    text: str | None = None  # known up front for text input / announcements
    speech_end: int | None = None  # STT byte offset where speech ended
    latency: TurnLatency | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    out: queue.Queue[Any] = field(default_factory=queue.Queue)
    queued_sentences: int = 0
    fillers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    finished: bool = False  # the brain returned and the turn is in its history
    heard: str | None = None  # from playback_done after an interrupt
    amended: bool = False
    seq: int = 0


class Session:
    def __init__(self, conn: Connection, ctx: BrainContext, stop: threading.Event) -> None:
        self.conn = conn
        self.ctx = ctx
        self.stop = stop
        self.session_id = uuid.uuid4().hex[:12]
        self.body_id = ""
        self._closing = threading.Event()
        self._jobs: queue.Queue[_Turn] = queue.Queue()
        self._turns: dict[int, _Turn] = {}  # active: recording or answering
        self._recent: OrderedDict[int, _Turn] = OrderedDict()  # answered, for amends
        self._lock = threading.RLock()
        self._recording: int | None = None
        self._answering: _Turn | None = None
        self._last_answered: _Turn | None = None
        self._unplayed: dict[int, float] = {}  # turn -> deadline for playback_done
        self._announcements: deque[str] = deque()
        self._announce_ids = itertools.count(1)
        self._filler_ids = itertools.count()
        self._send_lock = threading.Lock()
        self.latency = LatencyLog(ctx.latency_path)
        stt_cfg = ctx.server_cfg("stt", {}) or {}
        self._stt_settings = StreamingSettings.from_config(stt_cfg)
        self._frame_ms = int(ctx.server_cfg("tts_frame_ms", 80))
        filler = ctx.server_cfg("filler", {}) or {}
        self._filler_on = bool(filler.get("enabled", True)) and bool(ctx.fillers)
        self._fast_tools = set(filler.get("fast_tools", []) or [])
        self._wake_phrases = _wake_phrases(ctx)

    # ------------------------------------------------------------------ entry
    def run(self) -> None:
        if not self._handshake():
            return
        unsubs = [
            self.ctx.bus.subscribe(Topic.EMOTION, self._on_emotion),
            self.ctx.bus.subscribe(Topic.ANNOUNCE, self._on_announce),
            # A voice command changed the personality: tell the body.
            self.ctx.watch_brain_settings(
                lambda: self._send_quiet(Settings(values=self.ctx.settings_snapshot()))
            ),
        ]
        worker = threading.Thread(target=self._worker, name="roco-brain-turns", daemon=True)
        worker.start()
        try:
            self._receive_loop()
        finally:
            self._closing.set()
            with self._lock:
                for st in self._turns.values():
                    st.cancel.set()
            for unsub in unsubs:
                unsub()
            worker.join(timeout=5)
            self.latency.flush(force=True)
            self.ctx.body_lock.release()
            self.conn.close()
            log.info("Body %s disconnected (session %s)", self.body_id, self.session_id)

    # ------------------------------------------------------------------ handshake
    def _handshake(self) -> bool:
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        hello: Hello | None = None
        try:
            while hello is None:
                left = deadline - time.monotonic()
                if left <= 0 or self.stop.is_set():
                    return self._refuse("bad_message", "no hello received")
                try:
                    frame = self.conn.recv(timeout=min(left, 0.5))
                except ProtocolError as exc:
                    if "version" in str(exc):
                        return self._refuse("version", f"brain speaks protocol "
                                                       f"{PROTOCOL_VERSION}: {exc}")
                    return self._refuse("bad_message", str(exc))
                if frame is None:
                    continue
                if not isinstance(frame, Hello):
                    return self._refuse("bad_message", "first message must be hello")
                hello = frame
        except ConnectionClosed:
            self.conn.close()
            return False
        if hello.protocol != PROTOCOL_VERSION:
            return self._refuse("version", f"brain speaks protocol {PROTOCOL_VERSION}, "
                                           f"body sent {hello.protocol}")
        if not token_matches(self.ctx.token, hello.token):
            log.warning("Body %r refused: bad link token", hello.body_id)
            return self._refuse("auth", "bad link token (copy ~/.roco/link_token from the PC)")
        if not self.ctx.body_lock.acquire(blocking=False):
            log.warning("Body %r refused: another body is connected", hello.body_id)
            return self._refuse("busy", "another body is already connected")
        self.body_id = hello.body_id
        try:
            self._send(Welcome(
                protocol=PROTOCOL_VERSION,
                brain=self.ctx.brain_name,
                tts_rate=self.ctx.tts_rate,
                session=self.session_id,
                settings=self.ctx.settings_snapshot(),
            ))
        except ConnectionClosed:
            self.ctx.body_lock.release()
            self.conn.close()
            return False
        log.info("Body %r connected (session %s, mic %d Hz)", hello.body_id,
                 self.session_id, hello.mic_rate)
        return True

    def _refuse(self, code: str, message: str) -> bool:
        with contextlib.suppress(ConnectionClosed):
            self._send(Error(code=code, message=message, fatal=True))
        self.conn.close()
        return False

    # ------------------------------------------------------------------ receiving
    def _receive_loop(self) -> None:
        while not self.stop.is_set() and not self._closing.is_set():
            try:
                frame = self.conn.recv(timeout=RECV_POLL_S)
            except ConnectionClosed:
                return
            except ProtocolError as exc:
                log.warning("Bad frame from body: %s", exc)
                self._send_quiet(Error(code="bad_message", message=str(exc)))
                continue
            if frame is not None:
                try:
                    self._dispatch(frame)
                except ConnectionClosed:
                    return
                except Exception:
                    log.exception("Failed to handle %r", type(frame).__name__)
            self.latency.flush()

    def _dispatch(self, frame: Frame) -> None:
        if isinstance(frame, AudioFrame):
            self._on_mic(frame)
        elif isinstance(frame, TurnStart):
            self._on_turn_start(frame)
        elif isinstance(frame, TurnEnd):
            self._on_turn_end(frame)
        elif isinstance(frame, TextInput):
            self._on_text(frame)
        elif isinstance(frame, Interrupt):
            self._on_interrupt(frame)
        elif isinstance(frame, PlaybackDone):
            self._on_playback_done(frame)
        elif isinstance(frame, TurnMetrics):
            self.latency.body_metrics(frame.turn, frame.stages)
        elif isinstance(frame, Ping):
            self._send(Pong(id=frame.id, ts=frame.ts))
        elif isinstance(frame, Pong):
            pass
        elif isinstance(frame, Settings):
            try:
                snapshot = self.ctx.apply_settings(frame.values)
            except ValueError as exc:
                self._send(Error(code="bad_message", message=f"bad settings: {exc}"))
                return
            self._send(Settings(values=snapshot))
        elif isinstance(frame, Unknown):
            log.debug("Skipping unknown message type %r", frame.type)
        else:
            self._send(Error(code="bad_message", message=f"unexpected {frame.TYPE}"))

    def _on_turn_start(self, msg: TurnStart) -> None:
        with self._lock:
            for old in self._turns.values():
                if not old.cancel.is_set():
                    log.info("turn %d superseded by turn %d (%s)", old.turn, msg.turn,
                             msg.trigger)
                    old.cancel.set()
            st = _Turn(turn=msg.turn)
            st.stt = StreamingSTT(
                self.ctx.stt, self.ctx.stt_executor, settings=self._stt_settings,
                on_partial=lambda text, turn=msg.turn: self._send_quiet(
                    Transcript(turn=turn, text=text, final=False)),
            )
            self._turns[msg.turn] = st
            self._recording = msg.turn

    def _on_mic(self, frame: AudioFrame) -> None:
        if frame.kind != AudioKind.MIC:
            return
        st = self._turns.get(frame.turn)
        if st is None or st.stt is None or st.speech_end is not None:
            return
        if frame.rate != st.stt.rate:
            st.stt.rate = frame.rate
        st.stt.feed(frame.pcm)

    def _on_turn_end(self, msg: TurnEnd) -> None:
        arrived = time.monotonic()
        with self._lock:
            st = self._turns.get(msg.turn)
            if self._recording == msg.turn:
                self._recording = None
            if st is None or st.stt is None or st.cancel.is_set():
                return
            if msg.reason == "nothing":
                self._turns.pop(msg.turn, None)
                return
            trailing = max(0, int(msg.trailing_ms))
            st.speech_end = max(0, st.stt.total_bytes - st.stt.bytes_for_ms(trailing))
            st.latency = TurnLatency(msg.turn, arrived - trailing / 1000)
            st.latency.info["trailing_ms"] = trailing
        self._jobs.put(st)

    def _on_text(self, msg: TextInput) -> None:
        with self._lock:
            for old in self._turns.values():
                old.cancel.set()
            st = _Turn(turn=msg.turn, kind="text", text=msg.text)
            st.latency = TurnLatency(msg.turn, time.monotonic(), kind="text")
            self._turns[msg.turn] = st
        self._jobs.put(st)

    def _on_interrupt(self, msg: Interrupt) -> None:
        with self._lock:
            st = self._turns.get(msg.turn) or self._recent.get(msg.turn)
            self._unplayed.pop(msg.turn, None)
        if st is None:
            return
        st.cancel.set()
        if st.latency is not None:
            st.latency.info["interrupted"] = msg.reason
        log.info("turn %d interrupted (%s)", msg.turn, msg.reason)

    def _on_playback_done(self, msg: PlaybackDone) -> None:
        with self._lock:
            self._unplayed.pop(msg.turn, None)
            st = self._turns.get(msg.turn) or self._recent.get(msg.turn)
        if st is None or not msg.interrupted:
            return
        st.cancel.set()
        heard = msg.heard.strip()
        for filler in st.fillers:  # fillers are ours, not the model's words
            if heard.startswith(filler):
                heard = heard[len(filler):].strip()
        st.heard = heard  # applied by the worker (it owns the brain)

    # ------------------------------------------------------------------ bus
    def _on_emotion(self, emotion: Any) -> None:
        st = self._answering
        if st is not None and not st.cancel.is_set():
            st.out.put(("emotion", str(emotion)))

    def _on_announce(self, text: Any) -> None:
        text = str(text or "").strip()
        if text:
            with self._lock:
                self._announcements.append(text)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while not self._closing.is_set():
            self._apply_amends()
            try:
                st = self._jobs.get(timeout=RECV_POLL_S)
            except queue.Empty:
                self._maybe_announce()
                continue
            try:
                self._run_turn(st)
            except ConnectionClosed:
                self._closing.set()
            except Exception:
                log.exception("turn %d failed", st.turn)
            finally:
                with self._lock:
                    self._turns.pop(st.turn, None)
                    self._recent[st.turn] = st
                    while len(self._recent) > 8:
                        self._recent.popitem(last=False)

    def _apply_amends(self) -> None:
        st = self._last_answered
        if st is None or st.amended or st.heard is None or not st.finished:
            return
        st.amended = True
        log.info("turn %d: history keeps what was heard: %r", st.turn, st.heard)
        try:
            self.ctx.brain.amend_last_reply(st.heard)
        except Exception:
            log.exception("amend_last_reply failed")

    def _maybe_announce(self) -> None:
        with self._lock:
            if not self._announcements or self._recording is not None or self._turns:
                return
            now = time.monotonic()
            if any(deadline > now for deadline in self._unplayed.values()):
                return
            self._unplayed.clear()
            text = self._announcements.popleft()
            st = _Turn(turn=brain_turn_id(next(self._announce_ids)), kind="announce", text=text)
            self._turns[st.turn] = st
        log.info("Announcing: %r", text)
        try:
            self._send(Announce(turn=st.turn, text=text))
            self._speak_lines(st, [text])
            if not st.cancel.is_set():
                self._send(TurnDone(turn=st.turn, text=text, brain=self.ctx.brain_name))
                with self._lock:
                    self._unplayed[st.turn] = time.monotonic() + PLAYBACK_WAIT_S
        except ConnectionClosed:
            self._closing.set()
        finally:
            with self._lock:
                self._turns.pop(st.turn, None)
                self._recent[st.turn] = st

    def _run_turn(self, st: _Turn) -> None:
        if st.cancel.is_set():
            return
        lat = st.latency or TurnLatency(st.turn, time.monotonic())
        self._send(StateChange(state="thinking", turn=st.turn))
        text = st.text
        if text is None and st.stt is not None:
            text = st.stt.finish(st.speech_end)
            lat.mark("stt_done")
            lat.info["stt"] = dict(st.stt.stats)
        text = (text or "").strip()
        if st.kind == "voice":
            # The body sends a pre-roll before the wake word fired, so "Jarvis, what's on
            # today?" (or a mishearing such as "Hey, George, ...") often starts the audio.
            text = strip_wake(text, self._wake_phrases)
        lat.info["transcript"] = text
        if st.kind == "voice":
            self._send(Transcript(turn=st.turn, text=text))
        log.info("turn %d: you said %r", st.turn, text)
        if st.cancel.is_set():
            return
        if not text:
            self._send(TurnDone(turn=st.turn, text="", brain=self.ctx.brain_name,
                                stages=lat.stages()))
            self._send(StateChange(state="idle", turn=st.turn))
            return
        reply = self._answer(st, text, lat)
        if st.cancel.is_set():
            lat.info.setdefault("interrupted", "cancelled")
        else:
            self._send(TurnDone(turn=st.turn, text=reply, brain=self.ctx.brain_name,
                                stages=lat.stages()))
            with self._lock:
                self._unplayed[st.turn] = time.monotonic() + PLAYBACK_WAIT_S
        log.info("turn %d: ROCO said %r", st.turn, reply)
        self.latency.turn_done(lat, reply_chars=len(reply), tools=st.tools,
                               fillers=list(st.fillers), brain=self.ctx.brain_name)

    def _answer(self, st: _Turn, text: str, lat: TurnLatency) -> str:
        speech = threading.Thread(target=self._speech_loop, args=(st, lat),
                                  name=f"roco-speech-{st.turn}", daemon=True)
        speech.start()

        def on_sentence(sentence: str) -> None:
            if st.cancel.is_set():
                return
            lat.mark("first_sentence")
            st.queued_sentences += 1
            st.out.put(("sentence", sentence, False, None))

        def on_tools(names: list[str]) -> None:
            st.tools.extend(n for n in names if n)
            self._maybe_filler(st, names)

        hooks = TurnHooks(on_token=lambda: lat.mark("first_token"), on_tools=on_tools)
        self._answering = st
        reply = ""
        try:
            reply = self.ctx.brain.respond_turn(text, on_sentence, cancel=st.cancel, hooks=hooks)
        except Exception:
            log.exception("brain failed on turn %d", st.turn)
            on_sentence(APOLOGY)
            reply = APOLOGY
        finally:
            self._answering = None
            st.finished = True
            self._last_answered = st
            st.out.put(_END)
        while speech.is_alive() and not self._closing.is_set():
            speech.join(timeout=0.1)
        return reply

    def _maybe_filler(self, st: _Turn, names: list[str]) -> None:
        if not self._filler_on or st.queued_sentences or st.fillers or st.cancel.is_set():
            return
        if names and all(n in self._fast_tools for n in names):
            return
        filler: FillerLine = self.ctx.fillers[next(self._filler_ids) % len(self.ctx.fillers)]
        st.fillers.append(filler.text)
        st.queued_sentences += 1
        st.out.put(("sentence", filler.text, True, filler.audio))

    # ------------------------------------------------------------------ speech out
    def _speak_lines(self, st: _Turn, lines: Iterable[str]) -> None:
        """Speak fixed text on the calling thread (announcements)."""
        lat = TurnLatency(st.turn, time.monotonic(), kind="announce")
        for line in lines:
            st.out.put(("sentence", line, False, None))
        st.out.put(_END)
        self._speech_loop(st, lat)

    def _speech_loop(self, st: _Turn, lat: TurnLatency) -> None:
        index = 0
        while True:
            item = st.out.get()
            if item is _END or self._closing.is_set():
                return
            if st.cancel.is_set():
                continue  # drain: nothing more for a cancelled turn
            try:
                if item[0] == "emotion":
                    self._send(EmotionChange(emotion=item[1], turn=st.turn))
                    continue
                _, text, filler, audio = item
                self._send(Sentence(turn=st.turn, index=index, text=text, filler=filler))
                self._send_audio(st, index, audio if audio is not None else
                                 self._synthesize(text), lat)
                index += 1
            except ConnectionClosed:
                self._closing.set()
                st.cancel.set()
                return
            except Exception:
                log.exception("speech output failed on turn %d", st.turn)

    def _synthesize(self, text: str) -> Iterator[tuple[bytes, int]]:
        try:
            yield from self.ctx.tts(text)
        except Exception:
            log.exception("TTS failed for %r", text)

    def _send_audio(self, st: _Turn, index: int, chunks: Iterable[tuple[bytes, int]],
                    lat: TurnLatency) -> None:
        # Each block goes out the moment it exists; an empty END frame closes the
        # sentence (waiting to know which block is last would delay the first one).
        rate = self.ctx.tts_rate
        for pcm, rate in _frames(chunks, self._frame_ms, lat):
            if st.cancel.is_set():
                return  # the body drops anything for a cancelled turn
            self._send_frame(st, index, pcm, rate, end=False, lat=lat)
        if not st.cancel.is_set():
            self._send_frame(st, index, b"", rate, end=True, lat=lat)

    def _send_frame(self, st: _Turn, index: int, pcm: bytes, rate: int, *, end: bool,
                    lat: TurnLatency) -> None:
        self._send(AudioFrame(
            kind=AudioKind.TTS, turn=st.turn, seq=st.seq, rate=rate, pcm=pcm,
            sentence=index, flags=AudioFlag.END if end else AudioFlag.NONE,
        ))
        st.seq += 1
        if pcm:
            lat.mark("first_audio_sent")

    # ------------------------------------------------------------------ sending
    def _send(self, frame: Frame) -> None:
        with self._send_lock:
            self.conn.send(frame)

    def _send_quiet(self, frame: Frame) -> None:
        with contextlib.suppress(ConnectionClosed):
            self._send(frame)


def _frames(chunks: Iterable[tuple[bytes, int]], frame_ms: int,
            lat: TurnLatency) -> Iterator[tuple[bytes, int]]:
    """Re-cut TTS output into ~``frame_ms`` blocks, as soon as each block is available."""
    buf = bytearray()
    rate = 0
    for pcm, chunk_rate in chunks:
        lat.mark("first_tts_done")
        if rate and chunk_rate != rate and buf:
            yield bytes(buf), rate
            buf.clear()
        rate = int(chunk_rate)
        size = max(2, int(rate * frame_ms / 1000) * 2)
        buf += pcm
        while len(buf) >= size:
            yield bytes(buf[:size]), rate
            del buf[:size]
    if buf:
        yield bytes(buf[: len(buf) - len(buf) % 2]), rate


def _wake_phrases(ctx: BrainContext) -> list[str]:
    phrases: list[str] = []
    try:
        from roco.voice.models import wake_phrase

        phrases.append(wake_phrase(ctx.cfg))
    except Exception as exc:  # pragma: no cover - a broken wake word config
        log.debug("No wake phrase to strip: %s", exc)
    phrases += [str(p) for p in ctx.server_cfg("wake_aliases", []) or []]
    return [p for p in phrases if p and p.strip()]


def strip_wake(text: str, phrases: list[str]) -> str:
    """Remove one leading wake phrase (or alias) from a transcript."""
    from roco.voice.listener import strip_wake_phrase

    for phrase in phrases:
        stripped = strip_wake_phrase(text, phrase)
        if stripped != text:
            return stripped.strip()
    return text


def serve_connection(conn: Connection, ctx: BrainContext, stop: threading.Event) -> None:
    """Run one body session on ``conn`` until it closes or ``stop`` is set."""
    Session(conn, ctx, stop).run()
