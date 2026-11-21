"""The body client's brain-facing logic: turns, playback, barge-in, fallback.

One mic thread calls :meth:`BodyClient.step` with every (echo-cancelled) 80 ms frame.
The step is a small state machine::

    IDLE ──wake word──▶ RECORDING ──VAD: 450 ms silence──▶ REPLY ──audio drained──▶ FOLLOWUP
      ▲                     │ (frames stream up live)        │ speech over playback  │ speech
      │                     └── no speech: turn_end nothing  ▼ (barge-in)            ▼
      └───────────────────────────────── timeout ──── RECORDING (trigger barge_in / followup)

Messages from the brain arrive on the link thread (:meth:`on_frame`), playback
events on the player's monitor thread. Everything that touches the state holds
``self._lock``; network sends and the playback cut are quick, so holding it is fine.

Barge-in: while a reply is pending or playing, the echo-cancelled mic goes through a
second VAD endpointer (stricter: 0.6 probability for 160 ms). When it fires, the
playback is cut *locally first* (no round trip), then ``interrupt``,
``playback_done{interrupted, heard}`` and a new ``turn_start{barge_in}`` go up,
carrying the last few frames so the first syllable is not lost. If the echo
canceller is missing or proves unreliable (low ERLE, barge-ins that contain no
speech), barge-in falls back to the wake word, and the console says so.

Late frames: anything that arrives for a cancelled turn, or for a user turn that is
not the current reply, is dropped and counted (``stats["late_dropped"]``).

Fallback: when the link is down at the end of a turn (or drops before any answer
audio arrived), the utterance goes to the fallback chain (workstream D) on a worker
thread, and its sentences are spoken with a local TTS on the body.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from roco.bus import Emotion, EventBus, State
from roco.client.playback import Player
from roco.client.vad import EndpointSettings, Vad, VadEndpointer
from roco.protocol import (
    PROTOCOL_VERSION,
    Announce,
    AudioFrame,
    AudioKind,
    Connection,
    EmotionChange,
    Error,
    Frame,
    Hello,
    Interrupt,
    PlaybackDone,
    Sentence,
    Settings,
    StateChange,
    Transcript,
    TurnDone,
    TurnEnd,
    TurnMetrics,
    TurnStart,
    Unknown,
    Welcome,
    is_brain_turn,
)
from roco.voice.endpoint import Endpoint

log = logging.getLogger(__name__)

TTS = Callable[[str], Iterable[tuple[bytes, int]]]
LOCAL_TURN_BASE = -1  # local announcements (brain restored line) use negative ids


class Mode(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    REPLY = "reply"
    FOLLOWUP = "followup"


class BargeInMode(StrEnum):
    SPEECH = "speech"  # any speech over playback (needs a working AEC)
    WAKE_WORD = "wake_word"  # only the wake word interrupts
    OFF = "off"


@dataclass
class BodySettings:
    sample_rate: int = 16_000
    frame_ms: int = 80
    endpoint: EndpointSettings = field(default_factory=EndpointSettings)
    followup_s: float = 5.0
    wake_threshold: float = 0.5
    wake_tail_frames: int = 3  # frames after the wake word that are still "...Jarvis"
    preroll_frames: int = 4  # sent at the start of speech-triggered turns (320 ms)
    barge_in: BargeInMode = BargeInMode.SPEECH
    barge_threshold: float = 0.6
    barge_start_ms: float = 160.0
    echo_guard_frames: int = 4  # skipped after playback when there is no AEC
    reply_timeout_s: float = 30.0  # no answer at all from the brain: give up
    min_erle_db: float = 10.0  # below this (median, while playing) the AEC is not trusted
    erle_min_frames: int = 25
    max_false_barge_ins: int = 2  # consecutive barge-ins with no real speech


@dataclass
class _Recording:
    turn: int
    trigger: str
    endpointer: VadEndpointer
    local: bool  # the brain never heard of this turn (link was down)
    tail_left: int = 0
    seq: int = 0
    frames: list[bytes] = field(default_factory=list)
    speech_end_t: float = 0.0
    barge_in: bool = False


@dataclass
class _Reply:
    turn: int
    speech_end_t: float
    trailing_ms: int
    sent_t: float
    pcm: bytes = b""  # the utterance, kept for the fallback if the link drops
    local: bool = False  # answered by the fallback chain
    done: bool = False  # turn_done received (or the fallback finished)
    audio_t: float | None = None  # first sample handed to the device
    brain: str = ""


class BodyClient:
    def __init__(
        self,
        *,
        source: Any,
        vad: Vad,
        player: Player,
        bus: EventBus,
        connect: Callable[[], Connection] | None,
        token: str,
        wake: Any = None,
        settings: BodySettings | None = None,
        body_id: str = "body",
        caps: dict[str, Any] | None = None,
        aec_name: str = "none",
        echo_monitor: Any = None,
        fallback_factory: Callable[[], Any] | None = None,
        local_tts_factory: Callable[[], TTS] | None = None,
        output_latency: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        link_options: dict[str, Any] | None = None,
    ) -> None:
        from roco.client.link import LinkClient

        self.source = source
        self.vad = vad
        self.player = player
        self.bus = bus
        self.wake = wake
        self.s = settings or BodySettings()
        self.token = token
        self.body_id = body_id
        self.caps = dict(caps or {})
        self.aec_name = aec_name
        self.echo_monitor = echo_monitor
        self.fallback_factory = fallback_factory
        self.local_tts_factory = local_tts_factory
        self.output_latency = output_latency or (lambda: 0.0)
        self.clock = clock
        self.frame_s = self.s.frame_ms / 1000.0

        self.barge_mode = self.s.barge_in
        if self.barge_mode is BargeInMode.SPEECH and aec_name == "none":
            self.barge_mode = BargeInMode.WAKE_WORD
            self.degrade_reason = "no echo canceller installed"
            log.warning("Barge-in on any speech needs an echo canceller; using the wake word")
        else:
            self.degrade_reason = ""

        self._lock = threading.RLock()
        self.mode = Mode.IDLE
        self.muted = False
        self._next_turn = 1
        self._next_local = LOCAL_TURN_BASE
        self._rec: _Recording | None = None
        self._reply: _Reply | None = None
        self._followup: VadEndpointer | None = None
        self._guard = 0
        self._barge: VadEndpointer | None = None
        self._cancelled: set[int] = set()
        self._announcements: set[int] = set()
        self._preroll: deque[bytes] = deque(maxlen=max(1, self.s.preroll_frames))
        self._false_barges = 0
        self._barge_turns: set[int] = set()
        self._fallback: Any = None
        self._fallback_lock = threading.Lock()
        self._local_tts: TTS | None = None
        self._fallback_spoke = False
        self.stats: dict[str, Any] = {"late_dropped": 0, "barge_ins": [], "turns": []}
        self.last_metrics: dict[int, dict[str, int]] = {}
        self.brain_name = ""

        player.listener = self
        self.link = None
        if connect is not None:
            self.link = LinkClient(
                connect, self._hello, self.on_frame, on_up=self._on_link_up,
                on_down=self._on_link_down, clock=clock, **(link_options or {}),
            )

    # ================================================================== lifecycle
    def run(self, stop: threading.Event) -> None:
        """Mic loop. Returns when ``stop`` is set."""
        self.start()
        try:
            while not stop.is_set():
                frame = self.source.read(0.1)
                if frame is None:
                    self.tick()
                    continue
                self.step(frame)
        finally:
            self.close()

    def start(self) -> None:
        self.player.start()
        if self.link is not None:
            self.link.start()

    def close(self) -> None:
        if self.link is not None:
            self.link.stop()
        with contextlib.suppress(Exception):
            self.player.cut()
        self.player.close()
        fb = self._fallback
        if fb is not None and hasattr(fb, "close"):
            with contextlib.suppress(Exception):
                fb.close()

    @property
    def link_up(self) -> bool:
        return self.link is not None and self.link.is_up

    def _send(self, frame: Frame) -> bool:
        return self.link.send(frame) if self.link is not None else False

    def _hello(self) -> Hello:
        caps = {**self.caps, "aec": self.aec_name, "vad": getattr(self.vad, "name", "?"),
                "barge_in": str(self.barge_mode), "playback_rate": self.player.rate}
        return Hello(protocol=PROTOCOL_VERSION, body_id=self.body_id, token=self.token,
                     mic_rate=self.s.sample_rate, caps=caps)

    # ================================================================== mic side
    def step(self, frame: Any) -> None:
        """Process one mic frame (already echo-cancelled)."""
        pcm = frame.tobytes() if hasattr(frame, "tobytes") else bytes(frame)
        with self._lock:
            if self.muted:
                return
            scores = self.vad.scores(frame)
            wake = 0.0
            if self.wake is not None and self.mode is not Mode.RECORDING:
                wake = float(self.wake.score(frame))
            self._preroll.append(pcm)
            if self.mode is Mode.IDLE:
                self._step_idle(wake, scores)
            elif self.mode is Mode.RECORDING:
                self._step_recording(pcm, scores)
            elif self.mode is Mode.REPLY:
                self._step_reply(wake, scores)
            elif self.mode is Mode.FOLLOWUP:
                self._step_followup(wake, scores)
            self._check_timeouts()

    def tick(self) -> None:
        """No mic frame arrived: still enforce timeouts."""
        with self._lock:
            self._check_timeouts()

    def _woke(self, wake: float) -> bool:
        if wake >= self.s.wake_threshold:
            log.info("Wake word detected (score %.2f)", wake)
            if self.wake is not None:
                self.wake.reset()
            return True
        return False

    def _step_idle(self, wake: float, scores: list) -> None:
        if self._woke(wake):
            self._cut_announcements("wake")
            self._start_turn("wake", [], skip_tail=True)
            return
        if self._announcements and self._barge_detect(wake, scores):
            self._barge_in(None)

    def _step_followup(self, wake: float, scores: list) -> None:
        if self._woke(wake):
            self._start_turn("wake", [], skip_tail=True)
            return
        if self._guard > 0:
            self._guard -= 1
            return
        assert self._followup is not None
        event = self._followup.feed(scores)
        if event is Endpoint.SPEECH_START:
            self._start_turn("followup", list(self._preroll), skip_tail=False)
        elif event is Endpoint.NO_SPEECH:
            self._enter_idle()

    def _step_recording(self, pcm: bytes, scores: list) -> None:
        rec = self._rec
        assert rec is not None
        if rec.tail_left > 0:  # still the end of the wake phrase: drop it
            rec.tail_left -= 1
            if any(p >= self.s.endpoint.threshold for p, _ in scores):
                return
            rec.tail_left = 0
        self._send_mic(rec, pcm)
        event = rec.endpointer.feed(scores)
        if rec.endpointer.speech_started and rec.endpointer.silence_ms == 0:
            rec.speech_end_t = self.clock()
        if event is Endpoint.SILENCE:
            self._end_turn("silence")
        elif event is Endpoint.MAX_LENGTH:
            self._end_turn("max_length")
        elif event is Endpoint.NO_SPEECH:
            self._end_turn("nothing")

    def _step_reply(self, wake: float, scores: list) -> None:
        if self._barge_detect(wake, scores):
            self._barge_in(self._reply.turn if self._reply else None)

    def _barge_detect(self, wake: float, scores: list) -> bool:
        if self.barge_mode is BargeInMode.OFF:
            return False
        if self.barge_mode is BargeInMode.WAKE_WORD:
            return self._woke(wake)
        if self._barge is None:
            self._barge = self._new_barge_endpointer()
        if self._barge.feed(scores) is Endpoint.SPEECH_START:
            return True
        return self._woke(wake)  # the wake word always works too

    def _new_barge_endpointer(self) -> VadEndpointer:
        e = self.s.endpoint
        return VadEndpointer(EndpointSettings(
            threshold=self.s.barge_threshold,
            neg_threshold=max(e.neg_threshold, self.s.barge_threshold - 0.15),
            start_ms=self.s.barge_start_ms, trailing_ms=e.trailing_ms,
            max_s=1e9, start_timeout_s=1e9,
        ))

    # ---------------------------------------------------------------- turns
    def _start_turn(self, trigger: str, initial: list[bytes], skip_tail: bool) -> None:
        turn = self._next_turn
        self._next_turn += 1
        ep = VadEndpointer(EndpointSettings(**vars(self.s.endpoint)))
        if not skip_tail:  # speech is already under way (follow-up, barge-in)
            ep.speech_started = True
            ep.speech_ms = ep.s.start_ms
            if trigger == "barge_in" and self._barge is not None:
                ep.speech_ms = self._barge.speech_ms
        local = not self._send(TurnStart(turn=turn, trigger=trigger))
        rec = _Recording(turn=turn, trigger=trigger, endpointer=ep, local=local,
                         tail_left=self.s.wake_tail_frames if skip_tail else 0,
                         barge_in=trigger == "barge_in")
        rec.speech_end_t = self.clock()
        self._rec = rec
        self._barge = None
        self._followup = None
        self.mode = Mode.RECORDING
        self.bus.set_state(State.LISTENING)
        for pcm in initial:
            self._send_mic(rec, pcm)
        log.info("Turn %d started (%s)%s", turn, trigger, " [offline]" if local else "")

    def _send_mic(self, rec: _Recording, pcm: bytes) -> None:
        rec.frames.append(pcm)
        if not rec.local:
            ok = self._send(AudioFrame(kind=AudioKind.MIC, turn=rec.turn, seq=rec.seq,
                                       rate=self.s.sample_rate, pcm=pcm))
            if not ok:
                rec.local = True  # the link dropped mid-turn: the fallback will answer
        rec.seq += 1

    def _end_turn(self, reason: str) -> None:
        rec = self._rec
        assert rec is not None
        self._rec = None
        trailing = rec.endpointer.trailing_ms if reason != "nothing" else 0
        if not rec.local:
            self._send(TurnEnd(turn=rec.turn, reason=reason, trailing_ms=trailing))
        self._track_false_barge(rec, reason)
        if reason == "nothing":
            log.info("Turn %d: no speech", rec.turn)
            self._enter_idle()
            return
        now = self.clock()
        speech_end = rec.speech_end_t or now - trailing / 1000.0
        reply = _Reply(turn=rec.turn, speech_end_t=speech_end, trailing_ms=trailing,
                       sent_t=now, pcm=b"".join(rec.frames))
        self._reply = reply
        self.mode = Mode.REPLY
        self._barge = None
        if self.echo_monitor is not None:
            self.echo_monitor.reset()
        self.bus.set_state(State.THINKING)
        log.info("Turn %d ended (%s, %d ms trailing silence)", rec.turn, reason, trailing)
        if rec.local or not self.link_up:
            self._start_fallback(reply)

    def _track_false_barge(self, rec: _Recording, reason: str) -> None:
        """A barge-in is "false" when it turns out to hold no words: no speech at all,
        or (verdict from the brain's STT) an empty transcript. Local turns have no
        transcript, so for them very short speech counts as false."""
        if not rec.barge_in:
            return
        if reason == "nothing":
            self._barge_verdict(False)
        elif rec.local:
            self._barge_verdict(rec.endpointer.speech_ms >= 250)
        else:
            self._barge_turns.add(rec.turn)  # decided when the transcript arrives

    def _barge_verdict(self, real: bool) -> None:
        self._false_barges = 0 if real else self._false_barges + 1
        if self._false_barges >= self.s.max_false_barge_ins:
            self._degrade(f"{self._false_barges} barge-ins in a row had no words in them")

    def _barge_in(self, turn: int | None) -> None:
        """Speech over ROCO: cut locally first, then tell the brain."""
        t_decide = self.clock()
        detect_ms = int(self._barge.speech_ms) if self._barge is not None else 0
        targets = [turn] if turn is not None else []
        targets += sorted(self._announcements)
        heard = {t: self.player.cut(t) for t in targets}
        cut_ms = (self.clock() - t_decide) * 1000.0
        stop_ms = detect_ms + cut_ms + self.output_latency() * 1000.0
        self.stats["barge_ins"].append({"turn": turn, "cut_ms": round(cut_ms, 2),
                                        "detect_ms": detect_ms, "stop_ms": round(stop_ms)})
        log.info("Barge-in on turn %s: playback cut in %.1f ms (speech for %d ms before the "
                 "decision; ~%d ms from speech onset to silence)", turn, cut_ms, detect_ms,
                 stop_ms)
        for t in targets:
            self._cancelled.add(t)
            self._announcements.discard(t)
            if t < 0:
                continue
            self._send(Interrupt(turn=t, reason="barge_in"))
            self._send(PlaybackDone(turn=t, interrupted=True, heard=heard.get(t, "")))
            if t == turn and self._reply is not None:
                stages = self._stages(self._reply)
                stages.update(barge_in_cut=int(round(cut_ms)), barge_in_detect=detect_ms)
                self._send(TurnMetrics(turn=t, stages=stages))
        reply = self._reply
        if reply is not None and reply.local:
            self._cancel_fallback()
        self._reply = None
        self._start_turn("barge_in", list(self._preroll), skip_tail=False)

    def _cut_announcements(self, reason: str) -> None:
        for t in sorted(self._announcements):
            heard = self.player.cut(t)
            self._cancelled.add(t)
            if t >= 0:
                self._send(Interrupt(turn=t, reason="user"))
                self._send(PlaybackDone(turn=t, interrupted=True, heard=heard))
        self._announcements.clear()

    def _enter_idle(self) -> None:
        self.mode = Mode.IDLE
        self._followup = None
        self._barge = None
        if self.wake is not None:
            self.wake.reset()
        if not self._announcements:
            self.bus.set_state(State.IDLE)

    def _enter_followup(self) -> None:
        if self.s.followup_s <= 0:
            self._enter_idle()
            return
        self.mode = Mode.FOLLOWUP
        self._followup = VadEndpointer(EndpointSettings(
            **{**vars(self.s.endpoint), "start_timeout_s": self.s.followup_s}))
        self._guard = self.s.echo_guard_frames if self.aec_name == "none" else 0
        if self.wake is not None:
            self.wake.reset()
        self.bus.set_state(State.LISTENING)

    def _check_timeouts(self) -> None:
        reply = self._reply
        if (self.mode is Mode.REPLY and reply is not None and not reply.done
                and reply.audio_t is None
                and self.clock() - reply.sent_t > self.s.reply_timeout_s):
            log.warning("Turn %d: no answer after %.0f s, giving up", reply.turn,
                        self.s.reply_timeout_s)
            self._cancelled.add(reply.turn)
            self.player.cut(reply.turn)
            if not reply.local:
                self._send(Interrupt(turn=reply.turn, reason="user"))
            self._reply = None
            self._enter_idle()

    # ================================================================== brain side
    def on_frame(self, frame: Frame) -> None:
        """A frame from the brain (link thread)."""
        with self._lock:
            if isinstance(frame, AudioFrame):
                self._on_audio(frame)
            elif isinstance(frame, Sentence):
                if self._accepts(frame.turn):
                    self.player.set_text(frame.turn, frame.index, frame.text)
                    log.info("ROCO%s: %s", " (filler)" if frame.filler else "", frame.text)
                else:
                    self.stats["late_dropped"] += 1
            elif isinstance(frame, TurnDone):
                self._on_turn_done(frame)
            elif isinstance(frame, Announce):
                self._announcements.add(frame.turn)
                log.info("Announcement: %s", frame.text)
            elif isinstance(frame, EmotionChange):
                if self._accepts(frame.turn) or frame.turn == 0:
                    self.bus.set_emotion(Emotion.parse(frame.emotion, Emotion.NEUTRAL))
            elif isinstance(frame, StateChange):
                self._on_state(frame)
            elif isinstance(frame, Transcript):
                if frame.final:
                    log.info("Brain heard (turn %d): %r", frame.turn, frame.text)
                    if frame.turn in self._barge_turns:
                        self._barge_turns.discard(frame.turn)
                        self._barge_verdict(bool(frame.text.strip()))
            elif isinstance(frame, Settings):
                if "muted" in frame.values:
                    self.set_muted(bool(frame.values["muted"]))
            elif isinstance(frame, Error):
                log.warning("Brain error %s: %s", frame.code, frame.message)
            elif isinstance(frame, Unknown):
                log.debug("ignoring unknown message %s", frame.type)
            elif isinstance(frame, Welcome):
                pass

    def _accepts(self, turn: int) -> bool:
        """Is ``turn`` still wanted? (late frames of cancelled turns are not)"""
        if turn in self._cancelled:
            return False
        if is_brain_turn(turn):
            return True
        reply = self._reply
        return reply is not None and reply.turn == turn and not reply.local

    def _on_audio(self, frame: AudioFrame) -> None:
        if frame.kind is not AudioKind.TTS or not self._accepts(frame.turn):
            self.stats["late_dropped"] += 1
            return
        if not self.player.feed(frame.turn, frame.sentence, frame.pcm, frame.rate, frame.end):
            self.stats["late_dropped"] += 1

    def _on_turn_done(self, msg: TurnDone) -> None:
        if not self._accepts(msg.turn):
            self.stats["late_dropped"] += 1
            return
        reply = self._reply
        if reply is not None and reply.turn == msg.turn:
            reply.done = True
            reply.brain = msg.brain
            if msg.brain:
                self.brain_name = msg.brain
        self.player.finish(msg.turn)

    def _on_state(self, msg: StateChange) -> None:
        if msg.turn and msg.turn in self._cancelled:
            return
        # The body owns listening/speaking/idle (it knows what the mic and speaker do).
        if msg.state in (State.THINKING, State.SLEEPING):
            self.bus.set_state(State(msg.state))

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self.muted = muted
            if muted and self._rec is not None:
                self._end_turn("nothing")
            log.info("Mic %s", "muted" if muted else "unmuted")

    # ================================================================== player events
    def on_audio_started(self, turn: int, t: float) -> None:
        with self._lock:
            if turn in self._cancelled:
                return
            reply = self._reply
            if reply is not None and reply.turn == turn and reply.audio_t is None:
                reply.audio_t = t + self.output_latency()
            self.bus.set_state(State.SPEAKING)

    def on_turn_drained(self, turn: int) -> None:
        with self._lock:
            if turn in self._cancelled:
                return
            if turn in self._announcements:
                self._announcements.discard(turn)
                if turn >= 0:
                    self._send(PlaybackDone(turn=turn, heard=self.player.heard(turn)))
                self.player.forget(turn)
                if self.mode is Mode.IDLE and not self._announcements:
                    self.bus.set_state(State.IDLE)
                return
            reply = self._reply
            if reply is None or reply.turn != turn or not reply.done:
                return
            self._complete_reply(reply)

    def on_mouth(self, amplitude: float) -> None:
        self.bus.set_mouth(amplitude)

    def _stages(self, reply: _Reply) -> dict[str, int]:
        stages = {"vad_trailing": int(reply.trailing_ms)}
        if reply.audio_t is not None:
            stages["first_audio_played"] = int(round((reply.audio_t - reply.speech_end_t)
                                                     * 1000))
        return stages

    def _complete_reply(self, reply: _Reply) -> None:
        heard = self.player.heard(reply.turn)
        stages = self._stages(reply)
        self.last_metrics[reply.turn] = stages
        self.stats["turns"].append({"turn": reply.turn, "local": reply.local, **stages})
        if not reply.local:
            self._send(PlaybackDone(turn=reply.turn, interrupted=False, heard=heard))
            self._send(TurnMetrics(turn=reply.turn, stages=stages))
        log.info("Turn %d done (%s): end of speech -> first audio %s ms",
                 reply.turn, reply.brain or ("fallback" if reply.local else "brain"),
                 stages.get("first_audio_played", "-"))
        self.player.forget(reply.turn)
        self._reply = None
        self._check_echo()
        self._enter_followup()

    # ================================================================== AEC health
    def _check_echo(self) -> None:
        mon = self.echo_monitor
        if mon is None or self.barge_mode is not BargeInMode.SPEECH:
            return
        erle = mon.erle_db
        if erle is not None and mon.samples >= self.s.erle_min_frames:
            log.info("Echo cancellation: %.1f dB ERLE while speaking", erle)
            if erle < self.s.min_erle_db:
                self._degrade(f"echo cancellation only reaches {erle:.1f} dB")

    def _degrade(self, reason: str) -> None:
        if self.barge_mode is not BargeInMode.SPEECH:
            return
        self.barge_mode = BargeInMode.WAKE_WORD
        self.degrade_reason = reason
        msg = (f"Barge-in switched to the wake word only: {reason}. "
               "Say the wake word to interrupt ROCO.")
        log.warning(msg)
        print(msg, flush=True)

    # ================================================================== link events
    def _on_link_up(self, welcome: Welcome) -> None:
        self.brain_name = welcome.brain
        if welcome.tts_rate and welcome.tts_rate != self.player.rate:
            out = self.player.output
            if out is not None and hasattr(out, "reopen") and not self.player.playing:
                out.reopen(welcome.tts_rate)
        fb = self._fallback
        if fb is not None and self.link is not None and self.link.connects > 1:
            line = None
            with contextlib.suppress(Exception):
                line = fb.brain_restored_line()
            if line:
                threading.Thread(target=self._speak_local, args=(line,), daemon=True).start()

    def _on_link_down(self, reason: str) -> None:
        with self._lock:
            reply = self._reply
            if reply is not None and not reply.local and reply.audio_t is None \
                    and not reply.done:
                log.warning("Link lost before turn %d was answered: asking the fallback",
                            reply.turn)
                self._start_fallback(reply)
            elif reply is not None and not reply.local:
                self.player.finish(reply.turn)  # play what arrived, then move on
                reply.done = True
            rec = self._rec
            if rec is not None:
                rec.local = True

    # ================================================================== fallback
    def _start_fallback(self, reply: _Reply) -> None:
        reply.local = True
        if self.fallback_factory is None:
            log.error("Brain unreachable and no fallback configured")
            reply.done = True
            self.player.finish(reply.turn)
            return
        threading.Thread(target=self._fallback_turn, args=(reply.turn, reply.pcm),
                         name="roco-fallback", daemon=True).start()

    def _get_fallback(self) -> Any:
        with self._fallback_lock:
            if self._fallback is None and self.fallback_factory is not None:
                self._fallback = self.fallback_factory()
            return self._fallback

    def _get_tts(self) -> TTS | None:
        with self._fallback_lock:
            if self._local_tts is None and self.local_tts_factory is not None:
                try:
                    self._local_tts = self.local_tts_factory()
                except Exception:
                    log.exception("Local TTS unavailable")
                    self.local_tts_factory = None
            return self._local_tts

    def _fallback_turn(self, turn: int, pcm: bytes) -> None:
        fb = None
        index = [0]

        def on_sentence(text: str) -> None:
            if self.player.is_cancelled(turn) or not text.strip():
                return
            if index[0] == 0 and fb is not None:
                emotion = getattr(fb, "last_emotion", None)
                if emotion:
                    self.bus.set_emotion(emotion)
            self._speak_into(turn, index[0], text)
            index[0] += 1

        try:
            fb = self._get_fallback()
            log.info("Turn %d answered by the fallback (%s)", turn, getattr(fb, "name", "?"))
            fb.respond_audio(pcm, self.s.sample_rate, on_sentence)
        except Exception:
            log.exception("Fallback failed")
            line = getattr(fb, "offline_line", "") if fb is not None else ""
            if line and index[0] == 0:
                on_sentence(line)
        finally:
            self._fallback_spoke = True
            with self._lock:
                reply = self._reply
                if reply is not None and reply.turn == turn:
                    reply.done = True
                    reply.brain = getattr(fb, "name", "fallback")
            self.player.finish(turn)

    def _speak_into(self, turn: int, index: int, text: str) -> None:
        tts = self._get_tts()
        log.info("ROCO (local): %s", text)
        if tts is None:
            return
        self.player.set_text(turn, index, text)
        pending: tuple[bytes, int] | None = None
        for chunk in tts(text):
            if self.player.is_cancelled(turn):
                return
            if pending is not None:
                self.player.feed(turn, index, pending[0], pending[1], end=False)
            pending = chunk
        if pending is not None:
            self.player.feed(turn, index, pending[0], pending[1], end=True)

    def _speak_local(self, text: str) -> None:
        with self._lock:
            turn = self._next_local
            self._next_local -= 1
            self._announcements.add(turn)
        self._speak_into(turn, 0, text)
        self.player.finish(turn)

    def _cancel_fallback(self) -> None:
        fb = self._fallback
        if fb is not None and hasattr(fb, "cancel"):
            with contextlib.suppress(Exception):
                fb.cancel()
