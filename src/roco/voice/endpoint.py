"""Energy-based end-of-utterance detection with an adaptive noise floor.

Pure Python and frame-count based (one call per 80 ms frame), so it is
deterministic in tests and needs no numpy. The listener feeds it one RMS value
per frame; time is ``frames * frame_s``, which equals wall-clock time with a
real microphone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum


class NoiseFloor:
    """Tracks the background level (int16 RMS) of the room by minimum statistics.

    The floor follows the quietest frame of the last ``window`` frames (~3 s): speech
    always has gaps quieter than itself, so talking barely moves it, while a fan that
    switches on is learnt once it has run for a window. It drops at once to a quieter
    frame and climbs smoothly (``rise`` per frame) towards a louder minimum.
    """

    def __init__(
        self,
        initial: float = 200.0,
        minimum: float = 30.0,
        window: int = 40,
        rise: float = 0.2,
    ) -> None:
        self.level = float(initial)
        self.minimum = minimum
        self.rise = rise
        self._recent: deque[float] = deque(maxlen=window)

    def update(self, rms: float) -> float:
        self._recent.append(float(rms))
        target = min(self._recent)
        if target < self.level:
            self.level = target
        elif len(self._recent) == self._recent.maxlen:  # only trust a full window upwards
            self.level += self.rise * (target - self.level)
        self.level = max(self.minimum, self.level)
        return self.level


class Endpoint(StrEnum):
    CONTINUE = "continue"
    SPEECH_START = "speech_start"
    SILENCE = "silence"  # speech then enough silence: normal end
    MAX_LENGTH = "max_length"  # hit the recording cap
    NO_SPEECH = "no_speech"  # nobody started talking in time


@dataclass
class RecordingSettings:
    silence_s: float = 0.9
    max_s: float = 15.0
    margin: float = 2.5  # speech threshold = noise floor * margin
    min_threshold: float = 300.0  # never call anything below this speech (int16 RMS)
    start_frames: int = 2  # consecutive loud frames needed to count as speech
    start_timeout_s: float = 4.0  # give up if speech has not started by then

    @classmethod
    def from_config(
        cls, recording: dict, start_timeout_s: float | None = None
    ) -> RecordingSettings:
        s = cls(
            silence_s=float(recording.get("silence_s", cls.silence_s)),
            max_s=float(recording.get("max_s", cls.max_s)),
            margin=float(recording.get("noise_floor_margin", cls.margin)),
            min_threshold=float(recording.get("min_threshold", cls.min_threshold)),
            start_frames=int(recording.get("start_frames", cls.start_frames)),
            start_timeout_s=float(recording.get("start_timeout_s", cls.start_timeout_s)),
        )
        if start_timeout_s is not None:
            s.start_timeout_s = start_timeout_s
        return s


class SilenceDetector:
    """Decides, frame by frame, when an utterance has started and ended."""

    def __init__(
        self,
        settings: RecordingSettings,
        noise_floor: NoiseFloor | None = None,
        frame_s: float = 0.08,
    ) -> None:
        self.settings = settings
        self.floor = noise_floor or NoiseFloor()
        self.frame_s = frame_s
        self.frames = 0
        self.speech_started = False
        self.speech_frames = 0
        self.last_speech_frame = -1
        self._loud_run = 0
        self._silent_run = 0

    @property
    def threshold(self) -> float:
        return max(self.settings.min_threshold, self.floor.level * self.settings.margin)

    @property
    def elapsed_s(self) -> float:
        return self.frames * self.frame_s

    @property
    def speech_end_s(self) -> float | None:
        """Offset (s) of the end of the last speech frame, from the start of recording."""
        if self.last_speech_frame < 0:
            return None
        return (self.last_speech_frame + 1) * self.frame_s

    def feed(self, rms: float) -> Endpoint:
        self.frames += 1
        s = self.settings
        loud = rms >= self.threshold
        if loud:
            self._loud_run += 1
            self._silent_run = 0
            self.speech_frames += 1
            self.last_speech_frame = self.frames - 1
        else:
            self._loud_run = 0
            self._silent_run += 1
            self.floor.update(rms)  # while recording, only quiet frames teach the floor

        if not self.speech_started:
            if self._loud_run >= s.start_frames:
                self.speech_started = True
                return Endpoint.SPEECH_START
            if self.elapsed_s >= s.start_timeout_s:
                return Endpoint.NO_SPEECH
            if self.elapsed_s >= s.max_s:
                return Endpoint.NO_SPEECH
            return Endpoint.CONTINUE

        if self.elapsed_s >= s.max_s:
            return Endpoint.MAX_LENGTH
        if self._silent_run * self.frame_s >= s.silence_s:
            return Endpoint.SILENCE
        return Endpoint.CONTINUE
