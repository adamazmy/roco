"""Audio diagnostics that print numbers, for setting up a mic and speaker.

    python -m roco.voice.diag devices            list audio devices
    python -m roco.voice.diag mic [-s 3]         record, print peak/RMS level and noise floor
    python -m roco.voice.diag say "Hello Adam."  speak through Piper, print timing
    python -m roco.voice.diag wake [-s 60]       count wake-word triggers, print max score
    python -m roco.voice.diag stt                wake-free: record one utterance, transcribe

``--input``/``--output`` override the config's devices (index or name substring).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from roco.config import Config, load_config


def _cfg(args: argparse.Namespace) -> Config:
    overrides: dict = {"voice": {}}
    if args.input is not None:
        overrides["voice"]["input_device"] = args.input
    if args.output is not None:
        overrides["voice"]["output_device"] = args.output
    return load_config(args.config, overrides)


def cmd_devices(args: argparse.Namespace) -> int:
    import sounddevice as sd

    apis = sd.query_hostapis()
    din, dout = sd.default.device
    print("idx  in out  host API      name")
    for i, d in enumerate(sd.query_devices()):
        mark = ("<in " if i == din else "") + ("<out" if i == dout else "")
        print(f"{i:3d}  {d['max_input_channels']:2d} {d['max_output_channels']:3d}  "
              f"{apis[d['hostapi']]['name'][:12]:12s}  {d['name']}  {mark}")
    return 0


def cmd_mic(args: argparse.Namespace) -> int:
    from roco.voice.audio import FRAME_SAMPLES, frame_rms, rms_to_dbfs
    from roco.voice.endpoint import NoiseFloor
    from roco.voice.mic import MicStream, device_name, find_device

    cfg = _cfg(args)
    device = find_device(cfg.get_path("voice.input_device"), "input")
    print(f"Mic: {device_name(device, 'input')} ({'default' if device is None else device})")
    mic = MicStream(device)
    print(f"Recording {args.seconds:.0f} s at {mic.capture_rate} Hz ...", flush=True)
    rms_values, peak = [], 0
    floor = NoiseFloor()
    try:
        mic.flush()
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            frame = mic.read(0.5)
            if frame is None:
                continue
            assert len(frame) == FRAME_SAMPLES
            r = frame_rms(frame)
            rms_values.append(r)
            peak = max(peak, int(abs(frame.astype("int32")).max()))
            floor.update(r)
    finally:
        mic.close()
    if not rms_values:
        print("RESULT: no audio frames received (mic blocked or wrong device)")
        return 1
    rms_values.sort()
    mean = sum(rms_values) / len(rms_values)
    p90 = rms_values[int(len(rms_values) * 0.9)]
    print(f"RESULT frames={len(rms_values)} peak={peak} ({rms_to_dbfs(peak):.1f} dBFS) "
          f"rms_mean={mean:.0f} ({rms_to_dbfs(mean):.1f} dBFS) rms_p90={p90:.0f} "
          f"rms_min={rms_values[0]:.0f} noise_floor={floor.level:.0f} overflows={mic.overflows}")
    return 0


def cmd_say_raw(args: argparse.Namespace) -> int:
    """Synthesize once, then write a WAV file or play it in one go (no streaming)."""
    import wave

    from roco.voice.factory import create_tts

    tts = create_tts(_cfg(args))
    chunks = list(tts(args.text))
    pcm, rate = b"".join(c[0] for c in chunks), chunks[0][1]
    if args.mode == "wav":
        path = args.wav
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        print(f"RESULT wav={path} seconds={len(pcm) / 2 / rate:.2f} rate={rate}")
        return 0
    import numpy as np
    import sounddevice as sd

    from roco.voice.mic import find_device

    device = find_device(_cfg(args).get_path("voice.output_device"), "output")
    sd.play(np.frombuffer(pcm, dtype=np.int16), rate, device=device, blocking=True)
    print(f"RESULT played_direct seconds={len(pcm) / 2 / rate:.2f} rate={rate}")
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    if args.mode != "stream":
        return cmd_say_raw(args)
    from roco.bus import EventBus, Topic
    from roco.voice.factory import create_speaker

    cfg = _cfg(args)
    bus = EventBus()
    mouth: list[float] = []
    bus.subscribe(Topic.MOUTH, mouth.append)
    t0 = time.monotonic()
    speaker = create_speaker(cfg, bus)
    if args.latency:
        speaker._output.latency = float(args.latency) if args.latency[0].isdigit() else args.latency
    t_loaded = time.monotonic()
    first: list[float] = []
    speaker.on_audio_start(first.append)
    speaker.say(args.text)
    ok = speaker.wait_done(60)
    t_done = time.monotonic()
    underflows = getattr(speaker._output, "underflows", "?")
    speaker.close()
    opened = [m for m in mouth if m > 0.05]
    print(f"RESULT ok={ok} load_ms={(t_loaded - t0) * 1000:.0f} "
          f"first_audio_ms={((first[0] if first else t_done) - t_loaded) * 1000:.0f} "
          f"total_ms={(t_done - t_loaded) * 1000:.0f} mouth_blocks={len(mouth)} "
          f"mouth_max={max(mouth, default=0):.2f} mouth_open_blocks={len(opened)} "
          f"underflows={underflows}")
    return 0 if ok else 1


def cmd_wake(args: argparse.Namespace) -> int:
    from roco.voice.mic import MicStream, find_device
    from roco.voice.models import oww_dir, wakeword_model_path
    from roco.voice.wakeword import OpenWakeWordDetector

    cfg = _cfg(args)
    threshold = float(cfg.get_path("voice.wake_word.threshold", 0.5))
    det = OpenWakeWordDetector(wakeword_model_path(cfg), oww_dir())
    mic = MicStream(find_device(cfg.get_path("voice.input_device"), "input"))
    print(f"Listening {args.seconds:.0f} s for '{det.name}' (threshold {threshold}) ...",
          flush=True)
    triggers, max_score, refractory = 0, 0.0, 0
    t0 = time.monotonic()
    try:
        mic.flush()
        while time.monotonic() - t0 < args.seconds:
            frame = mic.read(0.5)
            if frame is None:
                continue
            score = det.score(frame)
            max_score = max(max_score, score)
            if refractory:
                refractory -= 1
            elif score >= threshold:
                triggers += 1
                refractory = 12  # ~1 s: one utterance = one trigger
                print(f"  trigger #{triggers} at {time.monotonic() - t0:5.1f}s score={score:.2f}",
                      flush=True)
                det.reset()
    finally:
        mic.close()
    print(f"RESULT triggers={triggers} max_score={max_score:.2f} seconds={args.seconds:.0f}")
    return 0


def cmd_stt(args: argparse.Namespace) -> int:
    from roco.voice.endpoint import RecordingSettings
    from roco.voice.factory import create_transcriber
    from roco.voice.listener import VoiceListener
    from roco.voice.mic import MicStream, find_device

    cfg = _cfg(args)
    transcriber = create_transcriber(cfg)
    t = time.monotonic()
    transcriber.load()
    transcriber.warm_up()
    print(f"Whisper loaded in {(time.monotonic() - t) * 1000:.0f} ms")

    class NoWake:
        def score(self, frame: object) -> float:
            return 0.0

        def reset(self) -> None:
            pass

    mic = MicStream(find_device(cfg.get_path("voice.input_device"), "input"))
    listener = VoiceListener(
        mic, NoWake(), transcriber,
        recording=RecordingSettings.from_config(cfg.get_path("voice.recording", {}) or {}),
    )
    print("Speak now (you have 6 s to start) ...", flush=True)
    try:
        text = listener.listen_followup(6.0)
    finally:
        listener.close()
    timing = listener.last_timing
    if timing is None:
        print("RESULT heard=nothing")
        return 1
    print(f"RESULT text={text!r} audio_s={timing.audio_s:.1f} "
          f"end_detect_ms={(timing.stt_start - timing.speech_end) * 1000:.0f} "
          f"stt_ms={(timing.stt_done - timing.stt_start) * 1000:.0f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m roco.voice.diag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config")
    parser.add_argument("--input", help="input device index or name substring")
    parser.add_argument("--output", help="output device index or name substring")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices").set_defaults(func=cmd_devices)
    p = sub.add_parser("mic")
    p.add_argument("-s", "--seconds", type=float, default=3.0)
    p.set_defaults(func=cmd_mic)
    p = sub.add_parser("say")
    p.add_argument("text", nargs="?", default="Hello Adam. I'm ROCO, and this is my voice.")
    p.add_argument("--latency", help="low | high | seconds (default high)")
    p.add_argument("--mode", choices=["stream", "direct", "wav"], default="stream",
                   help="stream = ROCO's speaker; direct = play in one go; wav = write a file")
    p.add_argument("--wav", default="roco_say.wav")
    p.set_defaults(func=cmd_say)
    p = sub.add_parser("wake")
    p.add_argument("-s", "--seconds", type=float, default=60.0)
    p.set_defaults(func=cmd_wake)
    sub.add_parser("stt").set_defaults(func=cmd_stt)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
