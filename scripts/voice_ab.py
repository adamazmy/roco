"""Voice A/B (spec §3): the same test lines through every TTS candidate, plus timings.

    python scripts/voice_ab.py                     # all candidates, WAVs in ~/.roco/voice_ab
    python scripts/voice_ab.py --threads 4         # limit onnxruntime threads (Pi 5 has 4 cores)
    python scripts/voice_ab.py --only kokoro_emma cori_tuned_fx

For each candidate it writes one WAV per line (``<candidate>_<n>.wav``) and all lines
joined (``<candidate>_all.wav``), then prints and saves (``results.json``):

- load time (model load + first, slow inference),
- time to synthesize the FIRST SENTENCE of each line, warm (median and max), which is what
  the spec's < 300 ms target is about: the brain streams sentence by sentence,
- real-time factor (synthesis time / audio duration).

Needs the voice extra; Kokoro candidates need ``pip install kokoro-onnx`` and the model files
in ``~/.roco/models/kokoro`` (they are skipped otherwise). Play a file on Windows with:
    (New-Object Media.SoundPlayer "<path>.wav").PlaySync()
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roco.paths import roco_home  # noqa: E402
from roco.voice.models import piper_dir  # noqa: E402
from roco.voice.postfx import PostFx, PostFxSettings, ProcessedTTS  # noqa: E402
from roco.voice.tts import PiperTTS, Synthesis  # noqa: E402

TTS = Callable[[str], Iterable[tuple[bytes, int]]]

LINES = [
    "Morning, handsome. You've got a lecture at ten and you look like you've had four hours "
    "of sleep.",
    "Phone. Down. I can literally see you.",
    "Done. Reminder set for six. You're welcome, as always.",
    "It's one in the morning. Even I'm tired, and I don't have a body.",
    "Tomorrow you've got the dentist at half past three, then football with Goose at seven.",
]

CORI = "en_GB-cori-high"


def kokoro_dir() -> Path:
    return roco_home() / "models" / "kokoro"


def make_candidates(threads: int | None, int8: bool) -> dict[str, Callable[[], Any]]:
    """name -> factory building a TTS callable (lazily, so missing deps only skip one)."""
    onnx, cfg = piper_dir() / f"{CORI}.onnx", piper_dir() / f"{CORI}.onnx.json"

    def piper(length: float, noise: float, noise_w: float, fx: bool) -> Callable[[], Any]:
        def build() -> Any:
            tts = PiperTTS(onnx, cfg, Synthesis(length, noise, noise_w))
            return ProcessedTTS(tts, PostFx(PostFxSettings())) if fx else tts

        return build

    def kokoro(voice: str, fx: bool = False) -> Callable[[], Any]:
        def build() -> Any:
            from roco.voice.kokoro_tts import KOKORO_MODEL, KOKORO_VOICES, KokoroTTS

            model = "kokoro-v1.0.int8.onnx" if int8 else KOKORO_MODEL
            tts = KokoroTTS(kokoro_dir() / model, kokoro_dir() / KOKORO_VOICES, voice,
                            threads=threads)
            return ProcessedTTS(tts, PostFx(PostFxSettings(eq=False))) if fx else tts

        return build

    return {
        # Reference: what ROCO uses today (Phase 2 smoothing, no post-processing).
        "cori_current": piper(1.0, 0.5, 0.333, fx=False),
        # (a) spec §3: slower, less jitter, light EQ/compression/room.
        "cori_tuned_fx": piper(1.08, 0.4, 0.25, fx=True),
        # (b) Kokoro British voices.
        "kokoro_emma": kokoro("bf_emma"),
        "kokoro_isabella": kokoro("bf_isabella"),
    }


def first_sentence(line: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", line.strip(), maxsplit=1)
    return parts[0]


def synth(tts: TTS, text: str) -> tuple[bytes, int, float]:
    start = time.perf_counter()
    chunks = list(tts(text))
    elapsed = time.perf_counter() - start
    rate = chunks[0][1] if chunks else 22050
    return b"".join(pcm for pcm, _ in chunks), rate, elapsed


def match_loudness(pcm: bytes, target_rms: float = 0.1, peak: float = 0.95) -> bytes:
    """Same loudness for every candidate in the listening files (louder sounds "better",
    which would bias the A/B). Timings and the real speaker are unaffected."""
    import numpy as np

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
    voiced = x[np.abs(x) > 0.01]
    rms = float(np.sqrt(np.mean(voiced**2))) if len(voiced) else 0.0
    if rms <= 0:
        return pcm
    gain = min(target_rms / rms, peak / max(float(np.abs(x).max()), 1e-9))
    return (x * gain * 32767.0).astype(np.int16).tobytes()


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)


def run_candidate(name: str, build: Callable[[], Any], out: Path, runs: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    tts = build()
    load = getattr(tts, "load", None)
    if callable(load):
        load()
    synth(tts, "Hello.")  # first inference is slow (onnxruntime graph optimisation)
    load_s = time.perf_counter() - t0

    first_ms: list[float] = []
    total_synth = total_audio = 0.0
    joined: list[bytes] = []
    rate = 22050
    for i, line in enumerate(LINES, 1):
        pcm, rate, elapsed = synth(tts, line)
        total_synth += elapsed
        total_audio += len(pcm) / 2 / rate
        pcm = match_loudness(pcm)
        write_wav(out / f"{name}_{i}.wav", pcm, rate)
        joined += [pcm, b"\x00\x00" * int(rate * 0.5)]
        sentence = first_sentence(line)
        times = [synth(tts, sentence)[2] * 1000 for _ in range(runs)]
        first_ms.append(statistics.median(times))
    write_wav(out / f"{name}_all.wav", b"".join(joined), rate)
    return {
        "load_s": round(load_s, 2),
        "first_sentence_ms_median": round(statistics.median(first_ms)),
        "first_sentence_ms_max": round(max(first_ms)),
        "first_sentence_ms": [round(x) for x in first_ms],
        "rtf": round(total_synth / total_audio, 3) if total_audio else None,
        "sample_rate": rate,
        "all_wav": str(out / f"{name}_all.wav"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=roco_home() / "voice_ab")
    ap.add_argument("--runs", type=int, default=3, help="timing runs per first sentence")
    ap.add_argument("--threads", type=int, default=None, help="onnxruntime threads (Kokoro)")
    ap.add_argument("--int8", action="store_true", help="Kokoro int8 model (Pi-sized)")
    ap.add_argument("--only", nargs="*", help="candidate names to run")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for name, build in make_candidates(args.threads, args.int8).items():
        if args.only and name not in args.only:
            continue
        try:
            results[name] = run_candidate(name, build, args.out, args.runs)
        except (ImportError, FileNotFoundError) as exc:
            print(f"{name}: skipped ({exc})")
            continue
        r = results[name]
        print(f"{name:16s} load {r['load_s']:5.2f} s | first sentence median "
              f"{r['first_sentence_ms_median']:4d} ms, max {r['first_sentence_ms_max']:4d} ms "
              f"| RTF {r['rtf']}")
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"WAVs and results.json in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
