"""Small audio helpers shared by the listener and the speaker.

Nothing here imports numpy or sounddevice at module level: the helpers accept
numpy arrays when numpy is installed, and plain ``bytes`` / ``array('h')`` /
lists otherwise, so the logic is testable in text-mode CI.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280: what openWakeWord expects
INT16_FULL_SCALE = 32768.0


def frame_rms(frame: Any) -> float:
    """RMS of one int16 frame (numpy array, bytes, array('h') or a list of ints)."""
    if isinstance(frame, bytes | bytearray | memoryview):
        frame = array("h", bytes(frame))
    if hasattr(frame, "dtype"):  # numpy array: fast path
        import numpy as np

        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    n = len(frame)
    if n == 0:
        return 0.0
    return math.sqrt(sum(int(s) * int(s) for s in frame) / n)


def rms_to_dbfs(rms: float) -> float:
    return 20 * math.log10(max(rms, 1e-9) / INT16_FULL_SCALE)


def mouth_from_rms(rms: float, full_open_rms: float = 6000.0) -> float:
    """Map playback RMS (int16 scale) to a mouth opening 0..1.

    A gentle power curve opens the mouth visibly on quiet syllables while loud
    ones saturate at 1. Piper's speech typically sits around 2000-8000 RMS.
    """
    if rms <= 0:
        return 0.0
    return min(1.0, (rms / full_open_rms) ** 0.7)


def iter_blocks(pcm: bytes, block_samples: int) -> Iterator[bytes]:
    """Split int16 mono PCM bytes into blocks of ``block_samples`` (last may be shorter)."""
    step = block_samples * 2
    for i in range(0, len(pcm), step):
        yield pcm[i : i + step]


# --------------------------------------------------------------------------- devices


def resolve_device(
    spec: int | str | None,
    kind: str,
    devices: Sequence[Mapping[str, Any]],
) -> int | None:
    """Turn a config device spec into a sounddevice index.

    - ``None`` / ``""`` -> None (system default).
    - an int, or a string of digits -> that index (validated).
    - any other string -> the first device whose name contains it (case-insensitive)
      and has channels in the right direction. On Windows the same device is listed
      once per host API (MME, DirectSound, WASAPI, WDM-KS); lower indices belong to
      MME, which resamples for us, so the first match is the safest choice.

    Raises ``ValueError`` with the list of candidates when nothing matches.
    """
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', not {kind!r}")
    channels_key = f"max_{kind}_channels"
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, bool):
        raise ValueError(f"Invalid {kind} device: {spec!r}")
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        index = int(spec)
        if not 0 <= index < len(devices):
            raise ValueError(
                f"No {kind} device with index {index}. {describe_devices(devices, kind)}"
            )
        if int(devices[index].get(channels_key, 0)) <= 0:
            raise ValueError(
                f"Device {index} has no {kind} channels. {describe_devices(devices, kind)}"
            )
        return index
    needle = str(spec).strip().lower()
    for index, dev in enumerate(devices):
        if needle in str(dev.get("name", "")).lower() and int(dev.get(channels_key, 0)) > 0:
            return index
    raise ValueError(f"No {kind} device matching {spec!r}. {describe_devices(devices, kind)}")


def describe_devices(devices: Sequence[Mapping[str, Any]], kind: str) -> str:
    channels_key = f"max_{kind}_channels"
    names = [
        f"[{i}] {d.get('name', '?')}"
        for i, d in enumerate(devices)
        if int(d.get(channels_key, 0)) > 0
    ]
    return f"Available {kind} devices: " + ("; ".join(names) if names else "none")


# --------------------------------------------------------------------------- resampling


def resample_int16(samples: Any, from_rate: int, to_rate: int) -> Any:
    """Resample a numpy int16 mono array. Used only when the mic cannot do 16 kHz.

    Integer ratios (48k -> 16k) average each group of samples, which doubles as a
    crude anti-alias filter; other ratios use linear interpolation.
    """
    import numpy as np

    if from_rate == to_rate:
        return samples
    x = samples.astype(np.float32)
    if from_rate % to_rate == 0:
        k = from_rate // to_rate
        usable = (len(x) // k) * k
        y = x[:usable].reshape(-1, k).mean(axis=1)
    else:
        n_out = int(round(len(x) * to_rate / from_rate))
        y = np.interp(
            np.linspace(0, len(x) - 1, n_out, dtype=np.float64),
            np.arange(len(x), dtype=np.float64),
            x,
        )
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)
