"""``roco doctor`` checks for the voice pipeline (appended to ``roco.doctor.CHECKS``).

Without the voice extra these report optional warnings, so text-only installs
still pass ``roco doctor``. With it installed they are real failures.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from roco.config import Config
from roco.doctor import CheckResult

_VOICE_HINT = 'pip install -e ".[voice]"'


def _voice_installed() -> bool:
    return importlib.util.find_spec("sounddevice") is not None


def _audio_device_check(kind: str) -> Callable[[Config], CheckResult]:
    label = "Microphone" if kind == "input" else "Speaker"
    key = f"voice.{kind}_device"

    def check(cfg: Config) -> CheckResult:
        if not _voice_installed():
            return CheckResult(label, False, "voice extra not installed", _VOICE_HINT,
                               optional=True)
        try:
            import sounddevice as sd

            from roco.voice.audio import resolve_device

            devices = list(sd.query_devices())
            spec = cfg.get_path(key)
            index = resolve_device(spec, kind, devices)
            info = sd.query_devices(index, kind=kind) if index is None else devices[index]
        except Exception as exc:  # PortAudioError, ValueError from resolve_device
            return CheckResult(
                label, False, str(exc),
                hint=f"Plug in a {label.lower()} or set {key} in config.yaml "
                     "(run `python -m roco.voice.diag devices` to list them).",
            )
        where = "system default" if index is None else f"#{index}"
        return CheckResult(label, True, f"{info['name']} ({where})")

    check.__name__ = f"check_{kind}_device"
    return check


def check_wakeword_files(cfg: Config) -> CheckResult:
    from roco.voice.models import wakeword_files

    missing = [p.name for p in wakeword_files(cfg) if not p.is_file()]
    model = str(cfg.get_path("voice.wake_word.model"))
    return CheckResult(
        f"Wake word model ({model})", not missing,
        "missing: " + ", ".join(missing) if missing else "ok",
        hint="Run: roco download-models", optional=not _voice_installed(),
    )


def check_piper_voice(cfg: Config) -> CheckResult:
    from roco.voice.models import piper_voice_paths

    missing = [p.name for p in piper_voice_paths(cfg) if not p.is_file()]
    voice = str(cfg.get_path("voice.tts.voice"))
    return CheckResult(
        f"Piper voice ({voice})", not missing,
        "missing: " + ", ".join(missing) if missing else "ok",
        hint="Run: roco download-models", optional=not _voice_installed(),
    )


check_input_device = _audio_device_check("input")
check_output_device = _audio_device_check("output")

VOICE_CHECKS = [check_input_device, check_output_device, check_wakeword_files, check_piper_voice]
