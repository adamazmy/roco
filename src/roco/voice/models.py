"""Where the voice models live, and ``roco download-models``.

Layout under ``roco.paths.models_dir()`` (``~/.roco/models``)::

    openwakeword/   melspectrogram.onnx, embedding_model.onnx, hey_jarvis_v0.1.onnx, ...
    piper/          en_GB-cori-high.onnx, en_GB-cori-high.onnx.json
    whisper/        faster-whisper's Hugging Face cache (base.en)
    silero/         silero_vad.onnx (v5, the body's voice activity detector)

Only ``httpx`` (a core dependency) is imported at module level; openwakeword and
faster-whisper are imported inside the download steps.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from roco.config import Config
from roco.paths import models_dir

log = logging.getLogger(__name__)

PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
OWW_FEATURE_FILES = ("melspectrogram.onnx", "embedding_model.onnx")
OWW_DEFAULT_VERSION = "v0.1"
# Silero VAD v5 (MIT). Pinned to a release tag so the file never changes under us.
SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
)

Printer = Callable[[str], None]


# --------------------------------------------------------------------------- paths


def oww_dir() -> Path:
    return models_dir() / "openwakeword"


def piper_dir() -> Path:
    return models_dir() / "piper"


def whisper_dir() -> Path:
    return models_dir() / "whisper"


def silero_vad_path() -> Path:
    """The body's VAD model. Falls back to the (older, v4) copy openWakeWord ships."""
    path = models_dir() / "silero" / "silero_vad.onnx"
    if not path.is_file() and (oww_dir() / "silero_vad.onnx").is_file():
        return oww_dir() / "silero_vad.onnx"
    return path


def _looks_like_path(value: str) -> bool:
    return value.endswith(".onnx") or os.sep in value or "/" in value


def wakeword_model_path(cfg: Config) -> Path:
    """``voice.wake_word.model``: a pretrained name (``hey_jarvis``) or a path to an .onnx."""
    name = str(cfg.get_path("voice.wake_word.model", "hey_jarvis"))
    if _looks_like_path(name):
        return Path(name).expanduser()
    if "_v" not in name:
        name = f"{name}_{OWW_DEFAULT_VERSION}"
    return oww_dir() / f"{name}.onnx"


def wake_phrase(cfg: Config) -> str:
    """Spoken wake phrase: ``hey_jarvis`` or ``.../hey_roco_v0.1.onnx`` -> "hey roco"."""
    stem = wakeword_model_path(cfg).stem
    stem = re.sub(r"_v\d+(\.\d+)*$", "", stem)
    return stem.replace("_", " ").strip()


def wakeword_files(cfg: Config) -> list[Path]:
    """Every file openWakeWord needs: the two feature models plus the wake word model."""
    return [oww_dir() / f for f in OWW_FEATURE_FILES] + [wakeword_model_path(cfg)]


def piper_voice_paths(cfg: Config) -> tuple[Path, Path]:
    """(model .onnx, config .onnx.json) for ``voice.tts.voice`` (a name or an .onnx path)."""
    voice = str(cfg.get_path("voice.tts.voice", "en_GB-cori-high"))
    onnx = Path(voice).expanduser() if _looks_like_path(voice) else piper_dir() / f"{voice}.onnx"
    return onnx, onnx.with_name(onnx.name + ".json")


def piper_voice_urls(voice: str) -> tuple[str, str]:
    """Hugging Face URLs for a voice name like ``en_GB-alan-medium``.

    Names are ``<lang>_<REGION>-<speaker>-<quality>``; the repo path is
    ``<lang>/<lang>_<REGION>/<speaker>/<quality>/``. Speakers may contain
    underscores (``northern_english_male``) but not hyphens.
    """
    parts = voice.split("-")
    if len(parts) < 3 or "_" not in parts[0]:
        raise ValueError(
            f"Not a Piper voice name: {voice!r} (expected e.g. 'en_GB-alan-medium')"
        )
    locale, quality, speaker = parts[0], parts[-1], "-".join(parts[1:-1])
    lang = locale.split("_")[0]
    base = f"{PIPER_VOICES_BASE}/{lang}/{locale}/{speaker}/{quality}/{voice}.onnx"
    return base, base + ".json"


# --------------------------------------------------------------------------- download


@dataclass
class DownloadResult:
    name: str
    status: str  # "downloaded" | "present" | "failed"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def _valid_file(path: Path, is_json: bool = False) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if is_json:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
    return True


def download_file(
    url: str,
    dest: Path,
    printer: Printer = print,
    client: httpx.Client | None = None,
) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` file, printing coarse progress.

    The final file only appears once complete, so an interrupted download is retried
    next time instead of being mistaken for a finished one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=60.0))
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)
            done, next_pct = 0, 10
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done * 100 // total >= next_pct:
                        printer(f"    {dest.name}: {done * 100 // total}% of {total / 1e6:.1f} MB")
                        next_pct += 10
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if own_client:
            client.close()


def download_wakeword(cfg: Config, force: bool = False, printer: Printer = print) -> DownloadResult:
    name = "openWakeWord models"
    files = wakeword_files(cfg)
    model = wakeword_model_path(cfg)
    if model.parent != oww_dir():
        files = files[:-1]  # a custom model path: we can only fetch the feature models
    if not force and all(_valid_file(f) for f in files):
        return DownloadResult(name, "present", str(oww_dir()))
    try:
        from openwakeword.utils import download_models
    except ImportError:
        return DownloadResult(name, "failed", 'openwakeword missing: pip install -e ".[voice]"')
    target = oww_dir()
    target.mkdir(parents=True, exist_ok=True)
    if force:
        # openwakeword skips files that exist (keyed on the .tflite name): clear them.
        stems = [Path(f).stem for f in OWW_FEATURE_FILES] + ["silero_vad", model.stem]
        for stem in stems:
            for suffix in (".onnx", ".tflite"):
                (target / f"{stem}{suffix}").unlink(missing_ok=True)
    printer(f"  Downloading openWakeWord models into {target} ...")
    try:
        download_models(model_names=[model.stem], target_directory=str(target))
    except Exception as exc:  # requests/OS errors from the library
        return DownloadResult(name, "failed", f"{exc.__class__.__name__}: {exc}")
    missing = [f.name for f in files if not _valid_file(f)]
    if missing:
        return DownloadResult(name, "failed", "still missing: " + ", ".join(missing))
    return DownloadResult(name, "downloaded", str(target))


def download_piper_voice(
    cfg: Config,
    force: bool = False,
    printer: Printer = print,
    client: httpx.Client | None = None,
) -> DownloadResult:
    voice = str(cfg.get_path("voice.tts.voice", "en_GB-cori-high"))
    name = f"Piper voice {voice}"
    onnx, config_json = piper_voice_paths(cfg)
    if not force and _valid_file(onnx) and _valid_file(config_json, is_json=True):
        return DownloadResult(name, "present", str(onnx.parent))
    if _looks_like_path(voice):
        return DownloadResult(name, "failed", f"custom voice file not found: {onnx}")
    try:
        onnx_url, json_url = piper_voice_urls(voice)
    except ValueError as exc:
        return DownloadResult(name, "failed", str(exc))
    printer(f"  Downloading {voice} into {onnx.parent} ...")
    try:
        # json first: it is small, and a valid json + valid onnx is our "present" test.
        for url, dest in ((json_url, config_json), (onnx_url, onnx)):
            download_file(url, dest, printer, client)
    except httpx.HTTPError as exc:
        return DownloadResult(name, "failed", f"{exc.__class__.__name__}: {exc}")
    if not _valid_file(config_json, is_json=True):
        return DownloadResult(name, "failed", f"{config_json.name} is not valid JSON")
    return DownloadResult(name, "downloaded", str(onnx.parent))


def download_whisper(cfg: Config, force: bool = False, printer: Printer = print) -> DownloadResult:
    model = str(cfg.get_path("voice.stt.model", "base.en"))
    name = f"faster-whisper {model}"
    if Path(model).expanduser().is_dir():
        return DownloadResult(name, "present", "local model directory")
    try:
        from faster_whisper import download_model
    except ImportError:
        return DownloadResult(name, "failed", 'faster-whisper missing: pip install -e ".[voice]"')
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cache = whisper_dir()
    if force:
        shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        download_model(model, local_files_only=True, cache_dir=str(cache))
        return DownloadResult(name, "present", str(cache))
    except Exception:
        pass  # not cached yet
    printer(f"  Downloading faster-whisper {model} into {cache} ...")
    try:
        download_model(model, cache_dir=str(cache))
    except Exception as exc:
        return DownloadResult(name, "failed", f"{exc.__class__.__name__}: {exc}")
    return DownloadResult(name, "downloaded", str(cache))


def download_silero_vad(
    cfg: Config,
    force: bool = False,
    printer: Printer = print,
    client: httpx.Client | None = None,
) -> DownloadResult:
    name = "Silero VAD"
    dest = models_dir() / "silero" / "silero_vad.onnx"
    if not force and _valid_file(dest):
        return DownloadResult(name, "present", str(dest.parent))
    printer(f"  Downloading Silero VAD into {dest.parent} ...")
    try:
        download_file(SILERO_VAD_URL, dest, printer, client)
    except httpx.HTTPError as exc:
        return DownloadResult(name, "failed", f"{exc.__class__.__name__}: {exc}")
    return DownloadResult(name, "downloaded", str(dest.parent))


def download_all(
    cfg: Config,
    force: bool = False,
    printer: Printer = print,
    steps: list[Callable[..., DownloadResult]] | None = None,
) -> list[DownloadResult]:
    results = []
    for step in steps or (download_wakeword, download_silero_vad, download_piper_voice,
                          download_whisper):
        result = step(cfg, force=force, printer=printer)
        mark = {"downloaded": "OK  ", "present": "OK  ", "failed": "FAIL"}[result.status]
        detail = f" ({result.detail})" if result.detail else ""
        printer(f"[{mark}] {result.name}: {result.status}{detail}")
        results.append(result)
    return results


def main_download(cfg: Config, force: bool = False) -> bool:
    """Entry point for ``roco download-models``. True when every model is in place."""
    for noisy in ("httpx", "httpx2", "huggingface_hub"):  # one INFO line per request
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Models directory: {models_dir()}" + ("  (--force: re-downloading)" if force else ""))
    results = download_all(cfg, force=force)
    ok = all(r.ok for r in results)
    if not ok:
        print("Some downloads failed; re-run `roco download-models` (finished files are kept).",
              file=sys.stderr)
    return ok
