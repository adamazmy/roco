"""Model paths, `roco download-models` idempotency (mocked HTTP) and doctor checks."""

from __future__ import annotations

import json
import sys
import types

import httpx
import pytest

from roco import cli, doctor
from roco.voice import checks, models

VOICE_JSON = json.dumps({"audio": {"sample_rate": 22050}}).encode()
VOICE_ONNX = b"\x08onnx-bytes" * 1000


class FakeHub:
    """httpx transport serving the Piper voice files; counts requests."""

    def __init__(self, fail_onnx: bool = False):
        self.requests: list[str] = []
        self.fail_onnx = fail_onnx

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith(".onnx.json"):
            return httpx.Response(200, content=VOICE_JSON)
        if url.endswith(".onnx"):
            if self.fail_onnx:
                return httpx.Response(503)
            return httpx.Response(200, content=VOICE_ONNX,
                                  headers={"content-length": str(len(VOICE_ONNX))})
        return httpx.Response(404)


@pytest.fixture
def hub():
    return FakeHub()


@pytest.fixture
def client(hub):
    with httpx.Client(transport=httpx.MockTransport(hub), follow_redirects=True) as c:
        yield c


def test_piper_voice_urls():
    onnx, js = models.piper_voice_urls("en_GB-alan-medium")
    assert onnx.endswith("/en/en_GB/alan/medium/en_GB-alan-medium.onnx")
    assert js == onnx + ".json"
    onnx, _ = models.piper_voice_urls("en_GB-cori-high")
    assert onnx.endswith("/en/en_GB/cori/high/en_GB-cori-high.onnx")
    onnx, _ = models.piper_voice_urls("en_GB-northern_english_male-medium")
    assert "/en/en_GB/northern_english_male/medium/" in onnx
    with pytest.raises(ValueError):
        models.piper_voice_urls("alan")


def test_model_paths_live_in_roco_home(config, isolated_home):
    onnx, js = models.piper_voice_paths(config)
    assert onnx == isolated_home / "models" / "piper" / "en_GB-cori-high.onnx"
    assert js.name == "en_GB-cori-high.onnx.json"
    names = [p.name for p in models.wakeword_files(config)]
    assert names == ["melspectrogram.onnx", "embedding_model.onnx", "hey_jarvis_v0.1.onnx"]


def test_custom_model_paths(config, tmp_path):
    config["voice"]["wake_word"]["model"] = str(tmp_path / "my_word.onnx")
    config["voice"]["tts"]["voice"] = str(tmp_path / "v.onnx")
    assert models.wakeword_model_path(config) == tmp_path / "my_word.onnx"
    assert models.piper_voice_paths(config)[1] == tmp_path / "v.onnx.json"


def test_piper_download_is_idempotent(config, hub, client):
    lines: list[str] = []
    first = models.download_piper_voice(config, printer=lines.append, client=client)
    assert first.status == "downloaded"
    onnx, js = models.piper_voice_paths(config)
    assert onnx.read_bytes() == VOICE_ONNX and json.loads(js.read_text())
    assert any("100%" in line for line in lines)  # progress was reported
    n = len(hub.requests)

    second = models.download_piper_voice(config, printer=lines.append, client=client)
    assert second.status == "present"
    assert len(hub.requests) == n  # nothing fetched again

    forced = models.download_piper_voice(config, force=True, printer=lines.append, client=client)
    assert forced.status == "downloaded" and len(hub.requests) == 2 * n


def test_failed_download_leaves_no_partial_file(config):
    hub = FakeHub(fail_onnx=True)
    with httpx.Client(transport=httpx.MockTransport(hub)) as c:
        result = models.download_piper_voice(config, printer=lambda s: None, client=c)
    assert result.status == "failed" and "503" in result.detail
    onnx, _ = models.piper_voice_paths(config)
    assert not onnx.exists()
    assert not list(onnx.parent.glob("*.part"))


def test_corrupt_json_is_redownloaded(config, hub, client):
    onnx, js = models.piper_voice_paths(config)
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(VOICE_ONNX)
    js.write_text("{truncated")
    assert models.download_piper_voice(config, printer=lambda s: None,
                                       client=client).status == "downloaded"


def _fake_openwakeword(monkeypatch, calls):
    def download_models(model_names, target_directory):
        calls.append(model_names)
        from pathlib import Path

        for stem in ["melspectrogram", "embedding_model", "silero_vad", *model_names]:
            (Path(target_directory) / f"{stem}.onnx").write_bytes(b"x")

    utils = types.ModuleType("openwakeword.utils")
    utils.download_models = download_models
    pkg = types.ModuleType("openwakeword")
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils)


def test_wakeword_download_is_idempotent(config, monkeypatch):
    calls: list = []
    _fake_openwakeword(monkeypatch, calls)
    say = lambda s: None  # noqa: E731
    assert models.download_wakeword(config, printer=say).status == "downloaded"
    assert calls == [["hey_jarvis_v0.1"]]
    assert models.download_wakeword(config, printer=say).status == "present"
    assert len(calls) == 1
    assert models.download_wakeword(config, force=True, printer=say).status == "downloaded"
    assert len(calls) == 2


def test_download_all_reports_each_step(config):
    def ok(cfg, force, printer):
        return models.DownloadResult("a", "present")

    def bad(cfg, force, printer):
        return models.DownloadResult("b", "failed", "offline")

    lines: list[str] = []
    results = models.download_all(config, printer=lines.append, steps=[ok, bad])
    assert [r.ok for r in results] == [True, False]
    assert lines == ["[OK  ] a: present", "[FAIL] b: failed (offline)"]


def test_cli_download_models_uses_force_flag(monkeypatch):
    seen = {}

    def fake_main_download(cfg, force=False):
        seen["force"] = force
        return True

    monkeypatch.setattr(models, "main_download", fake_main_download)
    assert cli.main(["download-models", "--force"]) == cli.EXIT_OK
    assert seen == {"force": True}


@pytest.mark.parametrize("argv", [["run"], ["run", "--legacy"], ["body"]])
def test_cli_run_fails_cleanly_without_voice_setup(capsys, argv):
    # CI: no voice/body extra; locally: no models in the isolated ROCO_HOME.
    assert cli.main(argv) == cli.EXIT_FAIL
    err = capsys.readouterr().err
    assert "roco download-models" in err or "pip install -e" in err


def test_voice_checks_are_registered_in_doctor():
    for check in checks.VOICE_CHECKS:
        assert check in doctor.CHECKS


def test_model_checks_point_to_download(config):
    results = doctor.run_checks(config, [checks.check_wakeword_files, checks.check_piper_voice])
    for r in results:
        assert not r.ok and "roco download-models" in r.hint


def test_device_checks_warn_only_without_voice_extra(config, monkeypatch):
    monkeypatch.setattr(checks, "_voice_installed", lambda: False)
    results = doctor.run_checks(config, [checks.check_input_device, checks.check_output_device,
                                         checks.check_piper_voice])
    assert all(r.optional and not r.ok for r in results)
    assert doctor.print_report(results)  # optional failures do not fail doctor


def test_device_checks_with_fake_sounddevice(config, monkeypatch):
    devices = [{"name": "USB Mic", "max_input_channels": 1, "max_output_channels": 0}]
    sd = types.ModuleType("sounddevice")
    sd.query_devices = lambda index=None, kind=None: (
        devices if index is None and kind is None else
        {"name": "Default " + kind} if index is None else devices[index]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(checks, "_voice_installed", lambda: True)
    mic = checks.check_input_device(config)
    assert mic.ok and "Default input" in mic.detail
    config["voice"]["input_device"] = "usb"
    assert checks.check_input_device(config).detail == "USB Mic (#0)"
    config["voice"]["output_device"] = "usb"
    spk = checks.check_output_device(config)
    assert not spk.ok and not spk.optional and "voice.output_device" in spk.hint


@pytest.mark.parametrize(
    "model,phrase",
    [("hey_jarvis", "hey jarvis"), ("hey_jarvis_v0.1", "hey jarvis"),
     ("C:/Users/adama/.roco/models/openwakeword/hey_roco.onnx", "hey roco"),
     ("models/hey_roco_v0.1.onnx", "hey roco")],
)
def test_wake_phrase_from_name_or_path(config, model, phrase):
    config["voice"]["wake_word"]["model"] = model
    assert models.wake_phrase(config) == phrase


def test_detector_loads_custom_model_path(monkeypatch, tmp_path, config):
    """A custom .onnx (e.g. hey_roco.onnx) works: openWakeWord keys scores by file stem."""
    model = tmp_path / "hey_roco.onnx"
    for f in (model, tmp_path / "melspectrogram.onnx", tmp_path / "embedding_model.onnx"):
        f.write_bytes(b"x")
    config["voice"]["wake_word"]["model"] = str(model)
    created = {}

    class FakeModel:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.resets = 0

        def predict(self, frame):
            return {"hey_roco": 0.9}

        def reset(self):
            self.resets += 1

    mod = types.ModuleType("openwakeword.model")
    mod.Model = FakeModel
    monkeypatch.setitem(sys.modules, "openwakeword", types.ModuleType("openwakeword"))
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)
    from roco.voice.wakeword import OpenWakeWordDetector

    det = OpenWakeWordDetector(models.wakeword_model_path(config), tmp_path)
    assert det.name == "hey_roco"
    assert created["wakeword_models"] == [str(model)]
    assert created["inference_framework"] == "onnx"
    assert det.score([0] * 1280) == pytest.approx(0.9)
    det.reset()
    assert det.model.resets == 1


def test_preload_llm_posts_empty_generate(monkeypatch, config):
    from roco.voice import app

    calls = []

    class Resp:
        def raise_for_status(self):
            pass

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    app._preload_llm(config).join(5)
    # localhost is pinned to IPv4: on Windows ::1 is tried first and costs ~2 s.
    assert calls == [("http://127.0.0.1:11434/api/generate",
                      {"model": "qwen2.5:7b", "keep_alive": "30m"})]


def test_preload_llm_failure_is_only_logged(monkeypatch, config):
    from roco.voice import app

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    app._preload_llm(config).join(5)  # no exception escapes the thread
