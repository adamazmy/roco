from __future__ import annotations

import threading

import httpx
import pytest

from roco import cli, doctor
from roco.body.base import Body, NullFace
from roco.brain.persona import system_prompt
from roco.bus import Emotion, State
from roco.runtime import run_with_face
from roco.voice.base import NullSpeaker


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("run", "chat", "doctor", "auth", "download-models", "demo-face"):
        assert command in out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == cli.EXIT_FAIL
    assert "usage" in capsys.readouterr().out


# run and download-models (workstream B) have their own tests in test_voice_models.py:
# with the voice extra installed, download-models would really hit the network.
@pytest.mark.parametrize("argv", [["chat"], ["auth", "google"], ["demo-face"]])
def test_stubs_exit_cleanly(argv):
    # Stubs return EXIT_NOT_IMPLEMENTED until their workstream lands; afterwards
    # these commands need real hardware/services and get their own tests.
    code = cli.main(argv)
    assert code in (cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_NOT_IMPLEMENTED)


def test_doctor_reports_unreachable_ollama(monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(doctor.httpx, "get", refuse)
    assert cli.main(["doctor"]) == cli.EXIT_FAIL
    out = capsys.readouterr().out
    assert "[FAIL] Ollama reachable" in out
    assert "ollama.com" in out


def test_doctor_passes_with_model_pulled(monkeypatch, config):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "qwen2.5:7b"}]}

    monkeypatch.setattr(doctor.httpx, "get", lambda *a, **k: Resp())
    results = doctor.run_checks(config, [doctor.check_python, doctor.check_ollama])
    assert all(r.ok for r in results)


def test_body_maps_bus_events_to_face(bus):
    face = NullFace()
    body = Body(face)
    body.attach(bus)
    bus.set_state(State.LISTENING)
    bus.set_emotion(Emotion.CURIOUS)
    bus.set_mouth(0.7)
    assert (face.state, face.emotion, face.mouth) == (State.LISTENING, Emotion.CURIOUS, 0.7)
    body.close()
    bus.set_state(State.SLEEPING)
    assert face.state == State.LISTENING


def test_run_with_face_stops_face_when_worker_ends():
    face = NullFace()
    ran = threading.Event()
    run_with_face(face, lambda stop: ran.set())
    assert ran.is_set()


def test_run_with_face_signals_worker_when_face_closes():
    face = NullFace()
    stopped = threading.Event()

    def worker(stop):
        face.stop()  # simulate the window being closed
        if stop.wait(2):
            stopped.set()

    run_with_face(face, worker)
    assert stopped.is_set()


def test_null_speaker_ignores_blank():
    s = NullSpeaker()
    s.say("Hello.")
    s.say("  ")
    assert s.spoken == ["Hello."] and s.wait_done() and not s.is_speaking


def test_system_prompt_mentions_rules():
    prompt = system_prompt()
    assert "Adam" in prompt and "[happy]" in prompt
    for emotion in Emotion:
        assert emotion.value in prompt
    assert "confirm" in prompt and "markdown" in prompt
