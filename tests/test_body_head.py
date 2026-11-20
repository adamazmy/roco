"""Servos, head controller and RobotBody without hardware."""

from __future__ import annotations

import random
import threading

import pytest

from roco.body.base import NullFace
from roco.body.head import GESTURES, AxisConfig, HeadController, motion_point
from roco.body.robot import EMOTION_GESTURES, RobotBody, create_body
from roco.body.servos import PCA9685Servos, SimulatedServos, create_servo_backend
from roco.bus import Emotion, State

PAN = AxisConfig(channel=0, min=30, max=150, center=90)
TILT = AxisConfig(channel=1, min=60, max=120, center=90)


def make_head(**kwargs) -> tuple[HeadController, SimulatedServos]:
    servos = SimulatedServos()
    kwargs.setdefault("step_s", 0.005)
    kwargs.setdefault("time_scale", 0.05)
    head = HeadController(servos, PAN, TILT, rng=random.Random(0), **kwargs)
    return head, servos


# --------------------------------------------------------------------------- servos


def test_simulated_backend_tracks_angles():
    servos = SimulatedServos()
    assert servos.get_angle(3) is None
    servos.set_angle(3, 42)
    assert servos.get_angle(3) == 42.0 and servos.history[-1] == (3, 42.0)
    servos.release(3)
    assert 3 in servos.released
    with pytest.raises(ValueError):
        servos.set_angle(3, 181)
    servos.close()
    servos.close()  # safe twice
    assert servos.closed


def test_factory(config):
    assert isinstance(create_servo_backend(config), SimulatedServos)
    assert isinstance(create_servo_backend({"backend": "simulated"}), SimulatedServos)
    assert isinstance(create_servo_backend(None), SimulatedServos)
    with pytest.raises(ValueError):
        create_servo_backend({"backend": "warp-drive"})


def test_pca9685_falls_back_without_library(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_servokit(name, *args, **kwargs):
        if name == "adafruit_servokit":
            raise ImportError("no servokit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_servokit)
    assert isinstance(create_servo_backend({"backend": "pca9685"}), SimulatedServos)


def test_pca9685_with_fake_kit(monkeypatch):
    import sys
    import types

    class FakeServo:
        def __init__(self):
            self.angle = None
            self.pulse = None

        def set_pulse_width_range(self, lo, hi):
            self.pulse = (lo, hi)

    class FakeKit:
        def __init__(self, channels, address):
            self.address = address
            self.servo = [FakeServo() for _ in range(channels)]

    monkeypatch.setitem(sys.modules, "adafruit_servokit", types.SimpleNamespace(ServoKit=FakeKit))
    backend = create_servo_backend({"backend": "pca9685", "i2c_address": 0x41})
    assert isinstance(backend, PCA9685Servos)
    backend.set_angle(1, 100)
    assert backend._kit.address == 0x41
    assert backend._kit.servo[1].angle == 100 and backend._kit.servo[1].pulse == (500, 2500)
    assert backend.get_angle(1) == 100
    backend.close()
    assert backend._kit.servo[1].angle is None  # released


# --------------------------------------------------------------------------- head


def test_axis_clamping():
    assert PAN.clamp(10) == 30 and PAN.clamp(170) == 150 and PAN.clamp(95) == 95
    assert TILT.angle_for(100) == 120 and TILT.angle_for(-100) == 60
    inverted = AxisConfig(channel=1, min=60, max=120, center=90, invert=True)
    assert inverted.angle_for(10) == 80
    with pytest.raises(ValueError):
        AxisConfig(channel=0, min=100, max=80, center=90)
    with pytest.raises(ValueError):
        AxisConfig(channel=0, min=0, max=200, center=90)


def test_head_starts_centered_and_from_config(config):
    servos = SimulatedServos()
    head = HeadController.from_config(servos, config)
    assert servos.get_angle(0) == 90 and servos.get_angle(1) == 90
    assert head.pan.min == 30 and head.tilt.max == 120
    head.close()


def test_motion_is_eased_and_ends_on_target():
    path = [motion_point((0, 0), (20, -10), i / 10) for i in range(11)]
    assert path[0] == (0, 0) and path[-1] == pytest.approx((20, -10))
    assert motion_point((0, 0), (20, -10), 3.0) == pytest.approx((20, -10))  # clamped
    steps = [abs(b[0] - a[0]) for a, b in zip(path, path[1:], strict=False)]
    assert max(steps) < 20 / 3  # no single jump
    assert steps[0] < steps[len(steps) // 2]  # eases in


def test_every_commanded_angle_is_within_limits():
    head, servos = make_head()
    for name in GESTURES:
        if name != "look_around":
            head.play(name)
            assert head.wait_idle(5)
    head.play("look_around")
    threading.Event().wait(0.3)
    head.close()
    for channel, angle in servos.history:
        axis = PAN if channel == 0 else TILT
        assert axis.min <= angle <= axis.max


def test_sleep_gesture_goes_to_tilt_limit():
    head, servos = make_head()
    head.play("sleep")
    assert head.wait_idle(10)
    assert servos.get_angle(1) == 120  # clamped to max, i.e. head fully down
    head.close(recenter=False)


def test_gesture_is_non_blocking_and_new_one_cancels_old():
    head, servos = make_head(idle_delay_s=0)
    head.look_around()  # endless
    deadline = threading.Event()
    deadline.wait(0.05)
    assert head.current_gesture == "look_around"
    head.nod()
    assert head.wait_idle(5)  # look_around was cancelled, nod ran to completion
    assert head.history[-2:] == ["look_around", "nod"]
    assert head.angles == (90, 90)  # nod ends centred
    head.close()


def test_cancel_stops_motion():
    head, servos = make_head(time_scale=1.0)
    head.play("sleep")  # 2.5 s motion
    threading.Event().wait(0.1)
    head.cancel()
    assert head.wait_idle(1)
    tilt = servos.get_angle(1)
    assert 90 <= tilt < 120
    threading.Event().wait(0.1)
    assert servos.get_angle(1) == tilt  # really stopped
    head.close(recenter=False)


def test_idle_look_around_only_while_idle():
    head, _ = make_head(idle_delay_s=0)
    head.set_idle(True)
    threading.Event().wait(0.05)
    assert head.current_gesture == "look_around"
    head.set_idle(False)
    assert head.wait_idle(2)
    assert head.current_gesture is None
    # A one-shot gesture while idle hands back to look-around afterwards.
    head.set_idle(True)
    head.nod()
    for _ in range(200):
        if head.history[-1:] == ["look_around"] and "nod" in head.history:
            break
        threading.Event().wait(0.01)
    assert head.history[-2:] == ["nod", "look_around"]
    head.close()


def test_unknown_gesture_and_closed_head():
    head, _ = make_head()
    with pytest.raises(ValueError):
        head.play("backflip")
    head.close()
    head.close()
    head.nod()  # ignored after close
    assert head.current_gesture is None


# --------------------------------------------------------------------------- body


class RecordingHead:
    def __init__(self):
        self.played: list[str] = []
        self.idle_calls: list[bool] = []
        self.closed = False

    def play(self, name):
        self.played.append(name)

    def set_idle(self, idle):
        self.idle_calls.append(idle)

    def close(self):
        self.closed = True


def test_every_emotion_has_a_mapping():
    assert set(EMOTION_GESTURES) == set(Emotion)
    for gesture in EMOTION_GESTURES.values():
        assert gesture is None or gesture in GESTURES


def test_body_maps_events_to_face_and_gestures(bus):
    face, head = NullFace(), RecordingHead()
    body = RobotBody(face, head=head)
    body.attach(bus)
    bus.set_emotion(Emotion.HAPPY)
    bus.set_emotion(Emotion.EXCITED)
    bus.set_emotion(Emotion.CONFUSED)
    bus.set_emotion(Emotion.SAD)
    bus.set_emotion(Emotion.NEUTRAL)
    assert head.played == ["nod", "nod", "tilt", "look_down"]
    assert face.emotion is Emotion.NEUTRAL

    head.played.clear()
    bus.set_state(State.LISTENING)
    assert head.played == ["center"] and head.idle_calls[-1] is False
    bus.set_state(State.SPEAKING)
    bus.set_mouth(0.4)
    assert face.state is State.SPEAKING and face.mouth == 0.4
    bus.set_state(State.IDLE)
    assert head.idle_calls[-1] is True
    bus.set_state(State.SLEEPING)
    assert head.played[-1] == "sleep"
    bus.set_emotion(Emotion.HAPPY)  # no nodding in your sleep
    assert head.played[-1] == "sleep"
    body.close()
    assert head.closed
    bus.set_state(State.LISTENING)
    assert face.state is State.SLEEPING  # detached


def test_body_without_servos_still_drives_face(bus):
    face = NullFace()
    body = RobotBody(face)
    body.attach(bus)
    bus.set_emotion(Emotion.HAPPY)
    bus.set_state(State.THINKING)
    assert (face.emotion, face.state) == (Emotion.HAPPY, State.THINKING)
    body.close()


def test_create_body_uses_simulated_servos(config, bus):
    face = NullFace()
    body = create_body(face, config)
    assert isinstance(body.servos, SimulatedServos) and body.head is not None
    body.attach(bus)
    bus.set_emotion(Emotion.HAPPY)
    threading.Event().wait(0.05)
    assert "nod" in body.head.history
    body.close()
    assert body.servos.closed
