"""Face logic without pygame: emotion table, easing, blinking, animator."""

from __future__ import annotations

import random
from dataclasses import fields

import pytest

from roco.body.expression import (
    EMOTIONS,
    BlinkScheduler,
    EyeShape,
    FaceAnimator,
    FaceParams,
    approach,
    ease_in_out_cubic,
    lerp_dataclass,
    target_params,
)
from roco.body.face import (
    EyeBox,
    FaceSettings,
    eye_boxes,
    lower_lid_edge,
    parse_color,
    upper_lid_edge,
)
from roco.bus import Emotion, State


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# --------------------------------------------------------------------------- table


def test_every_emotion_has_parameters():
    assert set(EMOTIONS) == set(Emotion)
    for emotion, params in EMOTIONS.items():
        assert isinstance(params, FaceParams), emotion
        for eye in (params.left, params.right):
            assert 0.5 <= eye.width <= 1.5 and 0.5 <= eye.height <= 1.5
            assert 0 <= eye.upper < 1 and 0 <= eye.lower < 1
            assert eye.upper + eye.lower + eye.lower_curve < 1, f"{emotion} eye fully covered"


def test_emotions_are_distinct():
    shapes = {(p.left, p.right, p.gaze_x, p.gaze_y) for p in EMOTIONS.values()}
    assert len(shapes) == len(Emotion)


def test_emotion_character():
    neutral, happy = EMOTIONS[Emotion.NEUTRAL], EMOTIONS[Emotion.HAPPY]
    assert happy.left.lower_curve > 0 and neutral.left.lower_curve == 0  # upturned lower lids
    assert EMOTIONS[Emotion.SLEEPY].left.upper >= 0.4  # half closed
    surprised = EMOTIONS[Emotion.SURPRISED].left
    assert surprised.height > 1.1 and surprised.width > 1.0  # wide
    assert EMOTIONS[Emotion.SAD].left.upper_tilt > 0  # outer corners droop
    for asymmetric in (Emotion.CURIOUS, Emotion.CONFUSED):
        p = EMOTIONS[asymmetric]
        assert p.left != p.right
        assert p.left.height != p.right.height


def test_state_modifiers():
    base = EMOTIONS[Emotion.NEUTRAL]
    listening = target_params(State.LISTENING, Emotion.NEUTRAL)
    assert listening.scale > 1 and listening.glow > 0
    assert target_params(State.THINKING, Emotion.NEUTRAL).gaze_y < 0  # glance up
    assert target_params(State.SPEAKING, Emotion.HAPPY).mouth == 1.0
    assert target_params(State.IDLE, Emotion.HAPPY).mouth == 0.0
    assert target_params(State.SLEEPING, Emotion.HAPPY).openness == 0.0
    assert target_params(State.IDLE, Emotion.NEUTRAL) == base


# --------------------------------------------------------------------------- easing


def test_easing_endpoints_and_monotonic():
    assert ease_in_out_cubic(0) == 0 and ease_in_out_cubic(1) == 1
    assert ease_in_out_cubic(-1) == 0 and ease_in_out_cubic(2) == 1
    values = [ease_in_out_cubic(i / 20) for i in range(21)]
    assert values == sorted(values)
    assert ease_in_out_cubic(0.5) == pytest.approx(0.5)


def test_approach_is_frame_rate_independent():
    one_big = approach(0.0, 1.0, 0.2, rate=10)
    many_small = 0.0
    for _ in range(20):
        many_small = approach(many_small, 1.0, 0.01, rate=10)
    assert one_big == pytest.approx(many_small)
    assert approach(0.3, 1.0, 0.0, 10) == 0.3


def test_lerp_dataclass_nested():
    a = FaceParams(left=EyeShape(height=1.0), gaze_x=0.0)
    b = FaceParams(left=EyeShape(height=2.0), gaze_x=1.0)
    mid = lerp_dataclass(a, b, 0.5)
    assert mid.left.height == pytest.approx(1.5) and mid.gaze_x == pytest.approx(0.5)
    assert lerp_dataclass(a, b, 0) == a and lerp_dataclass(a, b, 1) == b


# --------------------------------------------------------------------------- blinking


def test_blink_interval_within_bounds():
    for seed in range(20):
        blinker = BlinkScheduler(random.Random(seed), 3, 6)
        assert blinker.amount(100.0) == 0.0
        assert 103.0 <= blinker.next_blink <= 106.0


def test_blink_closes_then_opens():
    blinker = BlinkScheduler(random.Random(1), 3, 6, duration=0.2, double_chance=0)
    blinker.amount(0.0)
    start = blinker.next_blink
    samples = [blinker.amount(start + i * 0.01) for i in range(21)]
    peak = max(samples)
    assert peak == pytest.approx(1.0, abs=0.05)
    assert samples.index(peak) < 12  # closes faster than it opens
    assert blinker.amount(start + 0.25) == 0.0
    assert 3.0 <= blinker.next_blink - (start + 0.2) <= 6.0  # next one scheduled


def test_blink_double_blink():
    blinker = BlinkScheduler(random.Random(0), 3, 6, duration=0.1, double_chance=1.0)
    blinker.amount(0.0)
    first = blinker.next_blink
    blinker.amount(first + 0.15)  # past the first blink
    assert blinker.next_blink == pytest.approx(first + 0.1 + 0.09)


def test_blink_bad_interval():
    with pytest.raises(ValueError):
        BlinkScheduler(min_interval=5, max_interval=2)


# --------------------------------------------------------------------------- animator


def make_animator(t: float = 0.0) -> tuple[FaceAnimator, FakeClock]:
    clock = FakeClock(t)
    animator = FaceAnimator(clock=clock, rng=random.Random(3), transition_s=0.3)
    animator.blinker.next_blink = 1e9
    return animator, clock


def test_animator_eases_between_emotions():
    animator, clock = make_animator()
    animator.update()
    animator.set_emotion(Emotion.SURPRISED)
    start = animator.update().params.left.height
    clock.t += 0.15
    mid = animator.update().params.left.height
    clock.t += 0.5
    end = animator.update().params.left.height
    target = EMOTIONS[Emotion.SURPRISED].left.height
    assert start == pytest.approx(1.0)
    assert 1.0 < mid < target
    assert end == pytest.approx(target)


def test_animator_retarget_mid_transition_has_no_jump():
    animator, clock = make_animator()
    animator.set_emotion(Emotion.SURPRISED)
    clock.t += 0.15
    before = animator.update().params.left.height
    animator.set_emotion(Emotion.SAD)
    after = animator.update().params.left.height
    assert after == pytest.approx(before)


def test_animator_mouth_only_while_speaking():
    animator, clock = make_animator()
    animator.set_mouth(1.0)
    for _ in range(30):
        clock.t += 1 / 30
        frame = animator.update()
    assert frame.mouth_open == pytest.approx(0.0)
    animator.set_state(State.SPEAKING)
    for _ in range(30):
        clock.t += 1 / 30
        frame = animator.update()
    assert frame.mouth_open > 0.95 and frame.params.mouth == pytest.approx(1.0)
    animator.set_mouth(5.0)  # clamped
    assert animator._mouth_target == 1.0


def test_animator_sleeping_closes_eyes_and_breathes():
    animator, clock = make_animator()
    animator.set_state(State.SLEEPING)
    bobs = []
    for _ in range(150):
        clock.t += 1 / 30
        frame = animator.update()
        bobs.append(frame.bob)
    assert frame.sleeping and frame.eye_openness() == pytest.approx(0.0)
    assert max(bobs) > 0.01 and min(bobs) < -0.01
    assert frame.blink == 0.0


def test_animator_blink_reduces_openness():
    animator, clock = make_animator()
    animator.blinker.next_blink = 1.0
    clock.t = 1.0 + animator.blinker.duration * 0.4
    assert animator.update().eye_openness() < 0.1


def test_setters_are_thread_safe_and_cheap():
    import threading

    animator, _ = make_animator()

    def hammer():
        for i in range(500):
            animator.set_emotion(list(Emotion)[i % len(Emotion)])
            animator.set_state(list(State)[i % len(State)])
            animator.set_mouth(i / 500)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(200):
        animator.update()
    for t in threads:
        t.join()
    assert animator.update() is not None


# --------------------------------------------------------------------------- geometry


def test_eye_boxes_scale_with_window():
    animator, _ = make_animator()
    frame = animator.update()
    small = eye_boxes(frame, 800, 480)
    big = eye_boxes(frame, 1600, 960)
    assert big[0].w == pytest.approx(small[0].w * 2)
    assert small[0].cx < 400 < small[1].cx
    assert small[0].outer == -1 and small[1].outer == 1


def test_lid_tilt_mirrors_between_eyes():
    shape = EyeShape(upper=0.2, upper_tilt=20)
    left = EyeBox(200, 200, 100, 100, outer=-1)
    right = EyeBox(600, 200, 100, 100, outer=1)
    # Sad droop: the outer corner of each eye is lower (larger y) than the inner one.
    assert upper_lid_edge(left, shape, 150) > upper_lid_edge(left, shape, 250)
    assert upper_lid_edge(right, shape, 650) > upper_lid_edge(right, shape, 550)


def test_happy_lower_lid_bulges_up_in_the_middle():
    shape = EMOTIONS[Emotion.HAPPY].left
    box = EyeBox(200, 200, 100, 100, outer=-1)
    assert lower_lid_edge(box, shape, 200) < lower_lid_edge(box, shape, 155)


def test_face_settings_from_config(config):
    s = FaceSettings.from_config(config)
    assert (s.width, s.height, s.fps) == (800, 480, 30)
    assert s.eye_color == (0, 0xE5, 0xFF) and s.background == (0, 0, 0)
    s2 = FaceSettings.from_config({"width": 1024, "eye_color": "#f80"})
    assert s2.width == 1024 and s2.eye_color == (255, 136, 0)
    assert parse_color("nope", (1, 2, 3)) == (1, 2, 3)
    assert parse_color([0, 300, -4]) == (0, 255, 0)
    assert {f.name for f in fields(FaceSettings)} >= {"fullscreen", "fps"}
