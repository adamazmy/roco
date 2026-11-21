"""LinkClient: handshake, refusal, reconnect with backoff, ping/RTT, liveness."""

from __future__ import annotations

import random

from client_fakes import Brain, wait_for
from roco.client.link import LinkClient
from roco.protocol import Error, Hello, Ping, Pong, Sentence


def make(brain: Brain, frames: list, events: list, **kw) -> LinkClient:
    opts = dict(backoff_min_s=0.01, backoff_max_s=0.05, recv_timeout_s=0.02,
                rng=random.Random(1))
    opts.update(kw)
    return LinkClient(
        brain.connect,
        lambda: Hello(protocol=1, body_id="t", token="secret"),
        frames.append,
        on_up=lambda w: events.append(("up", w.brain)),
        on_down=lambda r: events.append(("down", r)),
        **opts,
    )


def test_handshake_then_frames_are_dispatched():
    brain, frames, events = Brain(), [], []
    link = make(brain, frames, events)
    link.start()
    try:
        assert wait_for(lambda: brain.ends)
        hello = brain.accept()
        assert hello.token == "secret"
        assert link.wait_up(2)
        assert events == [("up", "fake")]
        brain.end.send(Sentence(turn=1, index=0, text="hi"))
        assert wait_for(lambda: frames)
        assert frames[0] == Sentence(turn=1, index=0, text="hi")
        assert link.send(Pong(id=1, ts=0.0))
    finally:
        link.stop()


def test_send_while_down_returns_false():
    brain, frames, events = Brain(), [], []
    brain.fail_connect = True
    link = make(brain, frames, events)
    assert link.send(Pong(id=1, ts=0.0)) is False


def test_refused_handshake_is_retried_with_backoff():
    brain, frames, events = Brain(), [], []
    brain.refuse = Error(code="auth", message="bad token", fatal=True)
    link = make(brain, frames, events)
    link.start()
    try:
        brain_ends = []
        for _ in range(2):
            n = len(brain_ends)
            assert wait_for(lambda n=n: len(brain.ends) > n)
            brain_ends.append(brain.end)
            brain.accept()
        assert not link.is_up
        assert "auth" in link.last_error
        assert events == []
    finally:
        link.stop()


def test_reconnects_after_the_brain_restarts():
    brain, frames, events = Brain(), [], []
    link = make(brain, frames, events)
    link.start()
    try:
        assert wait_for(lambda: brain.ends)
        brain.accept()
        assert link.wait_up(2)
        brain.end.close()  # brain process dies
        assert wait_for(lambda: ("down" in [e[0] for e in events]))
        assert not link.is_up
        assert wait_for(lambda: len(brain.ends) == 2)
        brain.accept()
        assert wait_for(lambda: link.is_up)
        assert link.connects == 2
        assert [e[0] for e in events] == ["up", "down", "up"]
    finally:
        link.stop()


def test_connect_errors_keep_retrying():
    brain, frames, events = Brain(), [], []
    brain.fail_connect = True
    link = make(brain, frames, events)
    link.start()
    try:
        assert wait_for(lambda: link.attempts >= 3)
        brain.fail_connect = False
        assert wait_for(lambda: brain.ends)
        brain.accept()
        assert link.wait_up(2)
    finally:
        link.stop()


def test_ping_measures_rtt_and_answers_brain_pings():
    brain, frames, events = Brain(), [], []
    link = make(brain, frames, events, ping_interval_s=0.05)
    link.start()
    try:
        assert wait_for(lambda: brain.ends)
        brain.accept()
        ping = brain.recv()
        assert isinstance(ping, Ping)
        brain.end.send(Pong(id=ping.id, ts=ping.ts))
        assert wait_for(lambda: link.rtt_ms is not None)
        assert link.rtt_ms >= 0
        brain.end.send(Ping(id=9, ts=1.5))
        pong = brain.recv(skip=(Ping,))
        assert pong == Pong(id=9, ts=1.5)
        assert frames == []  # ping/pong never reach the body logic
    finally:
        link.stop()


def test_silent_brain_is_declared_dead():
    brain, frames, events = Brain(), [], []
    link = make(brain, frames, events, ping_interval_s=0.02, pong_timeout_s=0.1)
    link.start()
    try:
        assert wait_for(lambda: brain.ends)
        brain.accept()
        assert link.wait_up(2)
        assert wait_for(lambda: ("down", "no pong from the brain") in events)
    finally:
        link.stop()
