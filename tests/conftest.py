from __future__ import annotations

import pytest

from roco.bus import EventBus
from roco.config import Config, load_config


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.roco or a stray ./config.yaml during tests."""
    home = tmp_path / "roco_home"
    home.mkdir()
    monkeypatch.setenv("ROCO_HOME", str(home))
    monkeypatch.delenv("ROCO_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    return home


@pytest.fixture
def config() -> Config:
    return load_config()


@pytest.fixture
def bus() -> EventBus:
    return EventBus()
