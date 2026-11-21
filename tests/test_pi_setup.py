"""scripts/pi_setup.sh: lint and dry run (no sudo, apt or network)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pi_setup.sh"
UNIT = REPO / "scripts" / "roco-body.service"

posix_bash = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None, reason="needs bash on Linux/macOS"
)


def test_script_and_unit_use_lf_line_endings():
    for path in (SCRIPT, UNIT):
        assert b"\r\n" not in path.read_bytes(), f"{path.name} has CRLF line endings"


def test_unit_template_runs_roco_body_and_restarts():
    text = UNIT.read_text(encoding="utf-8")
    assert "ExecStart=@VENV@/bin/roco body" in text
    assert "Restart=always" in text
    assert "WantedBy=default.target" in text


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    result = subprocess.run(["shellcheck", "-S", "style", str(SCRIPT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


@posix_bash
def test_dry_run_changes_nothing(tmp_path):
    env = {**os.environ, "ROCO_HOME": str(tmp_path / "home"), "HOME": str(tmp_path)}
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--brain-url", "ws://adam-pc:8765"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "[dry-run]" in out and "pip install -q -e" in out and "body,pi,cloud" in out
    assert "systemctl --user enable roco-body.service" in out
    assert not (tmp_path / "home" / "config.yaml").exists()
    assert not (tmp_path / ".config").exists()


@posix_bash
def test_bad_arguments_fail():
    assert subprocess.run(["bash", str(SCRIPT), "--nope"], capture_output=True).returncode == 2
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--brain-url", "http://x"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "ws://" in result.stderr
