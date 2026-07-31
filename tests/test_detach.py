"""Contoso-safe tests for the portable detached launcher.

Regression: an unattended train launched through the shell's ``setsid`` never
started on macOS, because ``setsid`` is util-linux and is not installed there.
The launch has to detach without shelling out to anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from personality_protect.detach import (
    relaunch_self_detached,
    spawn_detached,
    timestamped_log_path,
)


class _RecordingPopen:
    def __init__(self, argv, **kwargs):  # noqa: ANN001, ANN003
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4242
        _RecordingPopen.last = self


def test_detached_launch_starts_its_own_session(tmp_path: Path):
    """A new session is what survives a signal aimed at the caller's group."""
    result = spawn_detached(
        ["echo", "hi"], log_path=tmp_path / "run.log", popen=_RecordingPopen
    )
    assert _RecordingPopen.last.kwargs["start_new_session"] is True
    assert result["pid"] == 4242


def test_detached_launch_never_shells_out(tmp_path: Path):
    """No shell means no dependency on a binary macOS does not ship."""
    spawn_detached(["echo", "hi"], log_path=tmp_path / "run.log", popen=_RecordingPopen)
    kwargs = _RecordingPopen.last.kwargs
    assert "shell" not in kwargs or kwargs["shell"] is False
    assert _RecordingPopen.last.argv == ["echo", "hi"]


def test_detached_launch_closes_stdin_and_unbuffers_output(tmp_path: Path):
    spawn_detached(["echo", "hi"], log_path=tmp_path / "run.log", popen=_RecordingPopen)
    kwargs = _RecordingPopen.last.kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_relaunch_uses_the_running_interpreter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "personality_protect.detach.spawn_detached",
        lambda argv, **kwargs: {"pid": 1, "log_path": "x", "argv": list(argv)},
    )
    result = relaunch_self_detached(["train", "--writer"], log_path=tmp_path / "l.log")
    assert result["argv"][:3] == [sys.executable, "-m", "personality_protect.cli"]
    assert result["argv"][3:] == ["train", "--writer"]


def test_detached_run_really_survives_as_a_separate_session(tmp_path: Path):
    log = tmp_path / "real.log"
    spawn_detached(
        [sys.executable, "-c", "print('contoso ok')"], log_path=log, popen=subprocess.Popen
    )
    for _ in range(200):
        if log.read_text(encoding="utf-8").strip():
            break
        import time

        time.sleep(0.02)
    assert "contoso ok" in log.read_text(encoding="utf-8")


def test_log_path_is_timestamped_under_the_target_directory(tmp_path: Path):
    path = timestamped_log_path(tmp_path / "dogfood", "train")
    assert path.parent.is_dir()
    assert path.name.startswith("train_")
    assert path.suffix == ".log"
