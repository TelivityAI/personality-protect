"""Portable detached process launch for long unattended runs.

A writer LoRA train is a multi-hour job, and running it in the foreground of a
shell ties its lifetime to that shell: closing the terminal, or a tool that
interrupts the command it started, takes the train down with it and the run is
lost with no checkpoint to resume from.

The usual shell answer, ``setsid``, is a util-linux binary and is **not present
on macOS** — a launcher that reaches for it dies before Python ever starts, and
the failure looks like "nothing trained" rather than "launcher broken". Python's
own ``start_new_session=True`` does the same thing (``setsid(2)``) on every
POSIX platform, so the detach happens in-process with nothing to shell out to.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def timestamped_log_path(directory: Path, prefix: str) -> Path:
    """UTC-stamped log file path (directory created on demand)."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{prefix}_{stamp}.log"


def spawn_detached(
    argv: Sequence[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    popen: Any = subprocess.Popen,
) -> dict[str, Any]:
    """Start ``argv`` in its own session, streaming output to ``log_path``.

    ``start_new_session=True`` detaches the child from the caller's process
    group, so a signal sent to that group — which is what an interrupted or
    closed shell delivers — does not reach it.

    stdin is closed rather than inherited: a detached job that blocks on a
    prompt it can never receive would hang until it is killed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    child_env.update(env or {})
    # Without this, the child's prints stay block-buffered into the redirected
    # log and a healthy run is indistinguishable from a hung one.
    child_env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("wb") as handle:
        process = popen(
            list(argv),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=child_env,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )
    return {"pid": int(process.pid), "log_path": str(log_path), "argv": list(argv)}


def relaunch_self_detached(
    cli_args: Sequence[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Re-run this CLI's own command detached, minus the flag that asked for it.

    Uses ``sys.executable -m`` rather than the console-script name so the child
    lands in the same interpreter and virtualenv as the parent, whatever the
    caller's PATH happens to resolve.
    """
    argv = [sys.executable, "-m", "personality_protect.cli", *cli_args]
    return spawn_detached(argv, log_path=log_path, env=env)
