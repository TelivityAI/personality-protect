"""Logo renderer tests — match brand/telivity-logo reference behavior."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest

from personality_protect.logo import render_logo, resolve_mode

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "brand" / "telivity-logo"
REF_PLAIN = ROOT / "brand" / "telivity-logo.txt"


def _run_ref(*args: str) -> str:
    return subprocess.check_output([str(REF), *args], text=True)


def test_plain_matches_reference_file():
    got = render_logo("plain", mark_only=False)
    expected = REF_PLAIN.read_text(encoding="utf-8")
    if not expected.endswith("\n"):
        expected += "\n"
    assert got == expected


def test_plain_matches_reference_script():
    assert render_logo("plain") == _run_ref("--plain")


def test_truecolor_matches_reference_script(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert render_logo("truecolor", color="always") == _run_ref("--truecolor")


def test_mark_only_truecolor_matches_reference(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert render_logo("truecolor", mark_only=True, color="always") == _run_ref(
        "--truecolor", "--mark-only"
    )
    assert "Telivity" not in render_logo(
        "truecolor", mark_only=True, color="always"
    )


def test_mark_only_plain_has_no_wordmark():
    out = render_logo("plain", mark_only=True)
    assert "Telivity" not in out


def test_no_ansi_in_plain():
    out = render_logo("plain")
    assert "\033" not in out


def test_auto_plain_when_not_tty():
    buf = io.StringIO()
    assert resolve_mode("auto", stream=buf) == "plain"
    out = render_logo("auto", stream=buf)
    assert "\033" not in out


def test_no_color_env_forces_plain(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    buf = io.StringIO()
    buf.isatty = lambda: True  # type: ignore[method-assign]
    assert resolve_mode("auto", stream=buf) == "plain"
    out = render_logo("truecolor", stream=buf, color="auto")
    assert "\033" not in out


def test_term_dumb_forces_plain(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    buf = io.StringIO()
    buf.isatty = lambda: True  # type: ignore[method-assign]
    assert resolve_mode("auto", stream=buf) == "plain"


def test_each_colored_line_resets(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    out = render_logo("truecolor", color="always")
    for line in out.splitlines():
        if "\033[" in line:
            # Every styled line includes a reset (trailing pad spaces may follow).
            assert "\033[0m" in line


def test_no_color_wins_over_always(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    out = render_logo("truecolor", color="always")
    assert "\033" not in out
