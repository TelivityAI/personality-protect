"""Telivity terminal logo renderer.

Ports brand/telivity-logo (shell visual spec) into pure Python.
No subprocess, no external rendering dependency.
"""

from __future__ import annotations

import os
import sys
from typing import Literal, TextIO

LogoMode = Literal["auto", "truecolor", "color", "plain", "ascii"]
ColorMode = Literal["auto", "always", "never"]
LogoDisplay = Literal["full", "mark", "off"]

# Telivity palette
_ORANGE = (255, 126, 30)
_RED = (238, 56, 54)
_BLUE = (21, 142, 203)
_YELLOW = (255, 183, 26)
_TEAL = (20, 184, 180)
_CYAN = (105, 221, 211)

_RESET = "\033[0m"
_BOLD = "\033[1m"


def _esc_truecolor(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"


def _esc_256(code: str) -> str:
    return f"\033[{code}m"


def _should_use_color(stream: TextIO, color: ColorMode) -> bool:
    # NO_COLOR and TERM=dumb always win (no ANSI escapes).
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "dumb") == "dumb":
        return False
    if color == "never":
        return False
    if color == "always":
        return True
    return bool(hasattr(stream, "isatty") and stream.isatty())


def resolve_mode(
    mode: LogoMode,
    *,
    stream: TextIO | None = None,
    color: ColorMode = "auto",
) -> LogoMode:
    """Resolve auto mode against TTY / NO_COLOR / COLORTERM."""
    stream = stream or sys.stdout
    if mode != "auto":
        if mode in ("truecolor", "color") and not _should_use_color(stream, color):
            return "plain"
        return mode

    if not _should_use_color(stream, color):
        return "plain"

    colorterm = os.environ.get("COLORTERM", "")
    if colorterm in ("truecolor", "24bit"):
        return "truecolor"
    return "color"


def render_logo(
    mode: LogoMode = "auto",
    *,
    mark_only: bool = False,
    stream: TextIO | None = None,
    color: ColorMode = "auto",
) -> str:
    """Return the logo as a multi-line string (including trailing newline)."""
    stream = stream or sys.stdout
    resolved = resolve_mode(mode, stream=stream, color=color)

    if resolved == "ascii":
        wordmark = "" if mark_only else "  Telivity"
        lines = [
            "       ____________      ",
            "   ___/            \\___  ",
            f"  <______      ________>{wordmark}",
            "         \\    \\         ",
            "          \\    \\        ",
            "           \\    \\       ",
            "            \\  /        ",
            "             \\/         ",
        ]
        return "\n".join(lines) + "\n"

    if resolved == "plain":
        wordmark = "" if mark_only else "  Telivity"
        lines = [
            "       ▄▄▄▄▄▄▄▄▄▄▄▄▄▄       ",
            "   ▄▄██████████████████▄▄   ",
            " ▄████████████████████████▄ ",
            f"▀██████████████████████████▀{wordmark}",
            "        ██████               ",
            "       ██████                ",
            "      ██████                 ",
            "     ██████                  ",
            "     ██▀                     ",
        ]
        return "\n".join(lines) + "\n"

    # truecolor or portable color — ANSI on every segment, reset each line
    if resolved == "truecolor":
        orange = _esc_truecolor(_ORANGE)
        red = _esc_truecolor(_RED)
        blue = _esc_truecolor(_BLUE)
        yellow = _esc_truecolor(_YELLOW)
        teal = _esc_truecolor(_TEAL)
        cyan = _esc_truecolor(_CYAN)
    else:
        orange = _esc_256("38;5;208")
        red = _esc_256("31")
        blue = _esc_256("34")
        yellow = _esc_256("33")
        teal = _esc_256("36")
        cyan = _esc_256("96")

    # Wordmark: terminal default foreground (reset + bold) for light/dark themes
    colored_wordmark = "" if mark_only else f"{_RESET}{_BOLD}  Telivity{_RESET}"

    lines = [
        f"       {orange}▄▄▄▄{red}▄▄{blue}▄▄▄▄{yellow}▄▄▄▄{_RESET}       ",
        f"   {orange}▄▄████{red}██{blue}████{yellow}██████▄▄{_RESET}   ",
        f" {orange}▄██████{red}██{blue}████{yellow}████████▄{_RESET} ",
        f"{orange}▀██████{red}██{blue}████{yellow}████████▀{_RESET}{colored_wordmark}",
        f"        {teal}██{blue}████{_RESET}           ",
        f"       {teal}████{blue}██{_RESET}            ",
        f"      {teal}████{cyan}██{_RESET}             ",
        f"     {teal}████{cyan}██{_RESET}              ",
        f"     {teal}██▀{_RESET}                  ",
    ]
    return "\n".join(lines) + "\n"


def print_logo(
    mode: LogoMode = "auto",
    *,
    mark_only: bool = False,
    file: TextIO | None = None,
    color: ColorMode = "auto",
    display: LogoDisplay = "full",
) -> None:
    """Print the logo to a stream. No-op when display is off."""
    if display == "off":
        return
    file = file or sys.stdout
    mark = mark_only or display == "mark"
    file.write(
        render_logo(mode, mark_only=mark, stream=file, color=color)
    )
    file.flush()


def should_show_logo(*, machine_readable: bool = False) -> bool:
    """Logos never appear before JSON/CSV/completions or other machine output."""
    return not machine_readable
