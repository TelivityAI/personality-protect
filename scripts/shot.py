#!/usr/bin/env python3
"""Render captured ANSI terminal bytes to a PNG on a fixed character grid.

Usage:
  script -qec "personality-protect --logo off demo" /dev/null \\
    | python3 scripts/shot.py docs/images/cli-demo.png "personality-protect demo"

Needs pillow. Fixed cell size keeps Rich box-drawing aligned.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Fixed grid — match existing docs shots (≈10×20 cell, macOS chrome).
CELL_W = 10
CELL_H = 20
PAD_X = 14
PAD_Y = 12
TITLE_H = 36
COLS = 94

BG = (18, 20, 24)
TITLE_BG = (40, 42, 48)
TITLE_EDGE = (48, 50, 56)
FG = (230, 232, 236)
DIM = (140, 146, 156)
BOLD = (255, 255, 255)
CYAN = (120, 210, 220)
GREEN = (120, 200, 140)
YELLOW = (220, 190, 100)
RED = (230, 120, 110)
MAGENTA = (200, 140, 200)
BLUE = (120, 160, 230)

# CSI / OSC / charset noise from `script` / Rich
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OTHER = re.compile(r"\x1b[@-Z\\-_]")
_ANSI_CHARSET = re.compile(r"\x1b[()][0-9A-Za-z]")


def _font(size: int = 15) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _sgr_color(code: int, fg: bool) -> tuple[int, int, int] | None:
    table = {
        30: (60, 64, 72),
        31: RED,
        32: GREEN,
        33: YELLOW,
        34: BLUE,
        35: MAGENTA,
        36: CYAN,
        37: FG,
        90: DIM,
        91: RED,
        92: GREEN,
        93: YELLOW,
        94: BLUE,
        95: MAGENTA,
        96: CYAN,
        97: BOLD,
    }
    if not fg:
        return None
    return table.get(code)


def parse_ansi(data: bytes) -> list[list[tuple[str, tuple[int, int, int], bool]]]:
    """Return rows of (char, fg, bold) cells."""
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_CHARSET.sub("", text)

    rows: list[list[tuple[str, tuple[int, int, int], bool]]] = [[]]
    fg = FG
    bold = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b":
            m = _ANSI_CSI.match(text, i)
            if m:
                seq = m.group(0)
                i = m.end()
                if seq.endswith("m"):
                    body = seq[2:-1]
                    parts = [int(x) for x in body.split(";") if x.isdigit()] if body else [0]
                    j = 0
                    while j < len(parts):
                        code = parts[j]
                        if code == 0:
                            fg, bold = FG, False
                        elif code == 1:
                            bold = True
                            if fg == FG:
                                fg = BOLD
                        elif code == 2:
                            fg = DIM
                        elif code == 22:
                            bold = False
                            if fg == BOLD:
                                fg = FG
                        elif 30 <= code <= 37 or 90 <= code <= 97:
                            color = _sgr_color(code, True)
                            if color:
                                fg = color
                        elif code == 39:
                            fg = BOLD if bold else FG
                        elif code == 38 and j + 1 < len(parts):
                            mode = parts[j + 1]
                            if mode == 2 and j + 4 < len(parts):
                                fg = (parts[j + 2], parts[j + 3], parts[j + 4])
                                j += 4
                            elif mode == 5 and j + 2 < len(parts):
                                # 256-color: approximate via grayscale/primary buckets
                                n = parts[j + 2]
                                if n < 16:
                                    basic = [
                                        (0, 0, 0),
                                        (205, 0, 0),
                                        (0, 205, 0),
                                        (205, 205, 0),
                                        (0, 0, 238),
                                        (205, 0, 205),
                                        (0, 205, 205),
                                        (229, 229, 229),
                                        (127, 127, 127),
                                        RED,
                                        GREEN,
                                        YELLOW,
                                        BLUE,
                                        MAGENTA,
                                        CYAN,
                                        BOLD,
                                    ]
                                    fg = basic[n]
                                elif n < 232:
                                    c = n - 16
                                    r = (c // 36) * 51
                                    g = ((c // 6) % 6) * 51
                                    b = (c % 6) * 51
                                    fg = (r, g, b)
                                else:
                                    v = 8 + (n - 232) * 10
                                    fg = (v, v, v)
                                j += 2
                            else:
                                j += 1
                        j += 1
                continue
            m2 = _ANSI_OTHER.match(text, i)
            if m2:
                i = m2.end()
                continue
            i += 1
            continue
        if ch == "\n":
            rows.append([])
            i += 1
            continue
        if ch == "\t":
            spaces = 4 - (len(rows[-1]) % 4)
            rows[-1].extend([(" ", fg, bold)] * spaces)
            i += 1
            continue
        if ch == "\x08":
            if rows[-1]:
                rows[-1].pop()
            i += 1
            continue
        if ord(ch) < 32:
            i += 1
            continue
        rows[-1].append((ch, fg, bold))
        i += 1

    # Drop trailing empty rows from script(1) noise
    while rows and not rows[-1]:
        rows.pop()
    return rows


def _strip_script_noise(rows: list[list[tuple[str, tuple[int, int, int], bool]]]):
    """Drop typescript headers / bare prompts that `script` sometimes emits."""
    cleaned = []
    for row in rows:
        line = "".join(c for c, _, _ in row).strip()
        if line.startswith("Script started") or line.startswith("Script done"):
            continue
        cleaned.append(row)
    return cleaned


def render(
    rows: list[list[tuple[str, tuple[int, int, int], bool]]],
    title: str,
    cols: int = COLS,
) -> Image.Image:
    rows = _strip_script_noise(rows)
    # Prepend a synthetic prompt line matching existing docs shots
    prompt = [
        (">", CYAN, True),
        (" ", FG, False),
    ]
    prompt.extend((c, BOLD if c != " " else FG, c != " ") for c in title)
    # If the capture already starts with the command, don't double it
    first = "".join(c for c, _, _ in rows[0]).strip() if rows else ""
    if not first.startswith(">") and title not in first:
        rows = [prompt] + rows

    width = PAD_X * 2 + cols * CELL_W
    height = TITLE_H + PAD_Y * 2 + max(1, len(rows)) * CELL_H
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, width, TITLE_H], fill=TITLE_BG)
    draw.line([(0, TITLE_H), (width, TITLE_H)], fill=TITLE_EDGE)
    for x, color in ((18, (255, 95, 86)), (38, (255, 189, 46)), (58, (39, 201, 63))):
        draw.ellipse([x, 12, x + 12, 24], fill=color)
    font_title = _font(13)
    tw = draw.textlength(title, font=font_title)
    draw.text(((width - tw) / 2, 10), title, fill=DIM, font=font_title)

    font = _font(15)
    y = TITLE_H + PAD_Y
    for row in rows:
        x = PAD_X
        for ch, color, _bold in row[:cols]:
            draw.text((x, y - 1), ch, fill=color, font=font)
            x += CELL_W
        y += CELL_H
    return img


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: shot.py OUT.png [TITLE]\n"
            "  reads ANSI bytes from stdin",
            file=sys.stderr,
        )
        return 2
    out = Path(argv[1])
    title = argv[2] if len(argv) > 2 else out.stem
    data = sys.stdin.buffer.read()
    rows = parse_ansi(data)
    img = render(rows, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
    print(f"wrote {out} ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
