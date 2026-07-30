"""Deterministic corpus style profile for RAG write prompts.

Computes aggregate voice stats from selected pieces and ships a fixed banned
AI-filler list into ``style_profile.json`` under the local profile root.
Contoso-safe heuristics only — no personal text in package defaults.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from personality_protect.config import ProfilePaths
from personality_protect.eval_compare import _sentence_word_counts, _word_tokens
from personality_protect.models import Piece
from personality_protect.select import selected_pieces

# Locked prompt + existing slop detectors (filter/eval). Keep lowercase.
BANNED_AI_FILLER: tuple[str, ...] = (
    "leverage",
    "delve",
    "moreover",
    "tapestry",
    "furthermore",
    "additionally",
    "synergies",
    "synergy",
    "robust",
    "unlock",
    "unlocking",
    "nestled",
    "testament",
    "vibrant",
    "cutting-edge",
    "paradigm",
    "paradigms",
    "in today's fast-paced world",
    "in today's digital world",
    "in today's landscape",
    "it is important to note",
)

_CONTRACTION_RE = re.compile(
    r"\b(?:I'm|I've|I'd|I'll|you're|you've|you'd|you'll|we're|we've|we'd|we'll|"
    r"they're|they've|they'd|they'll|it's|that's|what's|who's|there's|here's|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|"
    r"wouldn't|shouldn't|haven't|hasn't|hadn't|mustn't)\b",
    flags=re.I,
)


def style_profile_path(paths: ProfilePaths) -> Path:
    """Local profile path for the built style card (never committed)."""
    return paths.root / "style_profile.json"


def contraction_rate(text: str) -> float:
    """Share of word tokens that are common English contractions."""
    words = _word_tokens(text)
    if not words:
        return 0.0
    hits = len(_CONTRACTION_RE.findall(text or ""))
    return round(hits / len(words), 4)


def text_style_axes(text: str) -> dict[str, Any]:
    """Per-piece axes aggregated into the corpus style profile."""
    body = (text or "").strip()
    words = _word_tokens(body)
    n_words = max(1, len(words))
    sent = _sentence_word_counts(body)
    if sent:
        ordered = sorted(sent)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median = float(ordered[mid])
        else:
            median = (ordered[mid - 1] + ordered[mid]) / 2.0
    else:
        median = 0.0
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    short = sum(1 for ln in lines if len(_word_tokens(ln)) <= 8)
    short_ratio = (short / len(lines)) if lines else 0.0
    you_n = len(re.findall(r"\byou\b", body, flags=re.I))
    i_n = len(re.findall(r"\bI\b", body))
    return {
        "words": len(words),
        "median_sentence_words": round(median, 2),
        "short_line_ratio": round(short_ratio, 4),
        "contraction_rate": contraction_rate(body),
        "you_count": you_n,
        "i_count": i_n,
        "you_per_1k": round(you_n * 1000.0 / n_words, 2),
        "i_per_1k": round(i_n * 1000.0 / n_words, 2),
        "you_gt_i": you_n > i_n,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def corpus_style_stats(texts: Iterable[str]) -> dict[str, Any]:
    """Aggregate deterministic stats across corpus texts."""
    axes = [text_style_axes(t) for t in texts if (t or "").strip()]
    if not axes:
        return {
            "pieces": 0,
            "words": 0,
            "median_sentence_words": 0.0,
            "short_line_ratio": 0.0,
            "contraction_rate": 0.0,
            "you_count": 0,
            "i_count": 0,
            "you_per_1k": 0.0,
            "i_per_1k": 0.0,
            "you_gt_i": False,
        }

    total_words = sum(int(a["words"]) for a in axes)
    you_n = sum(int(a["you_count"]) for a in axes)
    i_n = sum(int(a["i_count"]) for a in axes)
    n_words = max(1, total_words)
    return {
        "pieces": len(axes),
        "words": total_words,
        "median_sentence_words": round(
            _median([float(a["median_sentence_words"]) for a in axes]), 2
        ),
        "short_line_ratio": round(
            _median([float(a["short_line_ratio"]) for a in axes]), 4
        ),
        "contraction_rate": round(
            _median([float(a["contraction_rate"]) for a in axes]), 4
        ),
        "you_count": you_n,
        "i_count": i_n,
        "you_per_1k": round(you_n * 1000.0 / n_words, 2),
        "i_per_1k": round(i_n * 1000.0 / n_words, 2),
        "you_gt_i": you_n > i_n,
    }


def build_style_profile(
    pieces: Iterable[Piece],
    *,
    banned_phrases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the style_profile.json payload from corpus pieces."""
    piece_list = list(pieces)
    stats = corpus_style_stats(p.text for p in piece_list)
    banned = list(banned_phrases) if banned_phrases is not None else list(BANNED_AI_FILLER)
    return {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "piece_ids": [p.id for p in piece_list],
        "stats": stats,
        "banned_ai_filler": banned,
    }


def save_style_profile(paths: ProfilePaths, profile: dict[str, Any]) -> Path:
    """Write style_profile.json under the local profile root."""
    paths.ensure()
    out = style_profile_path(paths)
    out.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_style_profile(paths: ProfilePaths) -> dict[str, Any]:
    path = style_profile_path(paths)
    if not path.is_file():
        raise FileNotFoundError(
            f"No style profile at {path}. Run: personality-protect build-style-profile"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_build_style_profile(paths: ProfilePaths) -> tuple[dict[str, Any], Path]:
    """Build + save style profile from the current selection."""
    pieces = selected_pieces(paths)
    if not pieces:
        raise FileNotFoundError(
            f"Selection at {paths.selection_path} resolved to 0 pieces. "
            "Run: personality-protect select"
        )
    profile = build_style_profile(pieces)
    out = save_style_profile(paths, profile)
    return profile, out
