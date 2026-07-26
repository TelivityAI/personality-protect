"""Build local SFT JSONL from selected writing pieces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from personality_protect.config import ProfilePaths
from personality_protect.models import Piece
from personality_protect.select import selected_pieces

SYSTEM_PROMPT = (
    "You are a personal voice rewriter. Rewrite the draft so it matches the user's "
    "authentic writing voice exactly. Keep factual meaning unchanged. Prefer their "
    "cadence, sentence rhythm, word choice, punctuation habits, and paragraph shape. "
    "Strip generic AI tells such as: leverage, synergies, delve, robust, "
    "In today's fast-paced world, It is important to note, Moreover, Furthermore, "
    "unlock, nestled, testament, vibrant. Do not invent facts, citations, or claims. "
    "Do not add hashtags, emoji, or marketing slogans unless the voice reference uses them."
)

USER_TEMPLATE = (
    "Rewrite the draft in my voice. Match my cadence and diction from the reference. "
    "Keep the same meaning. Remove AI-sounding filler.\n\n"
    "### Draft\n{draft}\n\n"
    "### My voice (reference)\n{reference}\n\n"
    "### Rewritten"
)


def piece_to_example(piece: Piece) -> dict:
    """Map a corpus piece to a supervised chat example.

    Training target is the user's real text. The "draft" side is a lightly
    neutralized prompt so the adapter learns to restore their voice.
    """
    draft = _neutral_draft(piece.text)
    user = USER_TEMPLATE.format(draft=draft, reference=_short_reference(piece.text))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": piece.text},
        ],
        "meta": {
            "piece_id": piece.id,
            "source": piece.source,
            "year": piece.year,
            "word_count": piece.word_count,
        },
    }


def _short_reference(text: str, max_chars: int = 400) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _neutral_draft(text: str) -> str:
    """Crude neutralization for SFT pairing (local heuristic, not cloud)."""
    # Keep structure; soften first-person flourishes slightly for the "draft" side.
    draft = text
    replacements = (
        ("I genuinely believe", "It seems"),
        ("I've found that", "One finds that"),
        ("In my experience,", "Often,"),
        ("Here's the thing:", "Note:"),
        ("Let me be clear:", "To clarify:"),
        ("I cut the corporate fog", "Remove corporate language"),
        ("I do not need", "One does not need"),
    )
    for a, b in replacements:
        draft = draft.replace(a, b)
    return draft


def build_sft_jsonl(pieces: Iterable[Piece], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for piece in pieces:
            if not piece.text.strip():
                continue
            fh.write(json.dumps(piece_to_example(piece), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_sft_from_profile(paths: ProfilePaths, out_path: Path | None = None) -> tuple[Path, int]:
    pieces = selected_pieces(paths)
    if not pieces:
        raise RuntimeError(
            "No selected pieces. Run ingest, then select, before building SFT."
        )
    dest = out_path or paths.sft_jsonl
    n = build_sft_jsonl(pieces, dest)
    return dest, n
