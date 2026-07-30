"""Brief→post SFT rows for the writer LoRA (not the translator path)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from personality_protect.config import ProfilePaths
from personality_protect.eval_write_holdout import mine_brief_from_holdout
from personality_protect.models import Piece, load_index
from personality_protect.prompt_write import WRITE_SYSTEM_PROMPT, build_write_user_content
from personality_protect.style_profile import load_style_profile, style_directives

WRITER_SFT_FILENAME = "writer_train.jsonl"
_POST_SOURCES = frozenset({"linkedin_post"})
_MIN_TARGET_WORDS = 50


def writer_sft_path(paths: ProfilePaths) -> Path:
    return paths.sft_dir / WRITER_SFT_FILENAME


def load_holdout_id_set(paths: ProfilePaths) -> set[str]:
    """Ids listed in the local dogfood holdout file (never indexed for eval)."""
    path = paths.root / "dogfood_holdout_ids.json"
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        ids = data.get("holdout_ids") or data.get("ids") or []
    else:
        ids = data
    return {str(piece_id) for piece_id in ids}


def piece_to_writer_example(
    piece: Piece,
    *,
    style_directives_list: list[str] | None = None,
) -> dict[str, Any] | None:
    """One chat example: lossy brief → author's post as assistant target."""
    body = (piece.text or "").strip()
    if len(body.split()) < _MIN_TARGET_WORDS:
        return None
    if piece.source not in _POST_SOURCES:
        return None
    try:
        brief = mine_brief_from_holdout(body, holdout_id=piece.id)
    except ValueError:
        return None
    user = build_write_user_content(
        topic=brief["topic"],
        points=brief["points"],
        examples=(),
        style_directives=style_directives_list or (),
    )
    return {
        "messages": [
            {"role": "system", "content": WRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": body},
        ],
        "meta": {
            "piece_id": piece.id,
            "source": piece.source,
            "year": piece.year,
            "word_count": len(body.split()),
            "pair_kind": "writer",
        },
    }


def build_writer_sft(
    pieces: Iterable[Piece],
    out_path: Path,
    *,
    holdout_ids: Iterable[str] = (),
    style_directives_list: list[str] | None = None,
) -> dict[str, Any]:
    """Write writer SFT JSONL; skip holdouts and unbriefable posts."""
    excluded = {str(piece_id) for piece_id in holdout_ids}
    rows: list[dict[str, Any]] = []
    skipped = 0
    for piece in pieces:
        if piece.id in excluded:
            skipped += 1
            continue
        example = piece_to_writer_example(
            piece, style_directives_list=style_directives_list
        )
        if example is None:
            skipped += 1
            continue
        rows.append(example)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "path": str(out_path),
        "examples": len(rows),
        "skipped": skipped,
        "holdouts_excluded": sorted(excluded),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def run_build_writer_sft(paths: ProfilePaths) -> dict[str, Any]:
    """Build writer SFT from all linkedin_post pieces in the corpus index.

    Uses the full index (not the year-gated selection) so recent posts past
    ``through_year`` still train the writer. Holdout ids are excluded.
    """
    paths.ensure()
    holdouts = load_holdout_id_set(paths)
    try:
        directives = style_directives(load_style_profile(paths))
    except FileNotFoundError:
        directives = []
    pieces = [
        piece
        for piece in load_index(paths.index_path)
        if piece.source in _POST_SOURCES
    ]
    if not pieces:
        raise FileNotFoundError(
            f"No linkedin_post pieces available for writer SFT under {paths.root}"
        )
    receipt = build_writer_sft(
        pieces,
        writer_sft_path(paths),
        holdout_ids=holdouts,
        style_directives_list=directives,
    )
    return receipt
