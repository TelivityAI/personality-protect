"""Brief→post SFT rows for the writer LoRA (not the translator path).

Every row is ``(D(y), y)``: a de-voiced note in, the author's post out. The
first writer adapter was trained on rows whose brief was a verbatim extract of
its own target — median 5-gram copy ratio 1.0 — so the cheapest way to fit the
data was to echo the input, and the adapter did exactly that at generation time.
Pair construction is therefore gated, not merely built: a row that cannot be
moved far enough from its target is dropped rather than trained on.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from personality_protect.config import ProfilePaths
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.devoice import (
    MAX_PAIR_COPY_RATIO,
    DevoiceRejected,
    mine_writer_brief,
)
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
    max_copy_ratio: float = MAX_PAIR_COPY_RATIO,
) -> tuple[dict[str, Any] | None, str]:
    """One chat example: de-voiced brief → author's post as assistant target.

    Returns ``(row, reason)``. ``reason`` names why a piece was dropped so the
    receipt can report *which* constraint pair construction is losing rows to,
    rather than a single opaque skip count.
    """
    body = normalize_corpus_text(piece.text or "")
    if piece.source not in _POST_SOURCES:
        return None, "not_a_post"
    if len(body.split()) < _MIN_TARGET_WORDS:
        return None, "too_short"
    try:
        brief, report = mine_writer_brief(
            body, holdout_id=piece.id, max_copy_ratio=max_copy_ratio
        )
    except DevoiceRejected as exc:
        return None, exc.reasons[0]
    except ValueError:
        return None, "unbriefable"

    user = build_write_user_content(
        topic=brief["topic"],
        points=brief["points"],
        examples=(),
        style_directives=style_directives_list or (),
    )
    return (
        {
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
                "devoiced": True,
                # Per-row provenance for the pair audit. Ratios only — never the
                # brief or the post body.
                "brief_copy_ratio": report["brief_copy_ratio"],
                "note_copy_ratio": report["copy_ratio"],
                "brief_words": report["brief_words"],
            },
        },
        "kept",
    )


def _quantiles(values: list[float]) -> dict[str, float | None]:
    """Median and p90 of a pair metric (empty-safe)."""
    if not values:
        return {"median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "median": round(ordered[len(ordered) // 2], 4),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
        "max": round(ordered[-1], 4),
    }


def build_writer_sft(
    pieces: Iterable[Piece],
    out_path: Path,
    *,
    holdout_ids: Iterable[str] = (),
    style_directives_list: list[str] | None = None,
    max_copy_ratio: float = MAX_PAIR_COPY_RATIO,
) -> dict[str, Any]:
    """Write writer SFT JSONL; skip holdouts and pairs that stayed near ``(y, y)``."""
    excluded = {str(piece_id) for piece_id in holdout_ids}
    rows: list[dict[str, Any]] = []
    dropped: dict[str, int] = {}
    for piece in pieces:
        if piece.id in excluded:
            dropped["holdout"] = dropped.get("holdout", 0) + 1
            continue
        example, reason = piece_to_writer_example(
            piece,
            style_directives_list=style_directives_list,
            max_copy_ratio=max_copy_ratio,
        )
        if example is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        rows.append(example)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "path": str(out_path),
        "examples": len(rows),
        "skipped": sum(dropped.values()),
        "dropped_by_reason": dict(sorted(dropped.items())),
        "holdouts_excluded": sorted(excluded),
        "pair_kind": "devoiced_brief_to_post",
        "max_copy_ratio": float(max_copy_ratio),
        # The headline pair-quality numbers. Before de-voicing this sat at 1.0.
        "brief_copy_ratio": _quantiles(
            [float(row["meta"]["brief_copy_ratio"]) for row in rows]
        ),
        "note_copy_ratio": _quantiles(
            [float(row["meta"]["note_copy_ratio"]) for row in rows]
        ),
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
