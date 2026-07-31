"""Deterministic article holdout carve for the article-channel eval.

Same shape as the writer carve in :mod:`personality_protect.writer_holdout`, and
deliberately so — a second selection rule would be a second thing to audit. The
one difference is the constraint that dominates at this corpus size: with
fourteen articles and a retrieval floor of five, the carve cannot be a fixed
fraction. Reserving a quarter of a fourteen-piece pool is fine; reserving a
quarter of a six-piece pool would leave the article channel below the floor it
refuses to draft under, and the eval would be measuring an empty index.

``resolve_article_holdout_n`` therefore takes the floor as an input and never
carves past it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from personality_protect.article_brief import is_article_briefable
from personality_protect.config import ProfilePaths
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.models import Piece
from personality_protect.write_article import ARTICLE_SOURCES, MIN_ARTICLE_CORPUS
from personality_protect.writer_holdout import _order_key

ARTICLE_HOLDOUT_FILENAME = "article_holdout_ids.json"

# Four, not three. A one-sided sign test on three decisive comparisons bottoms
# out at p=0.125 even when one arm sweeps, so a carve of three cannot clear a
# 0.10 bar under any outcome — it is a run that has agreed in advance not to
# produce a result. Four reaches p=0.0625 on a sweep. Five is where a
# fourteen-article corpus starts eating the retrieval floor.
MIN_ARTICLE_HOLDOUT_N = 4
MAX_ARTICLE_HOLDOUT_N = 5
DEFAULT_ARTICLE_HOLDOUT_FRACTION = 0.3


def resolve_article_holdout_n(
    briefable: int,
    *,
    total_articles: int,
    fraction: float = DEFAULT_ARTICLE_HOLDOUT_FRACTION,
    minimum: int = MIN_ARTICLE_HOLDOUT_N,
    maximum: int = MAX_ARTICLE_HOLDOUT_N,
    keep_indexed: int = MIN_ARTICLE_CORPUS,
) -> int:
    """Holdout size that leaves ``keep_indexed`` articles in retrieval."""
    allowed = max(0, int(total_articles) - max(0, int(keep_indexed)))
    target = min(int(maximum), max(int(minimum), round(max(0, briefable) * max(0.0, fraction))))
    return max(0, min(briefable, allowed, target))


def select_article_holdouts(
    pieces: Iterable[Piece],
    *,
    pinned_ids: Sequence[str] = (),
    fraction: float = DEFAULT_ARTICLE_HOLDOUT_FRACTION,
    minimum: int = MIN_ARTICLE_HOLDOUT_N,
    maximum: int = MAX_ARTICLE_HOLDOUT_N,
    keep_indexed: int = MIN_ARTICLE_CORPUS,
) -> dict[str, Any]:
    """Choose article holdouts and return a Contoso-safe receipt.

    Pinned ids are kept whether or not they are briefable: they are already out
    of retrieval, and re-admitting one would silently contaminate a later run
    with an earlier one.
    """
    candidates = [piece for piece in pieces if piece.source in ARTICLE_SOURCES]
    briefable = [
        piece.id
        for piece in candidates
        if is_article_briefable(normalize_corpus_text(piece.text or ""))
    ]
    pinned = [str(piece_id) for piece_id in pinned_ids]
    known = {piece.id for piece in candidates}
    missing_pinned = sorted(set(pinned) - known)

    target_n = resolve_article_holdout_n(
        len(briefable),
        total_articles=len(candidates),
        fraction=fraction,
        minimum=minimum,
        maximum=maximum,
        keep_indexed=keep_indexed,
    )

    chosen: list[str] = [piece_id for piece_id in pinned if piece_id in known]
    for piece_id in sorted(set(briefable) - set(chosen), key=_order_key):
        if len(chosen) >= target_n:
            break
        chosen.append(piece_id)

    return {
        "kind": "article_holdout_carve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_ids": sorted(chosen),
        "n_holdouts": len(chosen),
        "n_articles": len(candidates),
        "n_briefable": len(briefable),
        "target_n": target_n,
        "keep_indexed": int(keep_indexed),
        "articles_left_indexed": max(0, len(candidates) - len(chosen)),
        "pinned_ids": sorted(set(pinned) & known),
        "pinned_ids_missing_from_corpus": missing_pinned,
        "fraction": float(fraction),
        "selection": "blake2b(piece_id) ascending, pinned ids first",
    }


def load_pinned_article_holdout_ids(paths: ProfilePaths) -> list[str]:
    """Ids from the profile's existing article carve (empty when absent)."""
    path = paths.root / ARTICLE_HOLDOUT_FILENAME
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        ids = data.get("holdout_ids") or data.get("ids") or []
    else:
        ids = data
    return [str(piece_id) for piece_id in ids]


def save_article_holdout_ids(paths: ProfilePaths, receipt: dict[str, Any]) -> Any:
    """Persist the article carve. Ids and counts only — never piece text."""
    path = paths.root / ARTICLE_HOLDOUT_FILENAME
    payload = {
        "holdout_ids": receipt["holdout_ids"],
        "n_holdouts": receipt["n_holdouts"],
        "n_articles": receipt["n_articles"],
        "n_briefable": receipt["n_briefable"],
        "articles_left_indexed": receipt["articles_left_indexed"],
        "selection": receipt["selection"],
        "updated_at": receipt["created_at"],
        "note": "article-channel eval carve; ids only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
