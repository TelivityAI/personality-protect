"""Deterministic holdout carve for the writer LoRA ship gate.

The first gate ran on three holdouts and came back 2–1 against the adapter. At
that size the result carries almost no information: three paired comparisons
cannot separate a real regression from a coin flip, so "did not clear the bar"
was the only honest reading, and "training did not help" was not available.

Widening is therefore a precondition for the next gate, not a nice-to-have. The
carve is:

* **deterministic** — a stable digest of the piece id orders candidates, so the
  same corpus always yields the same holdout set and a gate can be re-run
* **pinned-compatible** — ids already carved out stay carved, so results remain
  comparable across runs and no piece silently re-enters retrieval
* **briefable-only** — a piece that cannot produce a de-voiced brief cannot be
  scored by either arm, so it would occupy a holdout slot and contribute nothing
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from personality_protect.config import ProfilePaths
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.devoice import DevoiceRejected, mine_writer_brief
from personality_protect.models import Piece

HOLDOUT_FILENAME = "dogfood_holdout_ids.json"
POST_SOURCES = frozenset({"linkedin_post"})
MIN_HOLDOUT_TARGET_WORDS = 50

# Share of the briefable pool to reserve. A quarter is the smallest carve that
# gets a paired sign test into useful territory on a corpus this size while
# leaving enough rows to train on: at n=20 a 15-5 split is p≈0.02, where n=3
# cannot go below p=0.125 even when the adapter sweeps.
DEFAULT_HOLDOUT_FRACTION = 0.25
MIN_HOLDOUT_N = 12
MAX_HOLDOUT_N = 24


def _order_key(piece_id: str) -> str:
    """Stable, corpus-order-independent shuffle key."""
    return hashlib.blake2b(str(piece_id).encode("utf-8"), digest_size=8).hexdigest()


def is_briefable(piece: Piece) -> bool:
    """True when a de-voiced brief can be mined from this piece.

    Uses the same entry point as pair construction: a holdout the writer path
    cannot brief is one neither arm can be asked to write, and scoring it would
    mean scoring an empty prompt.
    """
    if piece.source not in POST_SOURCES:
        return False
    body = normalize_corpus_text(piece.text or "")
    if len(body.split()) < MIN_HOLDOUT_TARGET_WORDS:
        return False
    try:
        mine_writer_brief(body, holdout_id=piece.id)
    except (DevoiceRejected, ValueError):
        return False
    return True


def resolve_holdout_n(
    pool_size: int,
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    minimum: int = MIN_HOLDOUT_N,
    maximum: int = MAX_HOLDOUT_N,
) -> int:
    """Holdout size for a briefable pool, clamped to a usable band.

    Never returns more than the pool: a carve that consumes every briefable
    piece would leave nothing to train on and the gate would compare two
    untrained arms.
    """
    target = round(max(0, pool_size) * max(0.0, fraction))
    return max(0, min(pool_size, max(minimum, min(maximum, int(target)))))


def select_writer_holdouts(
    pieces: Iterable[Piece],
    *,
    pinned_ids: Sequence[str] = (),
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    minimum: int = MIN_HOLDOUT_N,
    maximum: int = MAX_HOLDOUT_N,
) -> dict[str, Any]:
    """Choose a widened holdout set and return a Contoso-safe receipt.

    Pinned ids are kept whether or not they are briefable — they are already out
    of the retrieval index, and quietly re-admitting a previously carved piece
    would contaminate the comparison with the earlier run.
    """
    candidates = [piece for piece in pieces if piece.source in POST_SOURCES]
    briefable = [piece.id for piece in candidates if is_briefable(piece)]
    pinned = [str(piece_id) for piece_id in pinned_ids]
    known = {piece.id for piece in candidates}
    missing_pinned = sorted(set(pinned) - known)

    pool = sorted(set(briefable) | (set(pinned) & known))
    target_n = resolve_holdout_n(
        len(pool), fraction=fraction, minimum=minimum, maximum=maximum
    )

    chosen: list[str] = [piece_id for piece_id in pinned if piece_id in known]
    for piece_id in sorted(set(briefable) - set(chosen), key=_order_key):
        if len(chosen) >= target_n:
            break
        chosen.append(piece_id)

    return {
        "kind": "writer_holdout_carve",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_ids": sorted(chosen),
        "n_holdouts": len(chosen),
        "n_posts": len(candidates),
        "n_briefable": len(briefable),
        "pool_size": len(pool),
        "target_n": target_n,
        "pinned_ids": sorted(set(pinned) & known),
        "pinned_ids_missing_from_corpus": missing_pinned,
        "train_pairs_remaining": max(0, len(briefable) - len(set(chosen) & set(briefable))),
        "fraction": float(fraction),
        "selection": "blake2b(piece_id) ascending, pinned ids first",
    }


def load_pinned_holdout_ids(paths: ProfilePaths) -> list[str]:
    """Ids from the profile's existing carve file (empty when absent)."""
    path = paths.root / HOLDOUT_FILENAME
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("holdout_ids") or data.get("ids") or [] if isinstance(data, dict) else data
    return [str(piece_id) for piece_id in ids]


def save_holdout_ids(paths: ProfilePaths, receipt: dict[str, Any]) -> Any:
    """Persist the carve. Ids and counts only — never piece text."""
    path = paths.root / HOLDOUT_FILENAME
    payload = {
        "holdout_ids": receipt["holdout_ids"],
        "n_holdouts": receipt["n_holdouts"],
        "n_briefable": receipt["n_briefable"],
        "selection": receipt["selection"],
        "updated_at": receipt["created_at"],
        "note": "writer LoRA ship-gate carve; ids only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
