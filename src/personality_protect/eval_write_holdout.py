"""Holdout eval for RAG write vs bare-base (Camp A Lane G).

Carve holdout pieces that were never indexed, mine briefs from them, draft with
RAG and with a bare-base (no-exemplar) prompt, then score Contoso-safe receipts.

MLX is never imported here. Callers inject ``generate_fn`` in tests; the CLI
defaults to :func:`personality_protect.write.mlx_generate_no_adapter` which
gates Metal behind ``PP_MLX_ALLOW``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personality_protect.config import ProfilePaths, load_config
from personality_protect.models import Piece, load_index
from personality_protect.pair_gate import text_axes
from personality_protect.prompt_write import build_write_prompt
from personality_protect.voice_index import VECTORS_FILENAME
from personality_protect.write import (
    DEFAULT_WRITE_K,
    DEFAULT_WRITE_MAX_TOKENS,
    GenerateFn,
    build_brief,
    mlx_generate_no_adapter,
    normalize_sentence_case,
    run_write,
)
from personality_protect.writer_guards import check_invention, parrot_reject

TIE_EPSILON = 0.05
BARE_BASE_EXAMPLES: tuple[str, ...] = ()
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOPIC_WORD_CAP = 12

# Contoso tokens that must never appear in public receipts (personal leak check).
_RECEIPT_BANNED = (
    "linkedin.com",
    "@gmail",
    "dusan",
    "telivity.ai/private",
)


def _voice_index_dir(paths: ProfilePaths) -> Path:
    return paths.root / "voice_index"


def load_holdout_pieces(
    paths: ProfilePaths,
    holdout_ids: Iterable[str],
) -> list[Piece]:
    """Load corpus pieces matching holdout ids (order preserved by id sort)."""
    wanted = {str(piece_id) for piece_id in holdout_ids}
    if not wanted:
        raise ValueError("holdout_ids must not be empty")
    pieces = load_index(paths.index_path)
    by_id = {piece.id: piece for piece in pieces}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise FileNotFoundError(
            f"Holdout ids not in corpus index: {', '.join(missing)}"
        )
    return [by_id[piece_id] for piece_id in sorted(wanted)]


def indexed_piece_ids(paths: ProfilePaths) -> set[str]:
    """Return ids currently present in the voice retrieval index."""
    vectors_path = _voice_index_dir(paths) / VECTORS_FILENAME
    if not vectors_path.is_file():
        return set()
    ids: set[str] = set()
    with vectors_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            piece = row.get("piece") or {}
            piece_id = piece.get("id")
            if piece_id is not None:
                ids.add(str(piece_id))
    return ids


def verify_holdouts_never_indexed(
    paths: ProfilePaths,
    holdout_ids: Iterable[str],
) -> dict[str, Any]:
    """g1: confirm holdout ids are absent from voice_index vectors."""
    wanted = {str(piece_id) for piece_id in holdout_ids}
    indexed = indexed_piece_ids(paths)
    leaked = sorted(wanted & indexed)
    return {
        "ok": not leaked,
        "holdout_ids": sorted(wanted),
        "indexed_holdout_ids": leaked,
        # Profile-relative only — never absolute home paths (Contoso-safe).
        "profile": paths.name,
        "voice_index": "voice_index",
        "indexed_total": len(indexed),
    }


def mine_brief_from_holdout(
    text: str,
    *,
    holdout_id: str = "",
) -> dict[str, str]:
    """g2: deterministically mine topic + points from a holdout post.

    Topic is the first sentence (capped). Points are the full holdout text so
    invent-guard compares drafts against facts available in the source brief.
    """
    body = (text or "").strip()
    if not body:
        raise ValueError("holdout text must not be empty")
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(body) if part.strip()]
    first = sentences[0]
    words = first.split()
    if len(words) > _TOPIC_WORD_CAP:
        topic = " ".join(words[:_TOPIC_WORD_CAP]).rstrip(".,;:")
    else:
        topic = first.rstrip(".,;:")
    return {
        "holdout_id": holdout_id,
        "topic": topic,
        "points": body,
    }


def run_bare_base_write(
    topic: str,
    points: str,
    *,
    generate_fn: GenerateFn,
    base_model: str,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
) -> dict[str, Any]:
    """Generate from the locked prompt with no retrieved exemplars."""
    topic = topic.strip()
    points = points.strip()
    if not topic or not points:
        raise ValueError("topic and points must not be empty")
    prompt = build_write_prompt(topic=topic, points=points, examples=BARE_BASE_EXAMPLES)
    draft = str(
        generate_fn(prompt, base_model=base_model, max_tokens=max_tokens)
    ).strip()
    brief = build_brief(topic, points)
    invention = check_invention(brief, normalize_sentence_case(draft))
    return {
        "text": draft,
        "mode": "bare_base",
        "adapter": "none",
        "model": base_model,
        "k": 0,
        "exemplar_ids": [],
        "parrot_reject": parrot_reject(draft, ()),
        "invent_reject": not invention.passed,
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
        "prompt": prompt,
    }


def _axes_distance(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    """Lower = closer to the holdout voice band."""
    return round(
        abs(float(reference["short_line_ratio"]) - float(candidate["short_line_ratio"]))
        + abs(
            float(reference["median_sentence_words"])
            - float(candidate["median_sentence_words"])
        )
        / 20.0
        + abs(float(reference["you_gt_i"]) - float(candidate["you_gt_i"]))
        + abs(float(reference["proper_per_1k"]) - float(candidate["proper_per_1k"]))
        / 50.0
        + abs(int(reference["you_count"]) - int(candidate["you_count"])) / 10.0
        + abs(int(reference["i_count"]) - int(candidate["i_count"])) / 10.0,
        4,
    )


def score_draft_against_holdout(
    holdout_text: str,
    draft: str,
    brief: str,
) -> dict[str, Any]:
    """g4: axis distance + invent flags (counts + keys; no draft body)."""
    ref_axes = text_axes(holdout_text)
    draft_axes = text_axes(draft)
    invention = check_invention(brief, normalize_sentence_case(draft))
    return {
        "distance": _axes_distance(ref_axes, draft_axes),
        "axes": draft_axes,
        "invent_reject": not invention.passed,
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
        "invented_entities_count": len(invention.invented_entities),
        "invented_numbers_count": len(invention.invented_numbers),
    }


def score_rag_vs_base(
    holdout_text: str,
    rag_draft: str,
    base_draft: str,
    brief: str,
    *,
    tie_epsilon: float = TIE_EPSILON,
) -> dict[str, Any]:
    """Three-way score: holdout reference vs RAG draft vs bare-base draft."""
    holdout_axes = text_axes(holdout_text)
    rag = score_draft_against_holdout(holdout_text, rag_draft, brief)
    base = score_draft_against_holdout(holdout_text, base_draft, brief)
    delta = float(base["distance"]) - float(rag["distance"])
    if delta > float(tie_epsilon):
        winner = "rag"
    elif delta < -float(tie_epsilon):
        winner = "base"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "delta_base_minus_rag": round(delta, 4),
        "holdout_axes": holdout_axes,
        "rag": rag,
        "base": base,
        "tie_epsilon": float(tie_epsilon),
    }


def _item_receipt(
    *,
    holdout_id: str,
    brief: dict[str, str],
    rag_result: dict[str, Any],
    base_result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    """Contoso-safe per-holdout row: ids, scores, invent flags — never draft text."""
    return {
        "holdout_id": holdout_id,
        "topic_words": len(brief["topic"].split()),
        "points_words": len(brief["points"].split()),
        "winner": score["winner"],
        "delta_base_minus_rag": score["delta_base_minus_rag"],
        "rag_distance": score["rag"]["distance"],
        "base_distance": score["base"]["distance"],
        "rag_invent_reject": score["rag"]["invent_reject"],
        "base_invent_reject": score["base"]["invent_reject"],
        "rag_invented_entities_count": score["rag"]["invented_entities_count"],
        "base_invented_entities_count": score["base"]["invented_entities_count"],
        "rag_invented_numbers_count": score["rag"]["invented_numbers_count"],
        "base_invented_numbers_count": score["base"]["invented_numbers_count"],
        "exemplar_ids": list(rag_result.get("exemplar_ids") or []),
        "rag_k": int(rag_result.get("k") or 0),
        "base_k": int(base_result.get("k") or 0),
        "rag_adapter": rag_result.get("adapter", "none"),
        "base_adapter": base_result.get("adapter", "none"),
    }


def assert_receipt_contoso_safe(receipt: dict[str, Any]) -> None:
    """Fail closed if a receipt embeds personal or draft body markers."""
    blob = json.dumps(receipt, ensure_ascii=False).lower()
    for token in _RECEIPT_BANNED:
        if token in blob:
            raise ValueError(f"receipt leaks banned token: {token!r}")
    # Draft/holdout bodies must not be serialized under common text keys.
    stack: list[Any] = [receipt]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                if key in {"text", "draft", "holdout_text", "points", "topic", "prompt"}:
                    raise ValueError(f"receipt must omit body field {key!r}")
                stack.append(value)
        elif isinstance(cur, list):
            stack.extend(cur)


def run_eval_write_holdout(
    paths: ProfilePaths,
    holdout_ids: Sequence[str],
    *,
    k: int = DEFAULT_WRITE_K,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
    generate_fn: GenerateFn | None = None,
    generate_fn_base: GenerateFn | None = None,
    tie_epsilon: float = TIE_EPSILON,
) -> dict[str, Any]:
    """Run RAG + bare-base drafts per holdout and return a Contoso-safe receipt."""
    ids = [str(piece_id) for piece_id in holdout_ids]
    carve = verify_holdouts_never_indexed(paths, ids)
    if not carve["ok"]:
        raise ValueError(
            "Holdout ids are present in voice_index (data leak into retrieval): "
            + ", ".join(carve["indexed_holdout_ids"])
        )

    config = load_config(paths)
    model_id = config.base_model
    generator = generate_fn or mlx_generate_no_adapter
    base_generator = generate_fn_base or generator

    pieces = load_holdout_pieces(paths, ids)
    items: list[dict[str, Any]] = []
    wins = {"rag": 0, "base": 0, "tie": 0}

    for piece in pieces:
        brief = mine_brief_from_holdout(piece.text, holdout_id=piece.id)
        rag_result = run_write(
            brief["topic"],
            brief["points"],
            paths,
            k=k,
            max_tokens=max_tokens,
            generate_fn=generator,
        )
        base_result = run_bare_base_write(
            brief["topic"],
            brief["points"],
            generate_fn=base_generator,
            base_model=model_id,
            max_tokens=max_tokens,
        )
        score = score_rag_vs_base(
            piece.text,
            rag_result["text"],
            base_result["text"],
            build_brief(brief["topic"], brief["points"]),
            tie_epsilon=tie_epsilon,
        )
        wins[score["winner"]] = wins.get(score["winner"], 0) + 1
        items.append(
            _item_receipt(
                holdout_id=piece.id,
                brief=brief,
                rag_result=rag_result,
                base_result=base_result,
                score=score,
            )
        )

    receipt: dict[str, Any] = {
        "kind": "eval_write_holdout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "voice_mode": config.voice_mode,
        "adapter": "none",
        "model": model_id,
        "k": k,
        "n_holdouts": len(items),
        "holdout_ids": [item["holdout_id"] for item in items],
        "carve": carve,
        "wins": wins,
        "rag_beats_base": wins["rag"] > wins["base"],
        "items": items,
    }
    assert_receipt_contoso_safe(receipt)
    return receipt


def write_receipt(receipt: dict[str, Any], path: Path) -> Path:
    """Persist a Contoso-safe receipt JSON (gitignored evals/ recommended)."""
    assert_receipt_contoso_safe(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
