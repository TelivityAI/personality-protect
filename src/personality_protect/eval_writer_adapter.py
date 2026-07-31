"""Writer-LoRA ship gate: RAG+adapter vs RAG-alone on carved holdouts.

The previous gate was an ad-hoc script, so its bar lived only in whoever ran it.
It is committed here because a ship decision that cannot be re-run is not a gate.

Both arms retrieve the same exemplars and see the same de-voiced brief; the only
difference is whether the writer adapter is loaded. Scoring reuses the shipped
holdout scorer, including its disqualifications — a draft that invents entities
or parrots its context cannot win on rhythm, which is exactly how the first
adapter would otherwise have scored well while writing nothing.

MLX is never imported here. The CLI injects generators that load each arm's
weights once and reuse them across holdouts; tests inject plain callables.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from math import comb
from typing import Any

from personality_protect.config import ProfilePaths, load_config
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.devoice import DevoiceRejected, mine_writer_brief
from personality_protect.eval_write_holdout import (
    TIE_EPSILON,
    assert_receipt_contoso_safe,
    load_holdout_pieces,
    score_rag_vs_base,
    verify_holdouts_never_indexed,
)
from personality_protect.write import (
    DEFAULT_WRITE_K,
    DEFAULT_WRITE_MAX_TOKENS,
    GenerateFn,
    run_write,
)

# One-sided significance required to keep an adapter. Deliberately lenient for a
# local voice model — this is a ship decision, not a paper — but it is a real
# threshold: at n=3 a clean sweep still only reaches p=0.125, which is why the
# previous run could not have passed a bar of any kind.
SHIP_ALPHA = 0.10


def sign_test_p_value(wins: int, losses: int) -> float:
    """One-sided probability of ``wins`` or better from a fair coin.

    Ties are excluded rather than split: a tie says the two arms were
    indistinguishable on that holdout, which is evidence for neither.
    """
    decisive = int(wins) + int(losses)
    if decisive <= 0:
        return 1.0
    tail = sum(comb(decisive, k) for k in range(int(wins), decisive + 1))
    return round(tail / (2**decisive), 4)


def decide_ship(
    wins: dict[str, int],
    *,
    adapter_disqualified: int,
    rag_disqualified: int,
    alpha: float = SHIP_ALPHA,
) -> dict[str, Any]:
    """Keep-or-archive decision plus every reason it was reached.

    Three conditions, all required:

    * the adapter wins more holdouts than it loses
    * that margin is unlikely enough under a fair coin to be worth acting on
    * the adapter is not disqualified more often than the arm it replaces —
      invention and parroting were the real signal in the failed run, and an
      adapter that wins on rhythm while fabricating more is not shippable
    """
    adapter_wins = int(wins.get("adapter", 0))
    rag_wins = int(wins.get("rag", 0))
    p_value = sign_test_p_value(adapter_wins, rag_wins)
    reasons: list[str] = []
    if adapter_wins <= rag_wins:
        reasons.append("adapter_did_not_win_majority")
    if p_value > float(alpha):
        reasons.append("margin_within_chance")
    if adapter_disqualified > rag_disqualified:
        reasons.append("adapter_disqualified_more_often")
    return {
        "decision": "keep" if not reasons else "archive",
        "adapter_beats_rag": adapter_wins > rag_wins,
        "p_value": p_value,
        "alpha": float(alpha),
        "blocking_reasons": reasons,
    }


def _item_receipt(
    *,
    holdout_id: str,
    holdout_text: str,
    brief_report: dict[str, Any],
    adapter_result: dict[str, Any],
    rag_result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    """Contoso-safe per-holdout row: ids, ratios and flags — never body text.

    ``score_rag_vs_base`` labels its arms ``rag``/``base``; the adapter arm is
    passed in its ``rag`` slot, so the labels are remapped once, here, rather
    than left for a reader of the receipt to untangle.
    """
    adapter, rag = score["rag"], score["base"]
    return {
        "holdout_id": holdout_id,
        "holdout_words": len((holdout_text or "").split()),
        "brief_copy_ratio": brief_report.get("brief_copy_ratio"),
        "note_copy_ratio": brief_report.get("copy_ratio"),
        "winner": {"rag": "adapter", "base": "rag"}.get(score["winner"], "tie"),
        "delta_rag_minus_adapter": score["delta_base_minus_rag"],
        "adapter_distance": adapter["distance"],
        "rag_distance": rag["distance"],
        "adapter_disqualified": adapter["disqualified"],
        "rag_disqualified": rag["disqualified"],
        "adapter_parrot_reject": adapter["parrot_reject"],
        "rag_parrot_reject": rag["parrot_reject"],
        "adapter_invent_reject": adapter["invent_reject"],
        "rag_invent_reject": rag["invent_reject"],
        "adapter_brief_echo_reject": adapter["brief_echo_reject"],
        "rag_brief_echo_reject": rag["brief_echo_reject"],
        "adapter_invented_entities_count": adapter["invented_entities_count"],
        "rag_invented_entities_count": rag["invented_entities_count"],
        "adapter_draft_words": len(str(adapter_result.get("text") or "").split()),
        "rag_draft_words": len(str(rag_result.get("text") or "").split()),
        "exemplar_ids": list(rag_result.get("exemplar_ids") or []),
    }


def run_writer_adapter_gate(
    paths: ProfilePaths,
    holdout_ids: Sequence[str],
    *,
    generate_fn_adapter: GenerateFn,
    generate_fn_rag: GenerateFn,
    k: int = DEFAULT_WRITE_K,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
    tie_epsilon: float = TIE_EPSILON,
    alpha: float = SHIP_ALPHA,
    on_item: Any = None,
) -> dict[str, Any]:
    """Score both arms on every holdout and return a Contoso-safe receipt.

    ``on_item`` is called with each finished row so a long unattended run can
    report progress without the caller waiting for the whole gate.
    """
    ids = [str(piece_id) for piece_id in holdout_ids]
    carve = verify_holdouts_never_indexed(paths, ids)
    if not carve["ok"]:
        raise ValueError(
            "Holdout ids are present in voice_index (retrieval leak): "
            + ", ".join(carve["indexed_holdout_ids"])
        )

    config = load_config(paths)
    pieces = load_holdout_pieces(paths, ids)
    items: list[dict[str, Any]] = []
    wins = {"adapter": 0, "rag": 0, "tie": 0}
    skipped: list[str] = []
    adapter_dq = 0
    rag_dq = 0

    for piece in pieces:
        holdout_text = normalize_corpus_text(piece.text)
        try:
            brief, brief_report = mine_writer_brief(holdout_text, holdout_id=piece.id)
        except (DevoiceRejected, ValueError):
            # A holdout that cannot be briefed is not a loss for either arm.
            skipped.append(piece.id)
            continue

        adapter_result = run_write(
            brief["topic"],
            brief["points"],
            paths,
            k=k,
            max_tokens=max_tokens,
            use_adapter=True,
            generate_fn=generate_fn_adapter,
        )
        rag_result = run_write(
            brief["topic"],
            brief["points"],
            paths,
            k=k,
            max_tokens=max_tokens,
            use_adapter=False,
            generate_fn=generate_fn_rag,
        )
        score = score_rag_vs_base(
            holdout_text,
            adapter_result["text"],
            rag_result["text"],
            brief["guard_facts"],
            rag_exemplars=list(adapter_result.get("exemplar_texts") or []),
            tie_epsilon=tie_epsilon,
        )
        item = _item_receipt(
            holdout_id=piece.id,
            holdout_text=holdout_text,
            brief_report=brief_report,
            adapter_result=adapter_result,
            rag_result=rag_result,
            score=score,
        )
        wins[item["winner"]] = wins.get(item["winner"], 0) + 1
        adapter_dq += int(bool(item["adapter_disqualified"]))
        rag_dq += int(bool(item["rag_disqualified"]))
        items.append(item)
        if on_item is not None:
            on_item(item)

    verdict = decide_ship(
        wins,
        adapter_disqualified=adapter_dq,
        rag_disqualified=rag_dq,
        alpha=alpha,
    )
    receipt: dict[str, Any] = {
        "kind": "eval_writer_adapter_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.base_model,
        "voice_mode": config.voice_mode,
        "k": k,
        "pair_kind": "devoiced_brief_to_post",
        "n_holdouts": len(items),
        "n_requested": len(ids),
        "skipped_unbriefable": sorted(skipped),
        "holdout_ids": [item["holdout_id"] for item in items],
        "carve": carve,
        "wins": wins,
        "disqualified": {"adapter": adapter_dq, "rag": rag_dq},
        **verdict,
        "items": items,
    }
    assert_receipt_contoso_safe(receipt)
    return receipt


def write_gate_receipt(receipt: dict[str, Any], path: Any) -> Any:
    """Persist a Contoso-safe gate receipt."""
    assert_receipt_contoso_safe(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
