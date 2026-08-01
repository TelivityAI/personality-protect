"""Holdout eval for the article channel: outline→sections→stitch vs bare base.

The post channel has had an honest holdout gate since Lane G; the article
channel shipped without one, which meant its voice claim rested on nothing but
the fact that the code ran. This is the missing half.

The comparison is the same one the post path makes, with two article-specific
adjustments that exist so the result measures writing rather than editing:

* **both arms are trimmed to the same ceiling.** The article arm's length is the
  sum of its section budgets; the single-shot arm is trimmed to that same total.
  A length penalty is part of the distance score, so arms edited to different
  lengths would be separated before either of them wrote a word.
* **the brief is mined as an outline**, by
  :func:`~personality_protect.article_brief.mine_article_brief`, and capped at a
  fixed word count rather than a share of the source.

Disqualifications carry over unchanged from the post scorer: a draft that
parrots its exemplars, hands the brief back, or invents entities and figures
cannot win on rhythm, however close its cadence lands.

MLX is never imported here. The CLI injects a generator; tests inject callables.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from personality_protect.article_brief import (
    ARTICLE_MAX_BRIEF_OVERLAP,
    ARTICLE_MAX_BRIEF_WORDS,
    ARTICLE_MAX_COPY_RATIO,
    ArticleBriefRejected,
    mine_article_brief,
)
from personality_protect.chat_prompt import flatten_chat_messages
from personality_protect.config import ProfilePaths, load_config
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.draft_trim import drop_repeated_paragraphs, word_count
from personality_protect.eval_write_holdout import (
    TIE_EPSILON,
    assert_receipt_contoso_safe,
    load_holdout_pieces,
    score_rag_vs_base,
    verify_holdouts_never_indexed,
    write_raw_artifacts,
)
from personality_protect.eval_writer_adapter import sign_test_p_value
from personality_protect.style_profile import (
    article_section_words,
    load_style_profile,
)
from personality_protect.write import (
    DEFAULT_WRITE_K,
    GenerateFn,
    build_brief,
    mlx_generate_no_adapter,
)
from personality_protect.write_article import (
    DEFAULT_ARTICLE_SECTION_MAX_TOKENS,
    SECTION_TOKENS_PER_WORD,
    SECTION_TRIM_HEADROOM,
    _section_brief,
    build_section_messages,
    count_indexed_article_pieces,
    draft_section_with_repair,
    outline_from_brief,
    run_write_article,
    scale_section_words_for_brief,
    section_structure_directives,
)
from personality_protect.writer_guards import brief_allowed_facts

# One-sided bar, matching the writer ship gate. Stated rather than enforced:
# with fourteen articles a carve cannot exceed five, and a clean sweep of three
# only reaches p=0.125. The receipt reports the p-value so a run that cannot
# clear the bar is visibly a run that cannot clear the bar.
ARTICLE_ALPHA = 0.10


def article_word_budget(paths: ProfilePaths, topic: str, points: str) -> dict[str, Any]:
    """Outline and word budget both arms are held to for this brief.

    Derived from the same style profile ``run_write_article`` reads, so the
    control arm is asked for exactly the article the product arm is building.
    """
    style = load_style_profile(paths)
    sections = outline_from_brief(topic, points)
    brief = build_brief(topic, points)
    brief_words = word_count(brief)
    section_words = scale_section_words_for_brief(
        article_section_words(style, sections=len(sections)),
        brief_words=brief_words,
        sections=len(sections),
    )
    word_aim = section_words * len(sections)
    allowed = brief_allowed_facts(brief)
    return {
        "sections": sections,
        "section_count": len(sections),
        "word_aim": word_aim,
        "section_words": section_words,
        "section_trim_words": int(round(section_words * SECTION_TRIM_HEADROOM)),
        "word_ceiling": int(round(section_words * SECTION_TRIM_HEADROOM)) * len(sections),
        "max_tokens": max(
            DEFAULT_ARTICLE_SECTION_MAX_TOKENS,
            int(round(section_words * SECTION_TOKENS_PER_WORD)),
        ),
        "allowed_entities": allowed["entities"],
        "allowed_numbers": allowed["numbers"],
        "visible_brief": brief,
    }


def run_bare_base_article(
    topic: str,
    points: str,
    *,
    budget: dict[str, Any],
    generate_fn: GenerateFn,
    base_model: str,
    prompt_sink: list[str] | None = None,
) -> dict[str, Any]:
    """Write the same article with no exemplars and no cadence card.

    The post channel's control arm is a single-shot prompt, and that works
    there because a post is what the model writes by default. Asked the same
    way for an article it returned 58 to 150 words against holdouts of 600 to
    2,400, so the comparison was an article against a stub: the length penalty
    decided three of four holdouts, and the stub could not be disqualified for
    inventing entities because it had barely written any.

    The control therefore gets the same outline, the same per-section budget,
    and the same trim — and, since the product arm gained one, the same invent
    repair and mechanical scrub. A fact-lock that only the product arm has to
    survive would separate the arms by editing policy rather than by writing.
    What the control does not get is the voice machinery — retrieved exemplars
    and the measured style profile — which is the only thing the comparison is
    meant to be about.
    """
    invent_brief = str(budget.get("visible_brief") or build_brief(topic, points))
    section_drafts: list[str] = []
    dropped_sections: list[str] = []
    repaired_sections: list[str] = []
    scrubbed_sections: list[str] = []
    messages: list[dict[str, str]] = []
    attempts_total = 0
    for index, section in enumerate(budget["sections"], start=1):
        section_topic, section_points = _section_brief(topic, section, points)
        outcome = draft_section_with_repair(
            build_messages=partial(
                build_section_messages,
                topic=section_topic,
                points=section_points,
                examples=(),
                directives=section_structure_directives(
                    section=section,
                    index=index,
                    total=budget["section_count"],
                    word_aim=budget["word_aim"],
                    section_words=budget["section_words"],
                    section_trim_words=budget["section_trim_words"],
                    allowed_entities=budget.get("allowed_entities") or (),
                    allowed_numbers=budget.get("allowed_numbers") or (),
                ),
            ),
            generate_fn=generate_fn,
            base_model=base_model,
            invent_brief=invent_brief,
            section_words=budget["section_words"],
            section_trim_words=budget["section_trim_words"],
            max_tokens=budget["max_tokens"],
            prompt_sink=prompt_sink,
        )
        messages = outcome["messages"]
        attempts_total += int(outcome["attempts"])
        if outcome["status"] == "dropped":
            dropped_sections.append(section)
            continue
        section_drafts.append(str(outcome["draft"]))
        if outcome["status"] == "repaired":
            repaired_sections.append(section)
        elif outcome["status"] == "scrubbed":
            scrubbed_sections.append(section)

    text = drop_repeated_paragraphs(
        "\n\n".join(part for part in section_drafts if part.strip())
    ).strip()
    return {
        "text": text,
        "mode": "bare_base_article",
        "adapter": "none",
        "model": base_model,
        "k": 0,
        "exemplar_ids": [],
        "section_count": len(section_drafts),
        "attempts": attempts_total,
        "dropped_sections": dropped_sections,
        "repaired_sections": repaired_sections,
        "scrubbed_sections": scrubbed_sections,
        "prompt": flatten_chat_messages(messages) if budget["sections"] else "",
    }


def _item_receipt(
    *,
    holdout_id: str,
    holdout_text: str,
    brief_report: dict[str, Any],
    budget: dict[str, int],
    article_result: dict[str, Any],
    base_result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    """Contoso-safe per-holdout row: ids, ratios, flags — never body text.

    ``score_rag_vs_base`` names its arms ``rag``/``base``; the article arm is
    passed in the ``rag`` slot, so the labels are remapped once, here.
    """
    article, base = score["rag"], score["base"]
    return {
        "holdout_id": holdout_id,
        "holdout_words": len((holdout_text or "").split()),
        "brief_words": brief_report["brief_words"],
        "brief_bullets": brief_report["bullets"],
        "brief_overlap_ratio": brief_report["brief_overlap_ratio"],
        "brief_copy_ratio": brief_report["brief_copy_ratio"],
        "section_count": budget["section_count"],
        "word_ceiling": budget["word_ceiling"],
        "winner": {"rag": "article", "base": "base"}.get(score["winner"], "tie"),
        "delta_base_minus_article": score["delta_base_minus_rag"],
        "article_distance": article["distance"],
        "base_distance": base["distance"],
        "article_disqualified": article["disqualified"],
        "base_disqualified": base["disqualified"],
        "article_parrot_reject": article["parrot_reject"],
        "base_parrot_reject": base["parrot_reject"],
        "article_brief_echo_reject": article["brief_echo_reject"],
        "base_brief_echo_reject": base["brief_echo_reject"],
        "article_invent_reject": article["invent_reject"],
        "base_invent_reject": base["invent_reject"],
        "article_invented_entities_count": article["invented_entities_count"],
        "base_invented_entities_count": base["invented_entities_count"],
        "article_invented_numbers_count": article["invented_numbers_count"],
        "base_invented_numbers_count": base["invented_numbers_count"],
        "article_draft_words": len(str(article_result.get("text") or "").split()),
        "base_draft_words": len(str(base_result.get("text") or "").split()),
        "article_attempts": int(article_result.get("attempts") or 1),
        "base_attempts": int(base_result.get("attempts") or 1),
        # Counts, never titles: the outline is mined from the holdout body.
        "article_repaired_sections": len(article_result.get("repaired_sections") or []),
        "article_scrubbed_sections": len(article_result.get("scrubbed_sections") or []),
        "article_dropped_sections": len(article_result.get("dropped_sections") or []),
        "base_repaired_sections": len(base_result.get("repaired_sections") or []),
        "base_scrubbed_sections": len(base_result.get("scrubbed_sections") or []),
        "base_dropped_sections": len(base_result.get("dropped_sections") or []),
        "exemplar_ids": list(article_result.get("exemplar_ids") or []),
        "article_k": int(article_result.get("k") or 0),
    }


def decide_article_voice(
    wins: dict[str, int],
    *,
    article_disqualified: int,
    base_disqualified: int,
    alpha: float = ARTICLE_ALPHA,
    both_disqualified: int = 0,
    n_items: int = 0,
) -> dict[str, Any]:
    """Whether this run supports the claim that the article channel has voice.

    Same three conditions as the writer ship gate: win the majority, do it by a
    margin a fair coin would not produce this often, and do not fabricate more
    than the arm being compared against.

    ``both_disqualified`` exists because those three conditions cannot tell two
    different failures apart. An item where both arms are disqualified scores as
    a tie, so a run where *every* item is disqualified reports the same
    "did not win the majority" as a run where the voice arm was measured and
    drifted further from the author. The first never got as far as comparing
    cadence. Saying which one happened is the difference between a result and a
    number that looks like one.
    """
    article_wins = int(wins.get("article", 0))
    base_wins = int(wins.get("base", 0))
    p_value = sign_test_p_value(article_wins, base_wins)
    reasons: list[str] = []
    undecided = n_items > 0 and both_disqualified >= n_items
    if undecided:
        reasons.append("every_item_disqualified_in_both_arms")
    if article_wins <= base_wins:
        reasons.append("article_did_not_win_majority")
    if p_value > float(alpha):
        reasons.append("margin_within_chance")
    if article_disqualified > base_disqualified:
        reasons.append("article_disqualified_more_often")
    return {
        "verdict": "voice_supported" if not reasons else "not_supported",
        "article_beats_base": article_wins > base_wins,
        "distance_ever_decided": not undecided,
        "items_disqualified_in_both_arms": int(both_disqualified),
        "p_value": p_value,
        "alpha": float(alpha),
        "blocking_reasons": reasons,
    }


def run_eval_write_article(
    paths: ProfilePaths,
    holdout_ids: Sequence[str],
    *,
    k: int = DEFAULT_WRITE_K,
    generate_fn: GenerateFn | None = None,
    generate_fn_base: GenerateFn | None = None,
    tie_epsilon: float = TIE_EPSILON,
    alpha: float = ARTICLE_ALPHA,
    save_raw: bool = False,
    on_item: Any = None,
) -> dict[str, Any]:
    """Draft each holdout article both ways and return a Contoso-safe receipt.

    ``save_raw`` dumps the exact prompts and drafts under the profile's
    gitignored ``dogfood/raw`` directory. Those files are verbatim personal text
    for human review; the receipt never references them.
    """
    ids = [str(piece_id) for piece_id in holdout_ids]
    carve = verify_holdouts_never_indexed(paths, ids)
    if not carve["ok"]:
        raise ValueError(
            "Article holdout ids are present in voice_index (retrieval leak): "
            + ", ".join(carve["indexed_holdout_ids"])
        )

    config = load_config(paths)
    generator = generate_fn or mlx_generate_no_adapter
    base_generator = generate_fn_base or generator
    pieces = load_holdout_pieces(paths, ids)

    items: list[dict[str, Any]] = []
    wins = {"article": 0, "base": 0, "tie": 0}
    skipped: list[str] = []
    article_dq = 0
    base_dq = 0
    both_dq = 0

    for piece in pieces:
        holdout_text = normalize_corpus_text(piece.text)
        try:
            brief, brief_report = mine_article_brief(holdout_text, holdout_id=piece.id)
        except (ArticleBriefRejected, ValueError):
            # A holdout neither arm can be asked to write is not a loss for
            # either of them.
            skipped.append(piece.id)
            continue

        budget = article_word_budget(paths, brief["topic"], brief["points"])
        article_prompts: list[str] = []
        base_prompts: list[str] = []
        article_result = run_write_article(
            brief["topic"],
            brief["points"],
            paths,
            k=k,
            generate_fn=generator,
            prompt_sink=article_prompts,
        )
        base_result = run_bare_base_article(
            brief["topic"],
            brief["points"],
            budget=budget,
            generate_fn=base_generator,
            base_model=config.base_model,
            prompt_sink=base_prompts,
        )
        visible = f"{brief['topic']}\n{brief['points']}"
        score = score_rag_vs_base(
            holdout_text,
            article_result["text"],
            base_result["text"],
            # Invent against the same visible brief the product arm fact-locks
            # to — not the full source article the model never saw.
            visible,
            rag_exemplars=list(article_result.get("exemplar_texts") or []),
            tie_epsilon=tie_epsilon,
            visible_brief=visible,
        )
        # The product arm sets this when the stitch is empty or the stitched
        # text still invents — the two cases the scorer cannot see for itself,
        # since an empty draft invents nothing.
        if article_result.get("invent_reject"):
            score["rag"]["invent_reject"] = True
            score["rag"]["disqualified"] = True
            score["rag"]["invented_entities"] = sorted(
                set(score["rag"].get("invented_entities") or [])
                | set(article_result.get("invented_entities") or [])
            )
            score["rag"]["invented_numbers"] = sorted(
                set(score["rag"].get("invented_numbers") or [])
                | set(article_result.get("invented_numbers") or [])
            )
            score["rag"]["invented_entities_count"] = len(
                score["rag"]["invented_entities"]
            )
            score["rag"]["invented_numbers_count"] = len(
                score["rag"]["invented_numbers"]
            )
            if score["winner"] == "rag":
                score["winner"] = "base" if not score["base"]["disqualified"] else "tie"
        if base_result.get("section_count", 1) == 0:
            score["base"]["invent_reject"] = True
            score["base"]["disqualified"] = True
            if score["winner"] == "base":
                score["winner"] = (
                    "rag" if not score["rag"]["disqualified"] else "tie"
                )
        item = _item_receipt(
            holdout_id=piece.id,
            holdout_text=holdout_text,
            brief_report=brief_report,
            budget=budget,
            article_result=article_result,
            base_result=base_result,
            score=score,
        )
        wins[item["winner"]] = wins.get(item["winner"], 0) + 1
        article_dq += int(bool(item["article_disqualified"]))
        base_dq += int(bool(item["base_disqualified"]))
        both_dq += int(
            bool(item["article_disqualified"]) and bool(item["base_disqualified"])
        )
        items.append(item)
        if on_item is not None:
            on_item(item)
        if save_raw:
            for arm, result, sink in (
                ("article", article_result, article_prompts),
                ("bare_base", base_result, base_prompts),
            ):
                write_raw_artifacts(
                    paths,
                    holdout_id=piece.id,
                    arm=arm,
                    prompt=sink[-1] if sink else str(result.get("prompt") or ""),
                    draft=str(result.get("text") or ""),
                    brief=brief,
                )

    verdict = decide_article_voice(
        wins,
        article_disqualified=article_dq,
        base_disqualified=base_dq,
        alpha=alpha,
        both_disqualified=both_dq,
        n_items=len(items),
    )
    receipt: dict[str, Any] = {
        "kind": "eval_write_article",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "channel": "article",
        "model": config.base_model,
        "voice_mode": config.voice_mode,
        "adapter": "none",
        "k": k,
        "n_holdouts": len(items),
        "n_requested": len(ids),
        "skipped_unbriefable": sorted(skipped),
        "holdout_ids": [item["holdout_id"] for item in items],
        "articles_indexed": count_indexed_article_pieces(paths),
        "carve": carve,
        "brief_mining": {
            "hard_brief_word_cap": ARTICLE_MAX_BRIEF_WORDS,
            "max_brief_overlap": ARTICLE_MAX_BRIEF_OVERLAP,
            "max_brief_copy_ratio": ARTICLE_MAX_COPY_RATIO,
        },
        "raw_artifacts_saved": bool(save_raw),
        "wins": wins,
        "disqualified": {"article": article_dq, "base": base_dq},
        **verdict,
        "items": items,
    }
    assert_receipt_contoso_safe(receipt)
    return receipt


def write_article_eval_receipt(receipt: dict[str, Any], path: Path) -> Path:
    """Persist a Contoso-safe article eval receipt."""
    assert_receipt_contoso_safe(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
