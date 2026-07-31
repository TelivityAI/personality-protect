"""Contoso-safe article brief mining: lossy outline, never an extract."""

from __future__ import annotations

import pytest
from contoso_articles import contoso_article_text

from personality_protect.article_brief import (
    ARTICLE_MAX_BRIEF_WORDS,
    ARTICLE_MAX_COPY_RATIO,
    ARTICLE_MIN_POINTS,
    ArticleBriefRejected,
    is_article_briefable,
    mine_article_brief,
    select_outline_clauses,
)
from personality_protect.eval_write_holdout import brief_word_overlap_ratio


def _brief(seed: int = 0):
    return mine_article_brief(contoso_article_text(seed), holdout_id=f"contoso-{seed}")


def test_mined_brief_is_topic_plus_section_bullets():
    brief, report = _brief()
    bullets = [line for line in brief["points"].splitlines() if line.strip()]
    assert brief["topic"].strip()
    assert len(bullets) >= ARTICLE_MIN_POINTS
    assert all(line.startswith("- ") for line in bullets)
    assert report["bullets"] == len(bullets)


def test_brief_stays_inside_the_hard_word_cap():
    _, report = _brief()
    assert report["brief_words"] <= ARTICLE_MAX_BRIEF_WORDS


def test_brief_does_not_hand_back_the_article():
    brief, report = _brief()
    body = contoso_article_text(0)
    assert brief["points"].strip() != body.strip()
    assert report["brief_overlap_ratio"] <= report["max_overlap"]
    assert brief_word_overlap_ratio(brief, body) <= report["max_overlap"]
    assert report["brief_copy_ratio"] <= ARTICLE_MAX_COPY_RATIO


def test_overlap_cap_binds_harder_as_the_article_grows():
    """A longer source must not buy a proportionally longer brief."""
    _, short_report = mine_article_brief(contoso_article_text(0, sections=3))
    _, long_report = mine_article_brief(contoso_article_text(0, sections=6))
    assert long_report["source_words"] > short_report["source_words"]
    assert long_report["brief_overlap_ratio"] < short_report["brief_overlap_ratio"]
    assert long_report["brief_words"] <= ARTICLE_MAX_BRIEF_WORDS


def test_guard_facts_keep_the_whole_article_out_of_the_prompt():
    brief, _ = _brief()
    body = contoso_article_text(0)
    assert brief["guard_facts"].strip().startswith(body.split("\n", 1)[0])
    assert body.split("\n", 1)[0] not in brief["points"]


def test_short_source_is_rejected_rather_than_briefed():
    with pytest.raises(ArticleBriefRejected, match="article_too_short"):
        mine_article_brief("Contoso keeps the queue boring. You name one owner.")


def test_is_article_briefable_matches_mining():
    assert is_article_briefable(contoso_article_text(1))
    assert not is_article_briefable("Contoso ships. You own it.")


def test_outline_clauses_are_spread_across_the_piece():
    """Bullets come one per segment, not three from the densest paragraph."""
    candidates = [(index, f"clause {index}") for index in range(12)]
    # Load the substance into one segment so a global ranking would cluster.
    candidates[1] = (1, "Contoso Ledger cut twelve percent across four regions")
    candidates[2] = (2, "Contoso Ledger cut fourteen percent across five regions")
    candidates[3] = (3, "Contoso Ledger cut sixteen percent across six regions")
    chosen = select_outline_clauses(candidates, points=4)
    positions = [index for index, _ in chosen]
    assert positions == sorted(positions)
    assert max(positions) >= 8
    assert len(chosen) == 4


def test_briefs_are_deterministic():
    first, _ = _brief(2)
    second, _ = _brief(2)
    assert first == second
