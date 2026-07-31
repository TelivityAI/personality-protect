"""Lossy brief mining for article holdouts.

The post path already answers this question for posts: a brief is what the
author jotted down *before* writing, not an extract of the finished piece, and
:mod:`personality_protect.eval_write_holdout` enforces that with a hard word cap
plus a source-overlap cap. Articles need the same guarantee and cannot reuse the
post miner unchanged, for two reasons:

* **the overlap cap stops binding.** A 25% cap on a 1,000-word article permits a
  250-word "brief". The cap has to shrink as the source grows, so the article
  budget is a small fixed word count that a longer source cannot inflate.
* **an article brief is an outline.** Ranking every sentence by fact density and
  taking the top three returns three claims from whichever passage happens to be
  the densest. Bullets are therefore drawn one per segment of the piece, in
  document order, so the brief describes the shape of an article instead of one
  paragraph of it.

Mining runs against the de-voiced clauses from
:mod:`personality_protect.devoice`, so the bullets carry the author's claims
without the author's phrasing, and the result is measured against the original
article on both overlap and 5-gram copy ratio before it is returned.
"""

from __future__ import annotations

from typing import Any

from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.devoice import (
    MAX_PAIR_COPY_RATIO,
    devoice_sentences,
    pair_copy_ratio,
)
from personality_protect.eval_write_holdout import (
    _fact_score,
    _fit_phrase,
    _mine_topic_with_source,
    _word_tokens,
    brief_word_overlap_ratio,
)

# Absolute budget for the model-visible brief. Fixed rather than proportional:
# a share of the source grows with the source, and the whole point is that a
# longer article does not earn a longer head start.
ARTICLE_MAX_BRIEF_WORDS = 60
# Share of the source the brief may return. Binds on short articles, where the
# absolute cap alone would be generous.
ARTICLE_MAX_BRIEF_OVERLAP = 0.10
ARTICLE_TOPIC_WORD_CAP = 10
ARTICLE_POINT_WORD_CAP = 12
ARTICLE_MIN_POINTS = 3
ARTICLE_MAX_POINTS = 6
ARTICLE_MIN_POINT_WORDS = 3
# Share of brief words sitting inside a 5-gram of the article. Same meter and
# same threshold as the writer pair gate: a brief that trips it is an extract
# whichever channel produced it, and a second number here would be a second
# thing to justify.
ARTICLE_MAX_COPY_RATIO = MAX_PAIR_COPY_RATIO
# Below this a piece is a long post, not an article, and the outline segmenting
# has nothing to segment.
MIN_ARTICLE_BRIEF_WORDS = 200


class ArticleBriefRejected(ValueError):
    """An article could not be reduced to a brief that is not an extract."""

    def __init__(self, reasons: list[str], report: dict[str, Any]) -> None:
        super().__init__("article brief rejected: " + ", ".join(reasons))
        self.reasons = reasons
        self.report = report


def _outline_positions(candidate_count: int, points: int) -> list[int]:
    """Segment boundaries splitting ``candidate_count`` clauses into ``points``."""
    if candidate_count <= 0 or points <= 0:
        return []
    step = candidate_count / points
    return [int(index * step) for index in range(points + 1)]


def select_outline_clauses(
    candidates: list[tuple[int, str]],
    *,
    points: int,
) -> list[tuple[int, str]]:
    """Highest-substance clause from each equal segment, in document order.

    Coverage is the property that matters here. Global ranking is what the post
    miner does and it is correct for a post, where every sentence is in the same
    passage; on an article it returns a cluster.
    """
    wanted = max(1, min(int(points), len(candidates)))
    bounds = _outline_positions(len(candidates), wanted)
    chosen: list[tuple[int, str]] = []
    for start, end in zip(bounds, bounds[1:]):
        segment = candidates[start:max(end, start + 1)]
        if not segment:
            continue
        best = max(segment, key=lambda item: (_fact_score(item[1]), -item[0]))
        if best not in chosen:
            chosen.append(best)
    return sorted(chosen, key=lambda item: item[0])


def _content_word_count(topic: str, points: str) -> int:
    """Words the brief hands over, ignoring the bullet markers we added."""
    point_words = [word for word in points.split() if word not in {"-", "*", "•"}]
    return len(topic.split()) + len(point_words)


def mine_article_brief(
    text: str,
    *,
    holdout_id: str = "",
    max_points: int = ARTICLE_MAX_POINTS,
    max_overlap: float = ARTICLE_MAX_BRIEF_OVERLAP,
    max_copy_ratio: float = ARTICLE_MAX_COPY_RATIO,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Mine a topic plus section bullets from an article, and prove it is lossy.

    Returns ``(brief, report)``. ``brief['guard_facts']`` stays the original
    article so the invention guard can reject facts the author never wrote
    without those facts reaching the generation prompt — the same split the post
    path uses. Receipts serialize neither field.
    """
    original = normalize_corpus_text(text)
    source_words = len(_word_tokens(original))
    if source_words < MIN_ARTICLE_BRIEF_WORDS:
        raise ArticleBriefRejected(
            ["article_too_short"], {"source_words": source_words}
        )

    clauses = devoice_sentences(original)
    if len(clauses) < ARTICLE_MIN_POINTS + 1:
        raise ArticleBriefRejected(
            ["devoiced_clauses_too_few"],
            {"source_words": source_words, "clauses": len(clauses)},
        )

    budget = min(ARTICLE_MAX_BRIEF_WORDS, int(source_words * max_overlap))
    if budget < ARTICLE_MIN_POINTS * ARTICLE_MIN_POINT_WORDS:
        raise ArticleBriefRejected(
            ["brief_budget_too_small"],
            {"source_words": source_words, "budget": budget},
        )

    topic, topic_index = _mine_topic_with_source(
        clauses,
        word_cap=min(
            ARTICLE_TOPIC_WORD_CAP,
            budget - ARTICLE_MIN_POINTS * ARTICLE_MIN_POINT_WORDS,
        ),
    )
    candidates = [(i, clause) for i, clause in enumerate(clauses) if i != topic_index]
    target_points = max(ARTICLE_MIN_POINTS, min(ARTICLE_MAX_POINTS, int(max_points)))
    selected = select_outline_clauses(candidates, points=target_points)

    remaining = budget - len(topic.split())
    bullets: list[str] = []
    for _, clause in selected:
        minimum_after = max(
            0, (ARTICLE_MIN_POINTS - len(bullets) - 1) * ARTICLE_MIN_POINT_WORDS
        )
        word_cap = min(ARTICLE_POINT_WORD_CAP, remaining - minimum_after)
        if word_cap < ARTICLE_MIN_POINT_WORDS:
            break
        fitted = _fit_phrase(clause, word_cap)
        fitted_words = len(_word_tokens(fitted))
        if fitted_words < ARTICLE_MIN_POINT_WORDS:
            continue
        remaining -= fitted_words
        bullets.append("- " + fitted)

    if len(bullets) < ARTICLE_MIN_POINTS:
        raise ArticleBriefRejected(
            ["too_few_bullets"],
            {"source_words": source_words, "bullets": len(bullets)},
        )

    points = "\n".join(bullets)
    brief = {
        "holdout_id": holdout_id,
        "topic": topic,
        "points": points,
        "guard_facts": original,
    }
    brief_words = _content_word_count(topic, points)
    overlap = brief_word_overlap_ratio(brief, original)
    copy_ratio = pair_copy_ratio(f"{topic}\n{points}", original)
    report = {
        "source_words": source_words,
        "brief_words": brief_words,
        "bullets": len(bullets),
        "brief_overlap_ratio": overlap,
        "brief_copy_ratio": copy_ratio,
        "max_brief_words": ARTICLE_MAX_BRIEF_WORDS,
        "max_overlap": float(max_overlap),
        "max_copy_ratio": float(max_copy_ratio),
    }

    reasons: list[str] = []
    if brief_words > ARTICLE_MAX_BRIEF_WORDS:
        reasons.append("brief_word_cap")
    if overlap > float(max_overlap):
        reasons.append("brief_overlap")
    if copy_ratio > float(max_copy_ratio):
        reasons.append("brief_copy_ratio")
    if reasons:
        raise ArticleBriefRejected(reasons, report)
    return brief, report


def is_article_briefable(text: str) -> bool:
    """True when a lossy article brief can be mined from this text."""
    try:
        mine_article_brief(text)
    except (ArticleBriefRejected, ValueError):
        return False
    return True
