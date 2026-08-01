"""Deterministic corpus style profile for RAG write prompts.

Computes aggregate voice stats from selected pieces and ships a fixed banned
AI-filler list into ``style_profile.json`` under the local profile root.
Contoso-safe heuristics only — no personal text in package defaults.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from personality_protect.config import ProfilePaths
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.eval_compare import _sentence_word_counts, _word_tokens
from personality_protect.models import Piece
from personality_protect.select import selected_pieces

# Locked prompt + existing slop detectors (filter/eval). Keep lowercase.
BANNED_AI_FILLER: tuple[str, ...] = (
    "leverage",
    "delve",
    "moreover",
    "tapestry",
    "furthermore",
    "additionally",
    "synergies",
    "synergy",
    "robust",
    "unlock",
    "unlocking",
    "nestled",
    "testament",
    "vibrant",
    "cutting-edge",
    "paradigm",
    "paradigms",
    "in today's fast-paced world",
    "in today's digital world",
    "in today's landscape",
    "it is important to note",
)

_CONTRACTION_RE = re.compile(
    r"\b(?:I'm|I've|I'd|I'll|you're|you've|you'd|you'll|we're|we've|we'd|we'll|"
    r"they're|they've|they'd|they'll|it's|that's|what's|who's|there's|here's|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|couldn't|won't|"
    r"wouldn't|shouldn't|haven't|hasn't|hadn't|mustn't)\b",
    flags=re.I,
)


def style_profile_path(paths: ProfilePaths) -> Path:
    """Local profile path for the built style card (never committed)."""
    return paths.root / "style_profile.json"


def contraction_rate(text: str) -> float:
    """Share of word tokens that are common English contractions."""
    words = _word_tokens(text)
    if not words:
        return 0.0
    hits = len(_CONTRACTION_RE.findall(text or ""))
    return round(hits / len(words), 4)


def text_style_axes(text: str) -> dict[str, Any]:
    """Per-piece axes aggregated into the corpus style profile."""
    body = (text or "").strip()
    words = _word_tokens(body)
    n_words = max(1, len(words))
    sent = _sentence_word_counts(body)
    if sent:
        ordered = sorted(sent)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median = float(ordered[mid])
        else:
            median = (ordered[mid - 1] + ordered[mid]) / 2.0
    else:
        median = 0.0
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    short = sum(1 for ln in lines if len(_word_tokens(ln)) <= 8)
    short_ratio = (short / len(lines)) if lines else 0.0
    you_n = len(re.findall(r"\byou\b", body, flags=re.I))
    i_n = len(re.findall(r"\bI\b", body))
    return {
        "words": len(words),
        "median_sentence_words": round(median, 2),
        "short_line_ratio": round(short_ratio, 4),
        "contraction_rate": contraction_rate(body),
        "you_count": you_n,
        "i_count": i_n,
        "you_per_1k": round(you_n * 1000.0 / n_words, 2),
        "i_per_1k": round(i_n * 1000.0 / n_words, 2),
        "you_gt_i": you_n > i_n,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; deterministic and dependency-free."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def sentence_length_spread(texts: Iterable[str]) -> dict[str, float]:
    """Quartiles of sentence length across the corpus.

    A single median told the model to make every sentence that length, which
    reads as a caricature. The spread is what lets it vary deliberately.
    """
    lengths: list[float] = []
    for text in texts:
        lengths.extend(float(n) for n in _sentence_word_counts(text or "") if n)
    return {
        "sentence_words_p25": round(_percentile(lengths, 0.25), 1),
        "sentence_words_p75": round(_percentile(lengths, 0.75), 1),
    }


def multi_sentence_paragraph_ratio(texts: Iterable[str]) -> float:
    """Share of paragraphs carrying more than one sentence.

    Paragraphs are split on any run of newlines. The corpus separates them with
    a single newline while generated drafts use a blank line; splitting only on
    blank lines counted each stored post as one paragraph and reported 100%.
    """
    total = 0
    multi = 0
    for text in texts:
        for block in re.split(r"\n+", text or ""):
            stripped = block.strip()
            if not stripped:
                continue
            total += 1
            if len(_sentence_word_counts(stripped)) > 1:
                multi += 1
    return round(multi / total, 4) if total else 0.0


def corpus_style_stats(texts: Iterable[str]) -> dict[str, Any]:
    """Aggregate deterministic stats across corpus texts."""
    axes = [text_style_axes(t) for t in texts if (t or "").strip()]
    if not axes:
        return {
            "pieces": 0,
            "words": 0,
            "median_sentence_words": 0.0,
            "short_line_ratio": 0.0,
            "contraction_rate": 0.0,
            "you_count": 0,
            "i_count": 0,
            "you_per_1k": 0.0,
            "i_per_1k": 0.0,
            "you_gt_i": False,
        }

    total_words = sum(int(a["words"]) for a in axes)
    you_n = sum(int(a["you_count"]) for a in axes)
    i_n = sum(int(a["i_count"]) for a in axes)
    n_words = max(1, total_words)
    return {
        "pieces": len(axes),
        "words": total_words,
        "median_sentence_words": round(
            _median([float(a["median_sentence_words"]) for a in axes]), 2
        ),
        "short_line_ratio": round(
            _median([float(a["short_line_ratio"]) for a in axes]), 4
        ),
        "contraction_rate": round(
            _median([float(a["contraction_rate"]) for a in axes]), 4
        ),
        "you_count": you_n,
        "i_count": i_n,
        "you_per_1k": round(you_n * 1000.0 / n_words, 2),
        "i_per_1k": round(i_n * 1000.0 / n_words, 2),
        "you_gt_i": you_n > i_n,
    }


# Short posts and comments pull medians down; length targets come from posts
# that are already post-shaped. LinkedIn's ~3000 character limit is ~480–550
# words — that is the hard ceiling for the post channel.
_POST_LENGTH_SOURCES = frozenset({"linkedin_post"})
_MIN_LENGTH_SAMPLE_WORDS = 80
LINKEDIN_POST_WORD_CEILING = 550
DEFAULT_DRAFT_WORD_TARGET = 500
DEFAULT_DRAFT_WORD_FLOOR = 300


def post_length_stats(pieces: Iterable[Piece]) -> dict[str, float]:
    """Word-length percentiles from post-shaped pieces only.

    Falls back to all non-empty pieces when no linkedin_post rows qualify, so
    Contoso fixtures and note-only corpora still get a length card.
    """
    piece_list = list(pieces)
    posts = [
        p
        for p in piece_list
        if p.source in _POST_LENGTH_SOURCES
        and len(_word_tokens(p.text or "")) >= _MIN_LENGTH_SAMPLE_WORDS
    ]
    if not posts:
        posts = [p for p in piece_list if (p.text or "").strip()]
    lengths = [float(len(_word_tokens(p.text or ""))) for p in posts]
    if not lengths:
        return {
            "median_post_words": 0.0,
            "post_words_p75": 0.0,
            "post_words_p90": 0.0,
            "post_length_samples": 0.0,
        }
    return {
        "median_post_words": round(_median(lengths), 1),
        "post_words_p75": round(_percentile(lengths, 0.75), 1),
        "post_words_p90": round(_percentile(lengths, 0.90), 1),
        "post_length_samples": float(len(lengths)),
    }


# Articles are a different length regime, so their targets come from
# article-shaped pieces only. Falling back to the post band is what produced
# ~500-word "articles": the post ceiling is a LinkedIn character limit, not
# anything the author's longform does.
_ARTICLE_LENGTH_SOURCES = frozenset({"linkedin_article"})
_MIN_ARTICLE_SAMPLE_WORDS = 150
# Used only when the corpus has no article-shaped pieces to measure.
DEFAULT_ARTICLE_WORD_AIM = 1100
ARTICLE_WORD_FLOOR = 600
ARTICLE_WORD_CEILING = 3000
# Section budgets. Below the floor a section is a paragraph, above the ceiling
# the model stops writing sections and writes one undifferentiated essay.
MIN_ARTICLE_SECTION_WORDS = 180
MAX_ARTICLE_SECTION_WORDS = 600
# Words a section of a longform piece typically carries, used to turn a total
# length into a plausible section count.
TYPICAL_ARTICLE_SECTION_WORDS = 300
MIN_ARTICLE_SECTION_HINT = 2
MAX_ARTICLE_SECTION_HINT = 8


def article_length_stats(pieces: Iterable[Piece]) -> dict[str, float]:
    """Word-length percentiles from article-shaped pieces only.

    Returns zeros when the corpus carries no articles rather than borrowing the
    post band: an article target derived from posts is not a measurement of the
    author's longform, and the callers below fall back to a stated default.
    """
    articles = [
        p
        for p in pieces
        if p.source in _ARTICLE_LENGTH_SOURCES
        and len(_word_tokens(p.text or "")) >= _MIN_ARTICLE_SAMPLE_WORDS
    ]
    lengths = [float(len(_word_tokens(p.text or ""))) for p in articles]
    if not lengths:
        return {
            "median_article_words": 0.0,
            "article_words_p75": 0.0,
            "article_words_p90": 0.0,
            "article_length_samples": 0.0,
        }
    return {
        "median_article_words": round(_median(lengths), 1),
        "article_words_p75": round(_percentile(lengths, 0.75), 1),
        "article_words_p90": round(_percentile(lengths, 0.90), 1),
        "article_length_samples": float(len(lengths)),
    }


def article_word_aim(profile: dict[str, Any]) -> int:
    """Typical finished article length, stated in the prompt.

    The median is the aim because a longform piece should read like the
    author's usual article, not like their longest one.
    """
    stats = profile.get("stats") or {}
    for key in ("median_article_words", "article_words_p75", "article_words_p90"):
        value = float(stats.get(key) or 0)
        if value > 0:
            return int(min(ARTICLE_WORD_CEILING, max(ARTICLE_WORD_FLOOR, round(value))))
    return DEFAULT_ARTICLE_WORD_AIM


def article_word_target(profile: dict[str, Any]) -> int:
    """Hard word ceiling for a finished article (the author's long band)."""
    stats = profile.get("stats") or {}
    aim = article_word_aim(profile)
    for key in ("article_words_p90", "article_words_p75", "median_article_words"):
        value = float(stats.get(key) or 0)
        if value > 0:
            return int(min(ARTICLE_WORD_CEILING, max(aim, round(value))))
    return max(aim, DEFAULT_ARTICLE_WORD_AIM)


def article_section_count_hint(profile: dict[str, Any]) -> int:
    """Sections an article of the author's typical length would carry."""
    sections = round(article_word_aim(profile) / TYPICAL_ARTICLE_SECTION_WORDS)
    return max(MIN_ARTICLE_SECTION_HINT, min(MAX_ARTICLE_SECTION_HINT, int(sections)))


def article_section_words(profile: dict[str, Any], *, sections: int) -> int:
    """Per-section word budget that adds up to the author's article length."""
    count = max(1, int(sections))
    per_section = round(article_word_aim(profile) / count)
    return int(
        min(MAX_ARTICLE_SECTION_WORDS, max(MIN_ARTICLE_SECTION_WORDS, per_section))
    )


# Below this, article cadence is noise — fall back to the corpus-wide card.
MIN_ARTICLE_CADENCE_SAMPLES = 3


def article_cadence_stats(pieces: Iterable[Piece]) -> dict[str, Any]:
    """Cadence measured on linkedin_article pieces only.

    The corpus-wide card is dominated by short posts and comments. Feeding that
    into the article channel produces punchy LinkedIn-slop sentences instead of
    the author's longform rhythm — which is exactly the complaint a real article
    draft gets when the style card says most sentences run 3–14 words.
    """
    articles = [
        p
        for p in pieces
        if p.source in _ARTICLE_LENGTH_SOURCES and (p.text or "").strip()
    ]
    if len(articles) < MIN_ARTICLE_CADENCE_SAMPLES:
        return {"article_cadence_samples": float(len(articles))}
    # Article exports often carry Medium/Ghost CSS ahead of the body. Measuring
    # that raw paste reports 2–8 word "sentences" and 80%+ short lines — which
    # then tells the article channel to write LinkedIn-comment slop.
    texts: list[str] = []
    for piece in articles:
        cleaned = normalize_corpus_text(piece.text)
        if cleaned.strip():
            texts.append(cleaned)
    if len(texts) < MIN_ARTICLE_CADENCE_SAMPLES:
        return {"article_cadence_samples": float(len(texts))}
    axes = corpus_style_stats(texts)
    spread = sentence_length_spread(texts)
    return {
        "article_cadence_samples": float(len(texts)),
        "article_sentence_words_p25": spread["sentence_words_p25"],
        "article_sentence_words_p75": spread["sentence_words_p75"],
        "article_median_sentence_words": float(axes.get("median_sentence_words") or 0),
        "article_short_line_ratio": float(axes.get("short_line_ratio") or 0),
        "article_multi_sentence_paragraph_ratio": multi_sentence_paragraph_ratio(texts),
        "article_you_count": int(axes.get("you_count") or 0),
        "article_i_count": int(axes.get("i_count") or 0),
        "article_you_gt_i": bool(axes.get("you_gt_i")),
        "article_contraction_rate": float(axes.get("contraction_rate") or 0),
    }


def build_style_profile(
    pieces: Iterable[Piece],
    *,
    banned_phrases: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the style_profile.json payload from corpus pieces."""
    piece_list = list(pieces)
    stats = corpus_style_stats(p.text for p in piece_list)
    banned = list(banned_phrases) if banned_phrases is not None else list(BANNED_AI_FILLER)
    texts = [p.text for p in piece_list if (p.text or "").strip()]
    stats.update(post_length_stats(piece_list))
    stats.update(article_length_stats(piece_list))
    stats.update(article_cadence_stats(piece_list))
    stats.update(sentence_length_spread(texts))
    stats["multi_sentence_paragraph_ratio"] = multi_sentence_paragraph_ratio(texts)
    return {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "piece_ids": [p.id for p in piece_list],
        "stats": stats,
        "banned_ai_filler": banned,
    }


def draft_word_target(profile: dict[str, Any]) -> int:
    """Word ceiling for a finished post draft.

    Prefer the author's long-post band (p90, else p75, else median) so short
    comments in the selection cannot shrink drafts to stub length. Clamp to the
    LinkedIn character budget in words.
    """
    stats = profile.get("stats") or {}
    for key in ("post_words_p90", "post_words_p75", "median_post_words"):
        value = float(stats.get(key) or 0)
        if value > 0:
            target = int(round(max(value, DEFAULT_DRAFT_WORD_FLOOR)))
            return min(LINKEDIN_POST_WORD_CEILING, target)
    return DEFAULT_DRAFT_WORD_TARGET


def draft_word_aim(profile: dict[str, Any]) -> int:
    """Typical finished length stated in the prompt (below the hard ceiling)."""
    stats = profile.get("stats") or {}
    for key in ("post_words_p75", "post_words_p90", "median_post_words"):
        value = float(stats.get(key) or 0)
        if value > 0:
            aim = int(round(max(value, DEFAULT_DRAFT_WORD_FLOOR)))
            return min(draft_word_target(profile), aim)
    return min(DEFAULT_DRAFT_WORD_TARGET, draft_word_target(profile))


def style_directives(profile: dict[str, Any], *, channel: str = "post") -> list[str]:
    """Render the style card as prompt directives.

    Derived numbers, not the author's sentences. The exemplar path hands the
    model copyable text and it copies; cadence targets carry the same voice
    signal with nothing to paste.

    ``channel='article'`` drops the post length directive and, when enough
    articles were measured, uses article-only cadence instead of the
    comment-dominated corpus card. A word budget never transfers; cadence only
    transfers when there is no article sample to speak for itself.
    """
    stats = profile.get("stats") or {}
    is_article = (channel or "post").strip().lower() == "article"
    use_article_cadence = (
        is_article
        and float(stats.get("article_cadence_samples") or 0) >= MIN_ARTICLE_CADENCE_SAMPLES
    )
    directives: list[str] = []

    if use_article_cadence:
        low = float(stats.get("article_sentence_words_p25") or 0)
        high = float(stats.get("article_sentence_words_p75") or 0)
        median_sentence = float(stats.get("article_median_sentence_words") or 0)
        short_ratio = float(stats.get("article_short_line_ratio") or 0)
        multi_ratio = float(stats.get("article_multi_sentence_paragraph_ratio") or 0)
        you_n = int(stats.get("article_you_count") or 0)
        i_n = int(stats.get("article_i_count") or 0)
        you_gt_i = bool(stats.get("article_you_gt_i"))
        contraction_rate = float(stats.get("article_contraction_rate") or 0)
    else:
        low = float(stats.get("sentence_words_p25") or 0)
        high = float(stats.get("sentence_words_p75") or 0)
        median_sentence = float(stats.get("median_sentence_words") or 0)
        short_ratio = float(stats.get("short_line_ratio") or 0)
        multi_ratio = float(stats.get("multi_sentence_paragraph_ratio") or 0)
        you_n = int(stats.get("you_count") or 0)
        i_n = int(stats.get("i_count") or 0)
        you_gt_i = bool(stats.get("you_gt_i"))
        contraction_rate = float(stats.get("contraction_rate") or 0)

    if low and high and high > low:
        directives.append(
            f"Sentence length varies: most run {low:.0f}–{high:.0f} words. "
            "Do not write every sentence the same length."
        )
    elif median_sentence:
        directives.append(
            f"Sentences average about {median_sentence:.0f} words. Vary them, "
            "but do not write long academic sentences."
        )

    # Article exports mix short headings with long prose. Prefer the
    # multi-sentence paragraph signal when it is present, or the short-line
    # directive turns every section into comment-length punch lines.
    if use_article_cadence and multi_ratio >= 0.15:
        directives.append(
            f"About {multi_ratio * 100:.0f}% of paragraphs carry two or more "
            "sentences. Write in prose paragraphs, not a stack of one-liners."
        )
        if short_ratio >= 0.55:
            directives.append(
                f"About {short_ratio * 100:.0f}% of lines are 8 words or fewer. "
                "Short lines are fine for emphasis, not for the whole article."
            )
    else:
        if short_ratio:
            directives.append(
                f"About {short_ratio * 100:.0f}% of lines are 8 words or fewer. "
                "Use short standalone lines and frequent paragraph breaks."
            )
        if multi_ratio:
            directives.append(
                f"About {multi_ratio * 100:.0f}% of paragraphs carry two or more "
                "sentences. Do not write the whole post as single short lines."
            )

    median_post = float(stats.get("median_post_words") or 0)
    if median_post and not is_article:
        directives.append(
            f"Target roughly {median_post:.0f} words total. Never exceed "
            f"{draft_word_target(profile)} words. Stop when the point is made."
        )

    # Absent counts must stay silent: an empty profile asserting a pronoun lean
    # would put a made-up voice rule in the prompt. On articles, a pure you>i
    # rule erases first-person experience voice when both pronouns are real.
    if you_n or i_n:
        if use_article_cadence and you_n and i_n:
            directives.append(
                "Use first person for experience and judgment; use 'you' when "
                "giving the reader a direct instruction."
            )
        elif you_gt_i:
            directives.append("Address the reader as 'you' more often than 'I'.")
        else:
            directives.append("Speak in first person more often than addressing 'you'.")

    if contraction_rate > 0.01:
        directives.append("Use contractions; write the way people speak.")

    banned = [str(phrase) for phrase in (profile.get("banned_ai_filler") or [])][:12]
    if banned:
        directives.append("Never use these words: " + ", ".join(banned) + ".")

    return directives


def save_style_profile(paths: ProfilePaths, profile: dict[str, Any]) -> Path:
    """Write style_profile.json under the local profile root."""
    paths.ensure()
    out = style_profile_path(paths)
    out.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_style_profile(paths: ProfilePaths) -> dict[str, Any]:
    path = style_profile_path(paths)
    if not path.is_file():
        raise FileNotFoundError(
            f"No style profile at {path}. Run: personality-protect build-style-profile"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_build_style_profile(paths: ProfilePaths) -> tuple[dict[str, Any], Path]:
    """Build + save style profile from the current selection."""
    pieces = selected_pieces(paths)
    if not pieces:
        raise FileNotFoundError(
            f"Selection at {paths.selection_path} resolved to 0 pieces. "
            "Run: personality-protect select"
        )
    profile = build_style_profile(pieces)
    out = save_style_profile(paths, profile)
    return profile, out
