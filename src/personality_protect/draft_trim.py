"""Trim generated drafts to a stopping point the author would have chosen.

A 9B model handed a short brief writes a strong opening and then keeps going,
recycling its own lines until the token budget runs out ("Say no. Say no. Say
no."). Lowering ``max_tokens`` only moves the cut mid-sentence, so the tail is
removed after generation instead: drop paragraphs that restate an earlier one,
then stop at the last paragraph boundary inside the word target.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
# Above this, a paragraph is restating one already written rather than adding to
# it. Deliberately high: an author may legitimately reuse a short refrain.
NEAR_DUPLICATE_RATIO = 0.8


def _paragraphs(text: str) -> list[str]:
    return [block.strip() for block in _PARAGRAPH_SPLIT.split(text or "") if block.strip()]


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD_RE.finditer(text or "")]


def word_count(text: str) -> int:
    return len(_tokens(text))


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def drop_repeated_paragraphs(text: str) -> str:
    """Remove paragraphs that restate one already present."""
    seen: list[set[str]] = []
    kept: list[str] = []
    for block in _paragraphs(text):
        tokens = set(_tokens(block))
        if not tokens:
            continue
        if any(_similarity(tokens, earlier) >= NEAR_DUPLICATE_RATIO for earlier in seen):
            continue
        seen.append(tokens)
        kept.append(block)
    return "\n\n".join(kept)


# Section drafts are generated independently. A later section can rephrase an
# earlier one without any single paragraph hitting NEAR_DUPLICATE_RATIO, which
# is how a five-section stitch becomes the same argument five times. Coverage
# against the kept article catches that paraphrase; pairwise similarity catches
# near-copies of one prior section.
SECTION_RESTATE_SIMILARITY = 0.55
SECTION_RESTATE_COVERAGE = 0.62


def drop_restated_sections(sections: list[str] | tuple[str, ...]) -> list[str]:
    """Keep section drafts that still add substance after earlier ones."""
    kept: list[str] = []
    kept_token_sets: list[set[str]] = []
    for section in sections:
        text = (section or "").strip()
        if not text:
            continue
        tokens = set(_tokens(text))
        if not tokens:
            continue
        if any(
            _similarity(tokens, earlier) >= SECTION_RESTATE_SIMILARITY
            for earlier in kept_token_sets
        ):
            continue
        if kept_token_sets:
            prior = set().union(*kept_token_sets)
            if len(tokens & prior) / len(tokens) >= SECTION_RESTATE_COVERAGE:
                continue
        kept.append(text)
        kept_token_sets.append(tokens)
    return kept


def trim_to_word_target(text: str, max_words: int) -> str:
    """Keep whole paragraphs up to ``max_words``, never cutting mid-sentence.

    The first paragraph is always kept: returning an empty draft because the
    opening alone exceeds the target would hide the model's output rather than
    edit it.
    """
    blocks = _paragraphs(text)
    if not blocks or max_words <= 0:
        return "\n\n".join(blocks)

    kept = [blocks[0]]
    total = word_count(blocks[0])
    for block in blocks[1:]:
        words = word_count(block)
        if total + words > max_words:
            break
        kept.append(block)
        total += words
    return "\n\n".join(kept)


def trim_draft(text: str, *, max_words: int) -> str:
    """Drop restated paragraphs, then stop at the author's typical length."""
    return trim_to_word_target(drop_repeated_paragraphs(text), max_words).strip()
