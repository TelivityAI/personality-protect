"""Deterministic safety guards for exemplar-assisted writing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from personality_protect.eval_compare import extract_evidence_number_keys

DEFAULT_ENTITY_MASK = "[ENTITY]"
DEFAULT_PARROT_NGRAM = 8

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z](?:&[A-Z])+|"
    r"[A-Z][a-z][a-zA-Z0-9'-]*(?:\s+[A-Z][a-z][a-zA-Z0-9'-]*)*)\b"
)

# Sentence-initial words are syntactically capitalized, not necessarily names.
# Keep this deliberately small and generic; unknown capitalized words are safer
# to mask than to leak from a retrieved exemplar.
_NON_ENTITY_CAPS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "here",
    "how",
    "i",
    "if",
    "in",
    "it",
    "its",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "so",
    "stop",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "those",
    "to",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "we",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class InventionResult:
    """Entities and evidence figures a draft added beyond its brief."""

    invented_entities: frozenset[str]
    invented_numbers: frozenset[str]

    @property
    def passed(self) -> bool:
        return not self.invented_entities and not self.invented_numbers


def _entity_key(span: str) -> str:
    return re.sub(r"\s+", " ", span.strip()).casefold()


def _entity_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _ENTITY_RE.finditer(text or ""):
        span = match.group(0)
        if span.casefold() in _NON_ENTITY_CAPS:
            continue
        spans.append((match.start(), match.end(), _entity_key(span)))
    return spans


def extract_entity_keys(text: str) -> set[str]:
    """Return case-insensitive proper-name proxies from text."""

    return {key for _, _, key in _entity_spans(text)}


def mask_exemplar_entities(
    exemplar: str,
    brief: str,
    *,
    mask: str = DEFAULT_ENTITY_MASK,
) -> str:
    """Mask exemplar proper names that are not explicitly present in the brief."""

    if not exemplar:
        return ""
    allowed = extract_entity_keys(brief)
    pieces: list[str] = []
    cursor = 0
    for start, end, key in _entity_spans(exemplar):
        if key in allowed:
            continue
        pieces.append(exemplar[cursor:start])
        pieces.append(mask)
        cursor = end
    pieces.append(exemplar[cursor:])
    return "".join(pieces)


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD_RE.finditer(text or "")]


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1")
    tokens = _word_tokens(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def find_parroted_ngrams(
    draft: str,
    exemplars: Iterable[str],
    *,
    n: int = DEFAULT_PARROT_NGRAM,
) -> set[str]:
    """Return normalized draft n-grams copied from any exemplar."""

    draft_ngrams = _ngrams(draft, n)
    if not draft_ngrams:
        return set()
    exemplar_ngrams: set[tuple[str, ...]] = set()
    for exemplar in exemplars:
        exemplar_ngrams.update(_ngrams(exemplar, n))
    return {" ".join(ngram) for ngram in draft_ngrams & exemplar_ngrams}


def parrot_reject(
    draft: str,
    exemplars: Iterable[str],
    *,
    n: int = DEFAULT_PARROT_NGRAM,
) -> bool:
    """Reject a draft containing an exact normalized exemplar n-gram."""

    return bool(find_parroted_ngrams(draft, exemplars, n=n))


def check_invention(brief: str, draft: str) -> InventionResult:
    """Check draft entities and evidence figures against the brief only."""

    invented_entities = extract_entity_keys(draft) - extract_entity_keys(brief)
    invented_numbers = extract_evidence_number_keys(draft) - extract_evidence_number_keys(brief)
    return InventionResult(
        invented_entities=frozenset(invented_entities),
        invented_numbers=frozenset(invented_numbers),
    )
