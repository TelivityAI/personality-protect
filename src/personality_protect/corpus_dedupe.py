"""Collapse corpus pieces that repeat the same text under different ids.

Piece ids are derived partly from the source path, so the same post ingested
from two export paths (a fresh scrape plus the periodic full export) lands twice
with two ids and ``append_index``'s id check cannot see it. Text is the only
identity that survives a re-ingest, so dedupe compares normalized text.

Near-duplicates matter for the same reason: two captures of one post differ by a
trailing link or hashtag, so an exact key alone leaves the pair in the corpus and
double-weights it in retrieval and in the style stats.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.models import Piece

# Same post captured twice differs only in trailing link/hashtag noise. Anything
# looser starts merging distinct posts that share a template opening.
DEFAULT_NEAR_RATIO = 0.99
# Most specific/valuable source wins when identical text spans sources.
SOURCE_PRIORITY: tuple[str, ...] = ("linkedin_article", "linkedin_post", "linkedin_comment")
_WHITESPACE_RE = re.compile(r"\s+")


def duplicate_key(text: str) -> str:
    """Comparison key: cleaned corpus text, whitespace-collapsed, case-folded."""
    return _WHITESPACE_RE.sub(" ", normalize_corpus_text(text)).casefold().strip()


def _source_rank(source: str) -> int:
    try:
        return SOURCE_PRIORITY.index(source)
    except ValueError:
        return len(SOURCE_PRIORITY)


def _keeper_sort_key(piece: Piece, holdout_ids: frozenset[str], key: str) -> tuple[Any, ...]:
    return (
        0 if piece.id in holdout_ids else 1,
        _source_rank(piece.source),
        0 if piece.date else 1,
        -len(key),  # the fuller capture of a near-duplicate pair
        -len((piece.title or "").strip()),
        -len(piece.meta or {}),
        piece.id,
    )


@dataclass
class DuplicateGroup:
    """One set of pieces holding the same text, with the chosen keeper first."""

    keeper: Piece
    dropped: list[Piece]
    exact: bool

    @property
    def members(self) -> list[Piece]:
        return [self.keeper, *self.dropped]

    @property
    def sources(self) -> list[str]:
        return sorted({piece.source for piece in self.members})

    @property
    def cross_source(self) -> bool:
        return len(self.sources) > 1

    def to_report(self) -> dict[str, Any]:
        return {
            "keeper": self.keeper.id,
            "keeper_source": self.keeper.source,
            "keeper_words": self.keeper.word_count,
            "dropped": [piece.id for piece in self.dropped],
            "sources": self.sources,
            "cross_source": self.cross_source,
            "exact": self.exact,
        }


@dataclass
class DedupeResult:
    kept: list[Piece]
    groups: list[DuplicateGroup] = field(default_factory=list)

    @property
    def dropped_ids(self) -> list[str]:
        return [piece.id for group in self.groups for piece in group.dropped]

    def to_report(self) -> dict[str, Any]:
        dropped_by_source: dict[str, int] = defaultdict(int)
        for group in self.groups:
            for piece in group.dropped:
                dropped_by_source[piece.source] += 1
        return {
            "groups": len(self.groups),
            "exact_groups": sum(1 for group in self.groups if group.exact),
            "near_groups": sum(1 for group in self.groups if not group.exact),
            "dropped": len(self.dropped_ids),
            "dropped_by_source": dict(sorted(dropped_by_source.items())),
            "cross_source_groups": [
                group.to_report() for group in self.groups if group.cross_source
            ],
            "group_reports": [group.to_report() for group in self.groups],
        }


def _near_duplicate_links(
    pieces: Sequence[Piece],
    keys: dict[str, str],
    near_ratio: float,
) -> list[tuple[str, str]]:
    """Pair ids whose text is near-identical, comparing only plausible candidates.

    Candidates share a source and date, which is what a re-ingested capture of
    the same piece looks like; that keeps the quadratic comparison inside small
    buckets instead of across the whole corpus.
    """
    buckets: dict[tuple[str, str], list[Piece]] = defaultdict(list)
    for piece in pieces:
        if keys[piece.id]:
            buckets[(piece.source, piece.date or "")].append(piece)

    links: list[tuple[str, str]] = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        ordered = sorted(bucket, key=lambda piece: piece.id)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                matcher = SequenceMatcher(None, keys[left.id], keys[right.id])
                # Cheap upper bounds first: length, then shared character counts.
                if matcher.real_quick_ratio() < near_ratio:
                    continue
                if matcher.quick_ratio() < near_ratio:
                    continue
                if matcher.ratio() >= near_ratio:
                    links.append((left.id, right.id))
    return links


def find_duplicate_groups(
    pieces: Iterable[Piece],
    *,
    holdout_ids: Iterable[str] = (),
    near_ratio: float | None = None,
) -> list[DuplicateGroup]:
    """Group pieces sharing text and pick a deterministic keeper for each group.

    ``near_ratio`` also folds in near-identical captures of the same piece; pass
    ``None`` to compare exact normalized text only.
    """
    pieces = list(pieces)
    holdouts = frozenset(str(piece_id) for piece_id in holdout_ids)
    keys = {piece.id: duplicate_key(piece.text) for piece in pieces}

    parent: dict[str, str] = {piece.id: piece.id for piece in pieces}

    def find(piece_id: str) -> str:
        while parent[piece_id] != piece_id:
            parent[piece_id] = parent[parent[piece_id]]
            piece_id = parent[piece_id]
        return piece_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    exact_members: dict[str, list[str]] = defaultdict(list)
    for piece in pieces:
        if keys[piece.id]:
            exact_members[keys[piece.id]].append(piece.id)
    for members in exact_members.values():
        for piece_id in members[1:]:
            union(members[0], piece_id)

    exact_ids = {
        piece_id
        for members in exact_members.values()
        if len(members) > 1
        for piece_id in members
    }
    if near_ratio is not None:
        for left, right in _near_duplicate_links(pieces, keys, near_ratio):
            union(left, right)

    by_root: dict[str, list[Piece]] = defaultdict(list)
    for piece in pieces:
        if keys[piece.id]:
            by_root[find(piece.id)].append(piece)

    groups: list[DuplicateGroup] = []
    for members in by_root.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda p: _keeper_sort_key(p, holdouts, keys[p.id]))
        groups.append(
            DuplicateGroup(
                keeper=ordered[0],
                dropped=ordered[1:],
                exact=all(piece.id in exact_ids for piece in members)
                and len({keys[piece.id] for piece in members}) == 1,
            )
        )
    groups.sort(key=lambda group: group.keeper.id)
    return groups


def dedupe_pieces(
    pieces: Iterable[Piece],
    *,
    holdout_ids: Iterable[str] = (),
    near_ratio: float | None = None,
) -> DedupeResult:
    """Drop duplicate-text pieces, keeping input order and every holdout id."""
    pieces = list(pieces)
    holdouts = frozenset(str(piece_id) for piece_id in holdout_ids)
    groups = find_duplicate_groups(pieces, holdout_ids=holdouts, near_ratio=near_ratio)

    dropped = {piece.id for group in groups for piece in group.dropped}
    protected = dropped & holdouts
    if protected:
        raise AssertionError(
            "refusing to drop holdout ids: " + ", ".join(sorted(protected))
        )
    return DedupeResult(
        kept=[piece for piece in pieces if piece.id not in dropped],
        groups=groups,
    )
