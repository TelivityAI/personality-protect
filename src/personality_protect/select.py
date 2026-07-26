"""Select pieces for training with length/date filters and include/exclude."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from personality_protect.config import DEFAULT_MIN_WORDS, DEFAULT_THROUGH_YEAR, ProfilePaths, load_config
from personality_protect.models import Piece, load_index, summarize_by_source_year


@dataclass
class Selection:
    piece_ids: list[str] = field(default_factory=list)
    min_words: int = DEFAULT_MIN_WORDS
    through_year: int = DEFAULT_THROUGH_YEAR
    include_undated: bool = False
    include_ids: list[str] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Selection:
        return cls(
            piece_ids=list(data.get("piece_ids", [])),
            min_words=int(data.get("min_words", DEFAULT_MIN_WORDS)),
            through_year=int(data.get("through_year", DEFAULT_THROUGH_YEAR)),
            include_undated=bool(data.get("include_undated", False)),
            include_ids=list(data.get("include_ids", [])),
            exclude_ids=list(data.get("exclude_ids", [])),
            summary=dict(data.get("summary") or {}),
        )


def filter_pieces(
    pieces: list[Piece],
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    through_year: int = DEFAULT_THROUGH_YEAR,
    include_undated: bool = False,
    include_ids: list[str] | None = None,
    exclude_ids: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[Piece]:
    include_ids = include_ids or []
    exclude_ids = set(exclude_ids or [])
    include_set = set(include_ids)
    source_set = set(sources) if sources else None

    selected: list[Piece] = []
    for p in pieces:
        if p.id in exclude_ids:
            continue
        if source_set is not None and p.source not in source_set:
            if p.id not in include_set:
                continue
        # Explicit include bypasses length/date defaults
        if p.id in include_set:
            selected.append(p)
            continue
        if p.word_count < min_words:
            continue
        if p.year is None:
            if not include_undated:
                continue
        elif p.year > through_year:
            continue
        selected.append(p)
    return selected


def run_select(
    paths: ProfilePaths,
    *,
    min_words: int | None = None,
    through_year: int | None = None,
    include_undated: bool = False,
    include_ids: list[str] | None = None,
    exclude_ids: list[str] | None = None,
    sources: list[str] | None = None,
) -> tuple[Selection, list[Piece]]:
    config = load_config(paths)
    pieces = load_index(paths.index_path)
    mw = min_words if min_words is not None else config.min_words
    ty = through_year if through_year is not None else config.through_year

    selected = filter_pieces(
        pieces,
        min_words=mw,
        through_year=ty,
        include_undated=include_undated,
        include_ids=include_ids,
        exclude_ids=exclude_ids,
        sources=sources,
    )
    summary = summarize_by_source_year(selected)
    undated = [p for p in pieces if p.year is None]
    summary["undated_in_index"] = len(undated)
    summary["index_total"] = len(pieces)

    selection = Selection(
        piece_ids=[p.id for p in selected],
        min_words=mw,
        through_year=ty,
        include_undated=include_undated,
        include_ids=list(include_ids or []),
        exclude_ids=list(exclude_ids or []),
        summary=summary,
    )
    paths.selection_path.write_text(
        json.dumps(selection.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return selection, selected


def load_selection(paths: ProfilePaths) -> Selection:
    if not paths.selection_path.is_file():
        raise FileNotFoundError(
            f"No selection at {paths.selection_path}. Run: personality-protect select"
        )
    return Selection.from_dict(json.loads(paths.selection_path.read_text(encoding="utf-8")))


def selected_pieces(paths: ProfilePaths) -> list[Piece]:
    selection = load_selection(paths)
    by_id = {p.id: p for p in load_index(paths.index_path)}
    return [by_id[i] for i in selection.piece_ids if i in by_id]
