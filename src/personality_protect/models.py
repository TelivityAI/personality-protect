"""Corpus piece models and local JSONL index I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Piece:
    """One writing sample registered in the local index."""

    id: str
    source: str  # linkedin_post | linkedin_comment | linkedin_article | email | doc | note | demo
    text: str
    path: str = ""  # original path (read in place); empty if from unpack cache
    date: str | None = None  # ISO date YYYY-MM-DD when known
    year: int | None = None
    word_count: int = 0
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if self.year is None and self.date and len(self.date) >= 4:
            try:
                self.year = int(self.date[:4])
            except ValueError:
                self.year = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Piece:
        return cls(
            id=str(data["id"]),
            source=str(data.get("source", "doc")),
            text=str(data.get("text", "")),
            path=str(data.get("path", "")),
            date=data.get("date"),
            year=data.get("year"),
            word_count=int(data.get("word_count") or 0),
            title=str(data.get("title", "")),
            meta=dict(data.get("meta") or {}),
        )


def load_index(path: Path) -> list[Piece]:
    if not path.is_file():
        return []
    pieces: list[Piece] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pieces.append(Piece.from_dict(json.loads(line)))
    return pieces


def save_index(path: Path, pieces: Iterable[Piece]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for piece in pieces:
            fh.write(json.dumps(piece.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def append_index(path: Path, pieces: Iterable[Piece]) -> int:
    """Append pieces, skipping duplicate ids."""
    existing = {p.id for p in load_index(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with path.open("a", encoding="utf-8") as fh:
        for piece in pieces:
            if piece.id in existing:
                continue
            fh.write(json.dumps(piece.to_dict(), ensure_ascii=False) + "\n")
            existing.add(piece.id)
            added += 1
    return added


def iter_index(path: Path) -> Iterator[Piece]:
    yield from load_index(path)


def summarize_by_source_year(pieces: Iterable[Piece]) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_year: dict[str, int] = {}
    total_words = 0
    n = 0
    for p in pieces:
        n += 1
        total_words += p.word_count
        by_source[p.source] = by_source.get(p.source, 0) + 1
        key = str(p.year) if p.year is not None else "undated"
        by_year[key] = by_year.get(key, 0) + 1
    return {
        "pieces": n,
        "words": total_words,
        "by_source": dict(sorted(by_source.items())),
        "by_year": dict(sorted(by_year.items(), key=lambda kv: (kv[0] == "undated", kv[0]))),
    }
