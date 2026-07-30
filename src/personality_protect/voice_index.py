"""Build and query the local retrieval index for writing exemplars."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from personality_protect.config import DEFAULT_PROFILE, ProfilePaths, get_paths
from personality_protect.embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    EMBEDDER_NAME,
    LocalEmbedder,
    cosine_similarity,
)
from personality_protect.models import Piece, load_index

VOICE_INDEX_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
VECTORS_FILENAME = "vectors.jsonl"


def _voice_index_dir(paths: ProfilePaths) -> Path:
    return paths.root / "voice_index"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _vector_row(piece: Piece, vector: list[float]) -> dict[str, Any]:
    return {"piece": piece.to_dict(), "vector": vector}


def build_voice_index(
    paths: ProfilePaths,
    *,
    holdout_ids: Iterable[str] = (),
    embedder: LocalEmbedder | None = None,
) -> dict[str, Any]:
    """Replace the profile voice index from the current corpus index."""
    selected_embedder = embedder or LocalEmbedder()
    holdouts = {str(piece_id) for piece_id in holdout_ids}
    pieces = load_index(paths.index_path)
    indexed = sorted((piece for piece in pieces if piece.id not in holdouts), key=lambda p: p.id)

    rows = [
        _vector_row(piece, selected_embedder.embed(piece.text))
        for piece in indexed
    ]
    vectors_content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    manifest = {
        "schema_version": VOICE_INDEX_SCHEMA_VERSION,
        "embedder": {
            "name": selected_embedder.name,
            "dimensions": selected_embedder.dimensions,
        },
        "vectors": VECTORS_FILENAME,
        "indexed": len(rows),
    }
    manifest_content = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    directory = _voice_index_dir(paths)
    _atomic_write(directory / VECTORS_FILENAME, vectors_content)
    _atomic_write(directory / MANIFEST_FILENAME, manifest_content)
    return {
        "voice_index": str(directory),
        "source_total": len(pieces),
        "indexed": len(rows),
        "skipped_holdout": sum(piece.id in holdouts for piece in pieces),
        "schema_version": VOICE_INDEX_SCHEMA_VERSION,
        "embedder": selected_embedder.name,
        "dimensions": selected_embedder.dimensions,
    }


def _load_voice_index(paths: ProfilePaths) -> tuple[LocalEmbedder, list[dict[str, Any]]]:
    directory = _voice_index_dir(paths)
    manifest_path = directory / MANIFEST_FILENAME
    vectors_path = directory / VECTORS_FILENAME
    if not manifest_path.is_file() or not vectors_path.is_file():
        raise FileNotFoundError(
            f"No voice index at {directory}. Run: personality-protect index-voice"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VOICE_INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported voice index schema: {manifest.get('schema_version')}")
    embedder_meta = manifest.get("embedder") or {}
    if embedder_meta.get("name") != EMBEDDER_NAME:
        raise ValueError(f"Unsupported voice index embedder: {embedder_meta.get('name')}")
    dimensions = int(embedder_meta.get("dimensions") or DEFAULT_EMBEDDING_DIMENSIONS)

    rows: list[dict[str, Any]] = []
    with vectors_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return LocalEmbedder(dimensions=dimensions), rows


def retrieve(
    brief: str,
    k: int = 5,
    *,
    profile: str = DEFAULT_PROFILE,
    home: Path | None = None,
) -> list[dict[str, Any]]:
    """Return up to k indexed exemplars ranked by similarity to the brief."""
    if k <= 0:
        return []
    paths = get_paths(profile, home=home)
    embedder, rows = _load_voice_index(paths)
    query = embedder.embed(brief)

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        piece = dict(row["piece"])
        score = cosine_similarity(query, row["vector"])
        scored.append((score, piece))
    scored.sort(key=lambda item: (-item[0], str(item[1]["id"])))

    matches: list[dict[str, Any]] = []
    for score, piece in scored[:k]:
        matches.append({**piece, "score": score})
    return matches
