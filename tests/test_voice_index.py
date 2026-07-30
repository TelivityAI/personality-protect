"""Contoso-safe tests for the local RAG voice index."""

from __future__ import annotations

import json
import math
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import init_profile
from personality_protect.embedder import LocalEmbedder, cosine_similarity
from personality_protect.models import Piece, save_index
from personality_protect.voice_index import build_voice_index, retrieve

runner = CliRunner()


def _contoso_pieces() -> list[Piece]:
    return [
        Piece(
            id="contoso-pricing",
            source="linkedin_post",
            text="Pricing experiments work when teams compare customer value and renewal signals.",
        ),
        Piece(
            id="contoso-platform",
            source="linkedin_post",
            text="Platform migrations need careful service boundaries and boring rollback plans.",
        ),
        Piece(
            id="contoso-holdout",
            source="linkedin_post",
            text="Customer pricing and packaging lessons belong in the private holdout.",
        ),
    ]


def test_local_embedder_is_deterministic_normalized_and_topic_sensitive():
    embedder = LocalEmbedder(dimensions=128)

    first = embedder.embed("Contoso pricing experiments for customer value")
    again = embedder.embed("Contoso pricing experiments for customer value")
    related = embedder.embed("customer pricing and value")
    unrelated = embedder.embed("service migration rollback boundaries")

    assert first == again
    assert len(first) == 128
    assert math.isclose(sum(value * value for value in first), 1.0)
    assert cosine_similarity(first, related) > cosine_similarity(first, unrelated)


def test_rebuild_is_idempotent_skips_holdouts_and_drops_stale_vectors(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = _contoso_pieces()
    save_index(paths.index_path, pieces)

    result = build_voice_index(paths, holdout_ids={"contoso-holdout"})
    manifest_path = paths.root / "voice_index" / "manifest.json"
    vectors_path = paths.root / "voice_index" / "vectors.jsonl"
    first_manifest = manifest_path.read_bytes()
    first_vectors = vectors_path.read_bytes()

    assert result["indexed"] == 2
    assert result["skipped_holdout"] == 1
    assert json.loads(first_manifest)["schema_version"] == 2
    assert json.loads(first_manifest)["text_cleaner"] == "normalize_corpus_text"
    assert {
        json.loads(line)["piece"]["id"]
        for line in first_vectors.decode().splitlines()
    } == {"contoso-pricing", "contoso-platform"}

    build_voice_index(paths, holdout_ids={"contoso-holdout"})
    assert manifest_path.read_bytes() == first_manifest
    assert vectors_path.read_bytes() == first_vectors

    save_index(paths.index_path, [pieces[0], pieces[2]])
    build_voice_index(paths, holdout_ids={"contoso-holdout"})
    rows = [json.loads(line) for line in vectors_path.read_text().splitlines()]
    assert [row["piece"]["id"] for row in rows] == ["contoso-pricing"]


def test_build_cleans_text_before_embedding_and_storage(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    dirty = Piece(
        id="contoso-article",
        source="linkedin_article",
        text=(
            ".article { margin: 0 auto; width: 744px; }\n"
            "<article><p>Pricing should stay boring.</p>"
            "<p>Ship one clear test.</p></article>"
        ),
    )
    save_index(paths.index_path, [dirty])
    embedder = LocalEmbedder(dimensions=128)

    build_voice_index(paths, embedder=embedder)
    vectors_path = paths.root / "voice_index" / "vectors.jsonl"
    row = json.loads(vectors_path.read_text(encoding="utf-8"))

    assert row["piece"]["text"] == "Pricing should stay boring.\n\nShip one clear test."
    assert row["piece"]["word_count"] == 8
    assert row["vector"] == embedder.embed(row["piece"]["text"])
    assert "margin" not in row["piece"]["text"]
    assert "<article>" not in row["piece"]["text"]


def test_retrieve_returns_ranked_non_holdout_contoso_pieces(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    save_index(paths.index_path, _contoso_pieces())
    build_voice_index(paths, holdout_ids={"contoso-holdout"})

    matches = retrieve(
        "How should we test pricing against customer value?",
        k=2,
        profile="contoso",
        home=tmp_path,
    )

    assert [match["id"] for match in matches] == [
        "contoso-pricing",
        "contoso-platform",
    ]
    assert all("score" in match and "text" in match for match in matches)
    assert "contoso-holdout" not in {match["id"] for match in matches}


def test_cli_index_voice_reports_contoso_safe_rebuild(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    save_index(paths.index_path, _contoso_pieces())

    result = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "index-voice",
            "--profile",
            "contoso",
            "--home",
            str(tmp_path),
            "--holdout-id",
            "contoso-holdout",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["indexed"] == 2
    assert payload["skipped_holdout"] == 1
    assert payload["voice_index"] == str(paths.root / "voice_index")
