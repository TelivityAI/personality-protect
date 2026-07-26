"""Tests for ingest, select, and SFT builder."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from personality_protect.config import init_profile
from personality_protect.ingest import ingest_linkedin, ingest_local_paths, run_ingest
from personality_protect.models import load_index, summarize_by_source_year
from personality_protect.select import filter_pieces, run_select
from personality_protect.sft import build_sft_from_profile, piece_to_example

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
LINKEDIN = FIXTURES / "linkedin"
LOCAL = FIXTURES / "local_docs"


@pytest.fixture
def profile_home(tmp_path: Path):
    paths, _, _ = init_profile("test", home=tmp_path)
    return paths


def test_ingest_linkedin_csv(profile_home):
    pieces = ingest_linkedin(LINKEDIN, profile_home)
    sources = {p.source for p in pieces}
    assert "linkedin_post" in sources
    assert "linkedin_comment" in sources
    assert "linkedin_article" in sources
    assert all(p.word_count > 0 for p in pieces)
    assert any(p.year == 2023 for p in pieces)


def test_ingest_linkedin_zip(profile_home, tmp_path: Path):
    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in LINKEDIN.iterdir():
            zf.write(f, arcname=f.name)
    pieces = ingest_linkedin(zpath, profile_home)
    assert len(pieces) >= 3
    # unpacked under cache
    assert any(profile_home.cache_dir.iterdir())


def test_ingest_local_paths(profile_home):
    pieces = ingest_local_paths([LOCAL], source_hint="note")
    assert len(pieces) >= 2
    assert all(p.source == "note" for p in pieces)
    # read in place — path points at original
    assert any(str(LOCAL) in p.path for p in pieces)


def test_run_ingest_appends(profile_home):
    added, _ = run_ingest(profile_home, linkedin=LINKEDIN)
    assert added > 0
    added2, _ = run_ingest(profile_home, linkedin=LINKEDIN)
    assert added2 == 0  # duplicate ids skipped
    assert len(load_index(profile_home.index_path)) == added


def test_select_defaults_min_words_and_year(profile_home):
    run_ingest(profile_home, linkedin=LINKEDIN, local=[LOCAL], source_hint="note")
    selection, selected = run_select(
        profile_home, min_words=50, through_year=2024, include_undated=False
    )
    assert selection.min_words == 50
    assert selection.through_year == 2024
    assert all(p.word_count >= 50 for p in selected)
    assert all(p.year is not None and p.year <= 2024 for p in selected)
    # 2025 post excluded by default
    assert not any(p.year == 2025 for p in selected)
    assert "by_source" in selection.summary
    assert "by_year" in selection.summary
    summary = summarize_by_source_year(selected)
    assert summary["pieces"] == len(selected)


def test_select_include_exclude(profile_home):
    run_ingest(profile_home, local=[LOCAL], source_hint="note")
    pieces = load_index(profile_home.index_path)
    short = next(p for p in pieces if p.word_count < 50)
    long = next(p for p in pieces if p.word_count >= 50)
    selection, selected = run_select(
        profile_home,
        min_words=50,
        through_year=2024,
        include_ids=[short.id],
        exclude_ids=[long.id],
    )
    ids = {p.id for p in selected}
    assert short.id in ids
    assert long.id not in ids


def test_filter_pieces_unit():
    from personality_protect.models import Piece

    pieces = [
        Piece(id="a", source="doc", text=" ".join(["word"] * 60), date="2023-01-01", year=2023),
        Piece(id="b", source="doc", text=" ".join(["word"] * 10), date="2023-01-01", year=2023),
        Piece(id="c", source="doc", text=" ".join(["word"] * 60), date="2025-01-01", year=2025),
        Piece(id="d", source="doc", text=" ".join(["word"] * 60), date=None, year=None),
    ]
    out = filter_pieces(pieces, min_words=50, through_year=2024, include_undated=False)
    assert [p.id for p in out] == ["a"]
    out2 = filter_pieces(pieces, min_words=50, through_year=2024, include_undated=True)
    assert {p.id for p in out2} == {"a", "d"}


def test_sft_builder(profile_home):
    run_ingest(profile_home, linkedin=LINKEDIN)
    run_select(profile_home, min_words=20, through_year=2024, include_undated=True)
    path, n = build_sft_from_profile(profile_home)
    assert n > 0
    assert path.is_file()
    line = path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert "messages" in row
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_piece_to_example_has_assistant_voice():
    from personality_protect.models import Piece

    p = Piece(
        id="x",
        source="demo",
        text="I cut the corporate fog and keep the spine of the argument with care.",
        year=2023,
    )
    ex = piece_to_example(p)
    assert ex["messages"][-1]["content"] == p.text
