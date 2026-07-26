"""Tests for ingest, select, and SFT builder."""

from __future__ import annotations

import json
import re
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


def test_normalize_corpus_text_unwraps_linkedin_csv_quotes():
    from personality_protect.sft import normalize_corpus_text

    raw = (
        'Let\'s be real."\n'
        '""\n'
        '"I absolutely get why Anthropic would get into this. They have 0 knowledge '
        'about the vertical."\n'
        '"But Travelport? WTH?!"\n'
        '""\n'
        '"Travelport — you tied one end of your life vest to a speed boat."'
    )
    clean = normalize_corpus_text(raw)
    assert '""' not in clean
    assert not clean.startswith('"')
    assert '\n"' not in clean
    assert clean.startswith("Let's be real.")
    assert "life vest" in clean
    # Paragraph wrappers gone; no dangling line-start/line-end CSV quotes
    for line in clean.splitlines():
        if not line.strip():
            continue
        assert not line.startswith('"'), line
        assert not line.endswith('"'), line


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
    user = ex["messages"][1]["content"]
    draft = user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
    assert draft != p.text
    assert "### My voice" not in user
    assert "leverage" in draft.lower() or "important to note" in draft.lower() or "moreover" in draft.lower()


def test_neutral_draft_is_not_near_identity_copy():
    """Draft must teach rewrite→voice, not strip-opener-and-copy."""
    from personality_protect.models import Piece
    from personality_protect.sft import _neutral_draft, normalize_corpus_text

    text = (
        "Let's be real.\n\n"
        "I absolutely get why Anthropic would get into this. They have 0 knowledge "
        "about the vertical.\n\n"
        "But Travelport? WTH?!\n\n"
        "Travelport — you are about to give away decades of domain expertise. "
        "You tied one end of your life vest to a speed boat and the other to a submarine."
    )
    target = normalize_corpus_text(text)
    draft = _neutral_draft(text)
    assert draft != target
    assert "leverage" in draft.lower() or "important to note" in draft.lower()
    # Cadence markers must be flattened out of the draft side
    assert "let's be real" not in draft.lower()
    assert "wth" not in draft.lower()
    assert "life vest" not in draft.lower()
    assert "—" not in draft
    # Lexical overlap must drop below near-copy (was ~100% before fix)
    tw = set(re.findall(r"[a-z0-9']+", target.lower()))
    dw = set(re.findall(r"[a-z0-9']+", draft.lower()))
    overlap = len(tw & dw) / max(1, len(tw))
    assert overlap < 0.75, f"draft still near-identity (overlap={overlap:.2f})"
    # Entities / meaning retained
    assert "travelport" in draft.lower()
    assert "anthropic" in draft.lower()

    ex = piece_to_example(
        Piece(id="y", source="linkedin_post", text=text, year=2026, word_count=80)
    )
    assert "Let's be real" in ex["messages"][-1]["content"]
    assert "life vest" in ex["messages"][-1]["content"]
    assert "### My voice" not in ex["messages"][1]["content"]
    assert ex["messages"][1]["content"].count("### Draft") == 1
