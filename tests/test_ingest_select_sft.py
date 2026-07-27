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
        '"I absolutely get why Contoso Labs would get into this. They have 0 knowledge '
        'about the vertical."\n'
        '"But Northwind Analytics? WTH?!"\n'
        '""\n'
        '"Northwind Analytics — you torch the cash cow and drain the moat."'
    )
    clean = normalize_corpus_text(raw)
    assert '""' not in clean
    assert not clean.startswith('"')
    assert '\n"' not in clean
    assert clean.startswith("Let's be real.")
    assert "cash cow" in clean
    # Paragraph wrappers gone; no dangling line-start/line-end CSV quotes
    for line in clean.splitlines():
        if not line.strip():
            continue
        assert not line.startswith('"'), line
        assert not line.endswith('"'), line


def test_normalize_corpus_text_strips_embedded_html_css():
    """LinkedIn article exports sometimes paste CSS/HTML into the body."""
    from personality_protect.sft import normalize_corpus_text

    raw = (
        "We Asked 5 LLMs to Grade Our Code. They Averaged 8.4/10.\n"
        "body {\n"
        "      margin: 0 auto;\n"
        "      width: 744px;\n"
        "      font-family: Source Serif Pro, serif;\n"
        "}\n"
        "<p>The models agreed on one thing: clarity beats cleverness.</p>\n"
        "<div>Keep the verdict, drop the chrome.</div>"
    )
    clean = normalize_corpus_text(raw)
    assert "body {" not in clean
    assert "font-family" not in clean
    assert "744px" not in clean
    assert "<p>" not in clean.lower()
    assert "<div>" not in clean.lower()
    assert "clarity beats cleverness" in clean
    assert "Keep the verdict" in clean
    assert "8.4/10" in clean


def test_sft_truncates_long_targets_for_masked_seq_budget():
    """512-token mask_prompt trains NaN when prompt+target overflow the window."""
    from personality_protect.models import Piece
    from personality_protect.sft import (
        MAX_SFT_DRAFT_CHARS,
        MAX_SFT_TARGET_CHARS,
        piece_to_examples,
    )

    huge = ("Travel tech changes fast. " * 400).strip()
    assert len(huge) > MAX_SFT_TARGET_CHARS * 2
    examples = piece_to_examples(
        Piece(id="long", source="note", text=huge, year=2024, word_count=2000)
    )
    assert examples
    for ex in examples:
        target = ex["messages"][-1]["content"]
        assert len(target) <= MAX_SFT_TARGET_CHARS + 5
        user = ex["messages"][1]["content"]
        draft = user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
        assert len(draft) <= MAX_SFT_DRAFT_CHARS + 5


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


def test_piece_to_examples_includes_clean_draft_voice_pair():
    """Clean/neutral drafts must also train style transfer, not only slop-strip."""
    from personality_protect.models import Piece
    from personality_protect.sft import piece_to_examples

    text = (
        "Let's be real.\n\n"
        "I absolutely get why Contoso Labs would get into this. They have 0 knowledge "
        "about the vertical.\n\n"
        "But Northwind Analytics? WTH?!\n\n"
        "Northwind Analytics — you torch the cash cow and drain the moat."
    )
    examples = piece_to_examples(
        Piece(id="z", source="demo", text=text, year=2026, word_count=80)
    )
    assert len(examples) >= 3
    kinds = {ex["meta"]["pair_kind"] for ex in examples}
    assert "slop" in kinds
    assert "clean" in kinds
    # Multi-paragraph voice pieces get leave-alone pairs (don't fidget / flatten)
    assert "leave_alone" in kinds
    # Short multi-para pieces also emit cadence oversampling pairs
    assert "slop_cadence" in kinds or "clean_cadence" in kinds

    clean = next(ex for ex in examples if ex["meta"]["pair_kind"] == "clean")
    user = clean["messages"][1]["content"]
    draft = user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
    target = clean["messages"][-1]["content"]
    assert draft != target
    # Assistant targets keep real paragraph structure from the corpus voice
    assert "\n\n" in target
    # No heavy AI-tell scaffolding — this pair teaches voice-on-neutral rewrites
    assert "leverage" not in draft.lower()
    assert "fast-paced" not in draft.lower()
    assert "testament" not in draft.lower()
    assert "moreover, furthermore" not in draft.lower()
    # Cadence still flattened so the target must restore voice
    assert "let's be real" not in draft.lower()
    assert "cash cow" not in draft.lower()
    assert "northwind" in draft.lower()

    leave = next(ex for ex in examples if ex["meta"]["pair_kind"] == "leave_alone")
    leave_user = leave["messages"][1]["content"]
    leave_draft = leave_user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
    leave_target = leave["messages"][-1]["content"]
    assert leave_draft == leave_target
    assert "\n\n" in leave_target


def test_ensure_voice_paragraphs_splits_flat_sentence_blocks():
    """Flat multi-sentence voice must become multi-para rewrite targets."""
    from personality_protect.models import Piece
    from personality_protect.sft import _ensure_voice_paragraphs, piece_to_examples

    flat = (
        "Personal branding matters more than ever. AI tools flood every channel. "
        "Companies need a clear point of view, not another template post."
    )
    para = _ensure_voice_paragraphs(flat)
    assert "\n\n" in para
    assert para.count("\n\n") >= 1
    examples = piece_to_examples(
        Piece(id="flat1", source="demo", text=flat, year=2026, word_count=30)
    )
    rewrite = [ex for ex in examples if ex["meta"]["pair_kind"] in {"slop", "clean"}]
    assert rewrite
    for ex in rewrite:
        assert "\n\n" in ex["messages"][-1]["content"]
    # Artificially paragraphized flats must NOT mint leave-alone pass-through
    assert "leave_alone" not in {ex["meta"]["pair_kind"] for ex in examples}


def test_synthetic_short_cadence_examples_are_multi_para_voice():
    from personality_protect.sft import _synthetic_short_cadence_examples

    rows = _synthetic_short_cadence_examples()
    assert len(rows) >= 6
    kinds = {r["meta"]["pair_kind"] for r in rows}
    assert "slop" in kinds and "clean" in kinds
    for r in rows:
        user = r["messages"][1]["content"]
        draft = user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
        target = r["messages"][-1]["content"]
        assert draft.strip() != target.strip()
        assert "\n\n" in target
        # Synthetic-only entities / no personal corpus paths
        blob = (draft + "\n" + target).lower()
        assert "travelport" not in blob
        assert "linkedin.com" not in blob


def test_synthetic_multipara_cadence_examples_are_multipara_both_sides():
    from personality_protect.sft import _synthetic_multipara_cadence_examples

    rows = _synthetic_multipara_cadence_examples()
    assert len(rows) >= 18
    kinds = {r["meta"]["pair_kind"] for r in rows}
    assert "slop" in kinds
    assert any(k.endswith("_multipara") for k in kinds)
    multipara = [r for r in rows if r["meta"]["pair_kind"].endswith("_multipara")]
    assert multipara
    for r in multipara[:6]:
        user = r["messages"][1]["content"]
        draft = user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
        target = r["messages"][-1]["content"]
        assert "\n\n" in draft, "multipara slop draft must keep blank lines"
        assert "\n\n" in target
        assert "?" in target or "—" in target or "Let's" in target or "Here's" in target
        blob = (draft + "\n" + target).lower()
        assert "travelport" not in blob
        assert "life vest" not in blob
        assert "linkedin.com" not in blob
        assert "contoso" in blob or "northwind" in blob


def test_synthetic_clean_flat_voice_examples_restyle_bland_prose():
    from personality_protect.sft import _synthetic_clean_flat_voice_examples

    rows = _synthetic_clean_flat_voice_examples()
    assert len(rows) >= 18
    kinds = {r["meta"]["pair_kind"] for r in rows}
    assert "clean" in kinds
    assert "clean_flat" in kinds
    flats = [r for r in rows if r["meta"]["pair_kind"] == "clean_flat"]
    assert len(flats) >= 8
    for r in flats[:6]:
        user = r["messages"][1]["content"]
        draft = user.split("### Draft\n", 1)[1].split("\n\n### Rewritten", 1)[0]
        target = r["messages"][-1]["content"]
        assert "\n\n" not in draft.strip(), "clean-flat draft must be single-block bland"
        assert "\n\n" in target
        assert draft.strip() != target.strip()
        # Must be a real restyle, not blank-line-only of the same sentences.
        import re

        a = re.sub(r"\s+", " ", draft.lower())
        b = re.sub(r"\s+", " ", target.lower())
        assert a != b
        assert "leverage" not in draft.lower()
        assert "fast-paced" not in draft.lower()
        blob = (draft + "\n" + target).lower()
        assert "travelport" not in blob
        assert "life vest" not in blob
        assert "linkedin.com" not in blob
        assert "contoso" in blob or "northwind" in blob or "personal branding" in blob


def test_clean_generic_draft_forces_rewrite_even_on_flat_prose():
    """Already-clean drafts must NOT be near-identity — that teaches LoRA pass-through."""
    from personality_protect.sft import _clean_generic_draft, normalize_corpus_text

    text = (
        "Personal branding matters more than ever as AI tools flood every channel. "
        "Companies need a clear point of view, not another template post about authenticity."
    )
    target = normalize_corpus_text(text)
    draft = _clean_generic_draft(text)
    assert draft.strip() != target.strip()
    # Still clean — no AI-tell scaffolding
    assert "leverage" not in draft.lower()
    assert "fast-paced" not in draft.lower()
    assert "synerg" not in draft.lower()
    assert "testament" not in draft.lower()
    # Lexical overlap must drop so the assistant target teaches voice injection
    tw = set(re.findall(r"[a-z0-9']+", target.lower()))
    dw = set(re.findall(r"[a-z0-9']+", draft.lower()))
    overlap = len(tw & dw) / max(1, len(tw))
    assert overlap < 0.75, f"clean draft still near-identity (overlap={overlap:.2f})"
    # Meaning retained
    assert "branding" in draft.lower() or "brand" in draft.lower()
    assert (
        "authenticity" in draft.lower()
        or "authentic" in draft.lower()
        or "genuine" in draft.lower()
    )


def test_neutral_draft_is_not_near_identity_copy():
    """Draft must teach rewrite→voice, not strip-opener-and-copy."""
    from personality_protect.models import Piece
    from personality_protect.sft import _neutral_draft, normalize_corpus_text

    text = (
        "Let's be real.\n\n"
        "I absolutely get why Contoso Labs would get into this. They have 0 knowledge "
        "about the vertical.\n\n"
        "But Northwind Analytics? WTH?!\n\n"
        "Northwind Analytics — you are about to give away decades of domain expertise. "
        "You torch the cash cow, drain the moat, and chase a silver bullet."
    )
    target = normalize_corpus_text(text)
    draft = _neutral_draft(text)
    assert draft != target
    assert "leverage" in draft.lower() or "important to note" in draft.lower()
    # Cadence markers must be flattened out of the draft side
    assert "let's be real" not in draft.lower()
    assert "wth" not in draft.lower()
    assert "cash cow" not in draft.lower()
    assert "silver bullet" not in draft.lower()
    assert "—" not in draft
    # Lexical overlap must drop below near-copy (was ~100% before fix)
    tw = set(re.findall(r"[a-z0-9']+", target.lower()))
    dw = set(re.findall(r"[a-z0-9']+", draft.lower()))
    overlap = len(tw & dw) / max(1, len(tw))
    assert overlap < 0.75, f"draft still near-identity (overlap={overlap:.2f})"
    # Entities / meaning retained
    assert "northwind" in draft.lower()
    assert "contoso" in draft.lower()

    ex = piece_to_example(
        Piece(id="y", source="demo", text=text, year=2026, word_count=80)
    )
    assert "Let's be real" in ex["messages"][-1]["content"]
    # Target is seq-budget truncated; keep early voice markers + entities.
    assert "Northwind" in ex["messages"][-1]["content"]
    assert "### My voice" not in ex["messages"][1]["content"]
    assert ex["messages"][1]["content"].count("### Draft") == 1
