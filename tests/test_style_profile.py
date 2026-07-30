"""Contoso-safe tests for corpus style_profile build."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import get_paths, init_profile
from personality_protect.models import Piece, save_index
from personality_protect.select import Selection
from personality_protect.style_profile import (
    BANNED_AI_FILLER,
    LINKEDIN_POST_WORD_CEILING,
    build_style_profile,
    corpus_style_stats,
    draft_word_target,
    load_style_profile,
    run_build_style_profile,
    style_directives,
    style_profile_path,
    text_style_axes,
)

runner = CliRunner()

# Punchy Contoso voice: short lines, contractions, you > I.
CONTOSO_VOICED = (
    "Contoso Ledger.\n"
    "Category 12 isn't a feature.\n"
    "You ship the reconciliation or you own the outage.\n"
    "Partners already know.\n"
    "Don't pretend the roadmap is the work.\n"
)

# Longer Contoso article body: lower short-line ratio, still concrete.
CONTOSO_ARTICLE = (
    "Contoso Labs published another guidance memo on workplace communication "
    "quality. The document argues that organizations must evaluate not only "
    "the substance of internal writing, but also the texture of the prose "
    "itself. Leaders at Contoso suggest that compressed thinking is frequently "
    "misread as low-effort output, and that Northwind Analytics should adopt "
    "a similar review framework before shipping customer-facing narratives.\n\n"
    "Here's the problem: the people most likely to fail a Contoso style audit "
    "are not the people producing low-effort work. They are often the people "
    "who cut through ambiguity fastest.\n"
)


def test_banned_ai_filler_covers_locked_prompt_words():
    banned = {p.lower() for p in BANNED_AI_FILLER}
    for word in ("leverage", "delve", "moreover", "tapestry"):
        assert word in banned


def test_text_style_axes_contoso_voiced():
    axes = text_style_axes(CONTOSO_VOICED)
    assert axes["words"] > 10
    assert axes["short_line_ratio"] >= 0.5
    assert axes["contraction_rate"] > 0.0
    assert axes["you_gt_i"] is True
    assert axes["median_sentence_words"] > 0


def test_corpus_style_stats_aggregates_contoso_pieces():
    stats = corpus_style_stats([CONTOSO_VOICED, CONTOSO_ARTICLE])
    assert stats["pieces"] == 2
    assert stats["words"] > 40
    assert stats["median_sentence_words"] > 0
    assert 0.0 <= stats["short_line_ratio"] <= 1.0
    assert stats["contraction_rate"] >= 0.0
    assert "you_gt_i" in stats


def test_build_style_profile_includes_banned_list():
    pieces = [
        Piece(id="c1", source="linkedin_post", text=CONTOSO_VOICED, year=2024),
        Piece(id="c2", source="linkedin_article", text=CONTOSO_ARTICLE, year=2024),
    ]
    profile = build_style_profile(pieces)
    assert profile["version"] == 1
    assert profile["piece_ids"] == ["c1", "c2"]
    assert profile["stats"]["pieces"] == 2
    assert profile["banned_ai_filler"] == list(BANNED_AI_FILLER)
    assert "leverage" in profile["banned_ai_filler"]
    assert "moreover" in profile["banned_ai_filler"]


def test_build_style_profile_records_median_post_words():
    pieces = [
        Piece(id="c1", source="linkedin_post", text=CONTOSO_VOICED, year=2024),
        Piece(id="c2", source="linkedin_article", text=CONTOSO_ARTICLE, year=2024),
    ]
    stats = build_style_profile(pieces)["stats"]
    assert stats["median_post_words"] > 0
    assert stats["post_words_p75"] > 0
    assert stats["post_words_p90"] > 0


def test_post_length_ignores_short_comments_for_targets():
    """Length targets come from post-shaped pieces, not comment stubs."""
    short = "Ok." * 5
    long_post = (
        "Contoso Ledger. " * 40
        + "You ship the reconciliation or you own the outage. "
        * 20
    )
    pieces = [
        Piece(id="comment", source="linkedin_comment", text=short, year=2024),
        Piece(id="post", source="linkedin_post", text=long_post, year=2024),
    ]
    stats = build_style_profile(pieces)["stats"]
    assert stats["post_words_p90"] >= 80
    ceiling = draft_word_target(build_style_profile(pieces))
    assert ceiling >= 300
    assert ceiling <= LINKEDIN_POST_WORD_CEILING


def test_draft_word_target_clamps_to_linkedin_ceiling():
    huge = "word " * 2000
    profile = build_style_profile(
        [Piece(id="p", source="linkedin_post", text=huge, year=2024)]
    )
    assert draft_word_target(profile) == LINKEDIN_POST_WORD_CEILING


def test_style_directives_carry_cadence_without_copyable_prose():
    """The voice card must describe the voice, never quote it."""
    pieces = [
        Piece(id="c1", source="linkedin_post", text=CONTOSO_VOICED, year=2024),
        Piece(id="c2", source="linkedin_article", text=CONTOSO_ARTICLE, year=2024),
    ]
    profile = build_style_profile(pieces)
    directives = style_directives(profile)
    joined = " ".join(directives)

    assert any("words total" in line for line in directives)
    assert any("8 words or fewer" in line for line in directives)
    assert "leverage" in joined  # banned list is stated, not used
    for sentence in ("Category 12 isn't a feature", "Contoso Labs published"):
        assert sentence not in joined


def test_style_directives_reflect_you_versus_i_lean():
    i_heavy = Piece(
        id="i1",
        source="linkedin_post",
        text="I shipped the ledger. I owned the outage. I wrote the memo myself.",
    )
    directives = style_directives(build_style_profile([i_heavy]))
    assert any("first person" in line for line in directives)

    you_heavy = Piece(id="y1", source="linkedin_post", text=CONTOSO_VOICED)
    directives = style_directives(build_style_profile([you_heavy]))
    assert any("'you'" in line for line in directives)


def test_style_directives_empty_profile_is_safe():
    assert style_directives({}) == []


def test_style_directives_ask_for_varied_sentence_length():
    """A lone median told the model to make every sentence that length."""
    pieces = [
        Piece(id="c1", source="linkedin_post", text=CONTOSO_VOICED, year=2024),
        Piece(id="c2", source="linkedin_article", text=CONTOSO_ARTICLE, year=2024),
    ]
    profile = build_style_profile(pieces)
    stats = profile["stats"]
    assert stats["sentence_words_p75"] > stats["sentence_words_p25"]
    assert 0.0 < stats["multi_sentence_paragraph_ratio"] <= 1.0

    directives = style_directives(profile)
    assert any("Sentence length varies" in line for line in directives)
    assert any("two or more" in line for line in directives)


def test_run_build_style_profile_writes_json(tmp_path: Path):
    paths, _, _ = init_profile("style", home=tmp_path)
    pieces = [
        Piece(id="c1", source="linkedin_post", text=CONTOSO_VOICED, year=2023),
        Piece(id="c2", source="linkedin_post", text=CONTOSO_ARTICLE, year=2024),
    ]
    save_index(paths.index_path, pieces)
    selection = Selection(
        piece_ids=["c1", "c2"],
        min_words=10,
        through_year=2024,
        include_undated=True,
        summary={"pieces": 2},
    )
    paths.selection_path.write_text(
        json.dumps(selection.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    profile, out = run_build_style_profile(paths)
    assert out == style_profile_path(paths)
    assert out.is_file()
    loaded = load_style_profile(paths)
    assert loaded["stats"]["pieces"] == 2
    assert loaded["banned_ai_filler"][0] == BANNED_AI_FILLER[0]
    assert set(loaded["piece_ids"]) == {"c1", "c2"}
    assert profile["stats"]["you_count"] >= 1


def test_cli_build_style_profile_json(tmp_path: Path):
    home = str(tmp_path)
    r = runner.invoke(
        app,
        ["--logo", "off", "init", "--home", home, "--profile", "t", "--json"],
    )
    assert r.exit_code == 0, r.output

    paths = get_paths("t", home=tmp_path)
    pieces = [
        Piece(id="contoso-a", source="linkedin_post", text=CONTOSO_VOICED, year=2024),
        Piece(
            id="contoso-b",
            source="linkedin_article",
            text=CONTOSO_ARTICLE,
            year=2024,
        ),
    ]
    save_index(paths.index_path, pieces)
    paths.selection_path.write_text(
        json.dumps(
            Selection(
                piece_ids=["contoso-a", "contoso-b"],
                min_words=10,
                through_year=2024,
                summary={"pieces": 2},
            ).to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    r = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "build-style-profile",
            "--home",
            home,
            "--profile",
            "t",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["pieces"] == 2
    assert "leverage" in data["banned_ai_filler"]
    assert "delve" in data["banned_ai_filler"]
    assert data["stats"]["median_sentence_words"] > 0
    assert Path(data["path"]).is_file()


def test_cli_build_style_profile_requires_selection(tmp_path: Path):
    home = str(tmp_path)
    r = runner.invoke(
        app,
        ["--logo", "off", "init", "--home", home, "--profile", "t", "--json"],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "build-style-profile",
            "--home",
            home,
            "--profile",
            "t",
            "--json",
        ],
    )
    assert r.exit_code == 1
    assert "select" in r.output.lower() or "selection" in r.output.lower()
