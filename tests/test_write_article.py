"""Contoso-safe article channel: outline → sections → stitch."""

from __future__ import annotations

from pathlib import Path

import pytest
from contoso_articles import contoso_articles, contoso_post

from personality_protect.config import init_profile
from personality_protect.models import save_index
from personality_protect.style_profile import (
    article_section_words,
    article_word_aim,
    build_style_profile,
    draft_word_target,
    save_style_profile,
)
from personality_protect.voice_index import build_voice_index
from personality_protect.write import run_write
from personality_protect.write_article import (
    MIN_ARTICLE_CORPUS,
    SECTION_TRIM_HEADROOM,
    article_draft_ceiling,
    assert_article_corpus,
    count_indexed_article_pieces,
    outline_from_brief,
    run_write_article,
)

BRIEF_POINTS = "- Name one owner\n- Cut exceptions\n- Keep Ledger boring"


def _seed_articles(tmp_path: Path, n: int = MIN_ARTICLE_CORPUS) -> Path:
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = [*contoso_articles(n), contoso_post()]
    save_index(paths.index_path, pieces)
    build_voice_index(paths)
    save_style_profile(paths, build_style_profile(pieces))
    return tmp_path


def test_outline_from_brief_uses_bullets():
    sections = outline_from_brief("Contoso pricing", BRIEF_POINTS)
    assert sections == ["Name one owner", "Cut exceptions", "Keep Ledger boring"]


def test_outline_from_brief_requires_two_sections():
    with pytest.raises(ValueError, match="two section"):
        outline_from_brief("", "- Only one")


def test_assert_article_corpus_floor(tmp_path: Path):
    _seed_articles(tmp_path, n=2)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    with pytest.raises(FileNotFoundError, match="in the corpus"):
        assert_article_corpus(paths, minimum=MIN_ARTICLE_CORPUS)


def test_assert_article_corpus_checks_retrieval_not_just_the_corpus(tmp_path: Path):
    """A corpus full of articles none of which are indexed is not a channel."""
    _seed_articles(tmp_path, n=6)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    carved = {piece.id for piece in contoso_articles(6)}
    build_voice_index(paths, holdout_ids=carved)
    assert count_indexed_article_pieces(paths) == 0
    with pytest.raises(FileNotFoundError, match="in the voice index"):
        assert_article_corpus(paths, minimum=MIN_ARTICLE_CORPUS)


def test_run_write_article_stitches_sections(tmp_path: Path):
    _seed_articles(tmp_path, n=MIN_ARTICLE_CORPUS)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    calls: list = []

    bodies = (
        "Contoso Ledger names one owner before anybody writes a new tier.",
        "Exceptions are where a published price list quietly stops describing anyone.",
        "Keep the renewal test boring so its signal stays readable next quarter.",
    )

    def fake_generate(messages, **_kwargs: object) -> str:
        calls.append(messages)
        assert "Write only the section about:" in messages[1]["content"]
        return bodies[len(calls) - 1]

    result = run_write_article(
        "Contoso packaging", BRIEF_POINTS, paths, k=2, generate_fn=fake_generate
    )
    assert result["channel"] == "article"
    assert result["section_count"] == 3
    assert result["adapter"] == "none"
    assert len(calls) == 3
    assert result["text"] == "\n\n".join(bodies)
    assert result["article_count"] >= MIN_ARTICLE_CORPUS


def test_stitch_drops_sections_that_restate_each_other(tmp_path: Path):
    """Independent section calls can return the same paragraph three times."""
    _seed_articles(tmp_path, n=MIN_ARTICLE_CORPUS)
    paths, _, _ = init_profile("contoso", home=tmp_path)

    def fake_generate(_messages, **_kwargs: object) -> str:
        return "Contoso Ledger holds the line before the packaging change lands."

    result = run_write_article(
        "Contoso packaging", BRIEF_POINTS, paths, k=1, generate_fn=fake_generate
    )
    assert result["section_count"] == 3
    assert result["text"].count("Contoso Ledger holds the line") == 1


def test_section_budget_comes_from_article_length_not_the_post_ceiling(tmp_path: Path):
    _seed_articles(tmp_path, n=6)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    sinks: list[str] = []
    seen: list[str] = []

    def fake_generate(messages, **_kwargs: object) -> str:
        seen.append(messages[1]["content"])
        return "Contoso Ledger names one owner and keeps the packaging record short."

    result = run_write_article(
        "Contoso packaging",
        BRIEF_POINTS,
        paths,
        k=1,
        generate_fn=fake_generate,
        prompt_sink=sinks,
    )
    style = build_style_profile([*contoso_articles(6), contoso_post()])
    expected = article_section_words(style, sections=3)
    assert result["section_words"] == expected
    assert result["word_aim"] == article_word_aim(style)
    # The post ceiling is a LinkedIn character limit and must not be the target.
    assert f"Write about {expected} words in this section" in seen[0]
    assert f"Never exceed {draft_word_target(style)} words" not in seen[0]


def test_section_prompt_states_its_place_in_the_article(tmp_path: Path):
    _seed_articles(tmp_path, n=6)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    seen: list[str] = []

    def fake_generate(messages, **_kwargs: object) -> str:
        seen.append(messages[1]["content"])
        return "Contoso Ledger names an owner and writes the decision down."

    run_write_article(
        "Contoso packaging", BRIEF_POINTS, paths, k=1, generate_fn=fake_generate
    )
    assert "section 1 of 3" in seen[0]
    assert "section 3 of 3" in seen[2]


def test_section_trim_keeps_a_long_section_from_running_away(tmp_path: Path):
    _seed_articles(tmp_path, n=6)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    runaway = "\n\n".join(
        f"Contoso Ledger paragraph {n} names one owner for the packaging change."
        for n in range(200)
    )

    result = run_write_article(
        "Contoso packaging",
        BRIEF_POINTS,
        paths,
        k=1,
        generate_fn=lambda _m, **_k: runaway,
    )
    style = build_style_profile([*contoso_articles(6), contoso_post()])
    per_section = int(round(article_section_words(style, sections=3) * SECTION_TRIM_HEADROOM))
    assert result["section_trim_words"] == per_section
    assert result["draft_words"] <= article_draft_ceiling(style, sections=3)


def test_article_retrieval_never_mixes_in_posts(tmp_path: Path):
    _seed_articles(tmp_path, n=6)
    paths, _, _ = init_profile("contoso", home=tmp_path)

    result = run_write_article(
        "Contoso packaging",
        BRIEF_POINTS,
        paths,
        k=5,
        generate_fn=lambda _m, **_k: "Contoso Ledger keeps the record short.",
    )
    assert contoso_post().id not in result["exemplar_ids"]
    assert all(piece_id.startswith("contoso-article") for piece_id in result["exemplar_ids"])


def test_run_write_channel_article_delegates(tmp_path: Path):
    _seed_articles(tmp_path, n=MIN_ARTICLE_CORPUS)
    paths, _, _ = init_profile("contoso", home=tmp_path)

    result = run_write(
        "Contoso packaging",
        "- Name one owner\n- Cut exceptions",
        paths,
        channel="article",
        k=1,
        generate_fn=lambda _m, **_k: "Contoso section body with enough words to survive trim.",
    )
    assert result["channel"] == "article"
    assert result["section_count"] == 2
