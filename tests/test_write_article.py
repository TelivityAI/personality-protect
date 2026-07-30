"""Contoso-safe article channel: outline → sections → stitch."""

from __future__ import annotations

from pathlib import Path

import pytest

from personality_protect.config import init_profile
from personality_protect.models import Piece, save_index
from personality_protect.style_profile import build_style_profile, save_style_profile
from personality_protect.voice_index import build_voice_index
from personality_protect.write import run_write
from personality_protect.write_article import (
    MIN_ARTICLE_CORPUS,
    assert_article_corpus,
    outline_from_brief,
    run_write_article,
)


def _article(n: int, words: int = 200) -> Piece:
    body = ("Contoso Ledger section. You own the outage. " * (words // 8)).strip()
    return Piece(
        id=f"contoso-article-{n}",
        source="linkedin_article",
        text=body,
        year=2024,
    )


def _seed_articles(tmp_path: Path, n: int = MIN_ARTICLE_CORPUS) -> Path:
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = [_article(i) for i in range(n)]
    pieces.append(
        Piece(
            id="contoso-post",
            source="linkedin_post",
            text="Contoso keeps the queue boring. You name one owner.",
            year=2024,
        )
    )
    save_index(paths.index_path, pieces)
    build_voice_index(paths)
    save_style_profile(paths, build_style_profile(pieces))
    return tmp_path


def test_outline_from_brief_uses_bullets():
    sections = outline_from_brief(
        "Contoso pricing",
        "- Name one owner\n- Cut exceptions\n- Keep Ledger boring",
    )
    assert sections == ["Name one owner", "Cut exceptions", "Keep Ledger boring"]


def test_outline_from_brief_requires_two_sections():
    with pytest.raises(ValueError, match="two section"):
        outline_from_brief("", "- Only one")


def test_assert_article_corpus_floor(tmp_path: Path):
    _seed_articles(tmp_path, n=2)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    with pytest.raises(FileNotFoundError, match="at least"):
        assert_article_corpus(paths, minimum=MIN_ARTICLE_CORPUS)


def test_run_write_article_stitches_sections(tmp_path: Path):
    _seed_articles(tmp_path, n=MIN_ARTICLE_CORPUS)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    calls: list = []

    def fake_generate(messages, **_kwargs: object) -> str:
        calls.append(messages)
        user = messages[1]["content"]
        # Each section prompt names its focus.
        assert "Write only the section about:" in user
        return (
            "Contoso Ledger holds the line.\n\n"
            "You name one owner before the packaging change."
        )

    result = run_write_article(
        "Contoso packaging",
        "- Name one owner\n- Cut exceptions\n- Keep Ledger boring",
        paths,
        k=2,
        generate_fn=fake_generate,
    )
    assert result["channel"] == "article"
    assert result["section_count"] == 3
    assert result["adapter"] == "none"
    assert len(calls) == 3
    assert result["text"].count("Contoso Ledger holds the line.") == 3
    assert result["article_count"] >= MIN_ARTICLE_CORPUS


def test_run_write_channel_article_delegates(tmp_path: Path):
    _seed_articles(tmp_path, n=MIN_ARTICLE_CORPUS)
    paths, _, _ = init_profile("contoso", home=tmp_path)

    def fake_generate(messages, **_kwargs: object) -> str:
        return "Contoso section body with enough words to survive trim."

    result = run_write(
        "Contoso packaging",
        "- Name one owner\n- Cut exceptions",
        paths,
        channel="article",
        k=1,
        generate_fn=fake_generate,
    )
    assert result["channel"] == "article"
    assert result["section_count"] == 2
