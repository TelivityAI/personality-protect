"""Contoso-safe article carve: deterministic, pinned, never below the floor."""

from __future__ import annotations

import json
from pathlib import Path

from contoso_articles import contoso_article, contoso_articles, contoso_post
from typer.testing import CliRunner

from personality_protect.article_holdout import (
    MAX_ARTICLE_HOLDOUT_N,
    MIN_ARTICLE_HOLDOUT_N,
    load_pinned_article_holdout_ids,
    resolve_article_holdout_n,
    save_article_holdout_ids,
    select_article_holdouts,
)
from personality_protect.cli import app
from personality_protect.config import init_profile
from personality_protect.models import save_index
from personality_protect.write_article import MIN_ARTICLE_CORPUS

runner = CliRunner()


def test_carve_is_deterministic_and_article_only():
    pieces = [*contoso_articles(10), contoso_post()]
    first = select_article_holdouts(pieces)
    second = select_article_holdouts(list(reversed(pieces)))
    assert first["holdout_ids"] == second["holdout_ids"]
    assert first["n_articles"] == 10
    assert contoso_post().id not in first["holdout_ids"]


def test_carve_never_drops_retrieval_below_the_floor():
    receipt = select_article_holdouts(contoso_articles(7))
    assert receipt["articles_left_indexed"] >= MIN_ARTICLE_CORPUS
    assert receipt["n_holdouts"] <= 7 - MIN_ARTICLE_CORPUS


def test_carve_is_empty_when_the_corpus_cannot_spare_an_article():
    receipt = select_article_holdouts(contoso_articles(MIN_ARTICLE_CORPUS))
    assert receipt["holdout_ids"] == []
    assert receipt["articles_left_indexed"] == MIN_ARTICLE_CORPUS


def test_carve_size_stays_inside_the_band():
    receipt = select_article_holdouts(contoso_articles(14))
    assert MIN_ARTICLE_HOLDOUT_N <= receipt["n_holdouts"] <= MAX_ARTICLE_HOLDOUT_N


def test_resolve_n_respects_the_retrieval_floor():
    assert resolve_article_holdout_n(9, total_articles=14) <= 14 - MIN_ARTICLE_CORPUS
    assert resolve_article_holdout_n(9, total_articles=6, keep_indexed=5) == 1
    assert resolve_article_holdout_n(0, total_articles=14) == 0


def test_pinned_ids_stay_carved():
    pieces = contoso_articles(12)
    pinned = [pieces[-1].id]
    receipt = select_article_holdouts(pieces, pinned_ids=pinned)
    assert pinned[0] in receipt["holdout_ids"]
    assert receipt["pinned_ids"] == pinned


def test_unbriefable_articles_are_not_carved():
    stub = contoso_article(0)
    stub.text = "Contoso ships. You own it."
    receipt = select_article_holdouts([stub, *contoso_articles(9)[1:], contoso_article(11)])
    assert stub.id not in receipt["holdout_ids"]


def test_save_and_load_round_trip_ids_only(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    receipt = select_article_holdouts(contoso_articles(12))
    written = save_article_holdout_ids(paths, receipt)
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert payload["holdout_ids"] == receipt["holdout_ids"]
    assert "text" not in json.dumps(payload)
    assert load_pinned_article_holdout_ids(paths) == receipt["holdout_ids"]


def test_cli_reports_before_it_applies(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    save_index(paths.index_path, [*contoso_articles(12), contoso_post()])

    report = runner.invoke(
        app,
        ["select-article-holdouts", "--profile", "contoso", "--home", str(tmp_path), "--json"],
    )
    assert report.exit_code == 0, report.output
    assert not (paths.root / "article_holdout_ids.json").is_file()

    applied = runner.invoke(
        app,
        [
            "select-article-holdouts",
            "--apply",
            "--profile",
            "contoso",
            "--home",
            str(tmp_path),
            "--json",
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert load_pinned_article_holdout_ids(paths) == json.loads(report.stdout)["holdout_ids"]


def test_index_voice_from_carve_excludes_article_holdouts(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    save_index(paths.index_path, [*contoso_articles(12), contoso_post()])
    receipt = select_article_holdouts(contoso_articles(12))
    save_article_holdout_ids(paths, receipt)

    result = runner.invoke(
        app,
        [
            "index-voice",
            "--from-carve",
            "--profile",
            "contoso",
            "--home",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["skipped_holdout"] == receipt["n_holdouts"]
