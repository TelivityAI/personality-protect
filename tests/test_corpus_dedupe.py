"""Contoso-safe tests for duplicate-text cleanup of the corpus index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import init_profile
from personality_protect.corpus_dedupe import (
    DEFAULT_NEAR_RATIO,
    dedupe_pieces,
    duplicate_key,
    find_duplicate_groups,
)
from personality_protect.models import Piece, load_index, save_index

runner = CliRunner()

CONTOSO_POST = (
    "Contoso shipped a pricing test this quarter. We compared renewal signals "
    "against customer value, then killed the variant that only moved clicks. "
    "The boring version won and the rollback plan stayed one page long.\n\n"
    "Three things made it work. We wrote the failure condition before the "
    "experiment started, so nobody could relabel a flat quarter as momentum. "
    "We gave the support team the rollback switch instead of routing it through "
    "a release meeting. And we published the losing numbers in the same note as "
    "the winning ones, which is the part most teams quietly skip.\n\n"
    "The uncomfortable finding: our best-converting plan was also our worst "
    "renewing plan. Cheaper entry pricing pulled in accounts that never "
    "onboarded, and the churn showed up two quarters later where nobody was "
    "looking for it. Pricing is a retention decision wearing an acquisition "
    "costume, and the dashboard that tracks signups will never tell you that."
)


def _piece(piece_id: str, **overrides) -> Piece:
    fields = {
        "id": piece_id,
        "source": "linkedin_post",
        "text": CONTOSO_POST,
        "date": "2024-03-04",
    }
    fields.update(overrides)
    return Piece(**fields)


def test_duplicate_key_ignores_case_whitespace_and_export_wrapping():
    assert duplicate_key("  Contoso   Pricing\n\n") == duplicate_key("contoso pricing")
    assert duplicate_key('"Contoso pricing."') == duplicate_key("Contoso pricing.")
    assert duplicate_key("<p>Contoso pricing</p>") == duplicate_key("Contoso pricing")
    assert duplicate_key("   ") == ""


def test_exact_duplicate_from_second_export_path_is_dropped():
    result = dedupe_pieces(
        [
            _piece("aaaa", path="exports/first.csv"),
            _piece("bbbb", path="exports/second.csv"),
            _piece("cccc", text="Contoso platform migrations need boring rollback plans."),
        ]
    )

    assert result.dropped_ids == ["bbbb"]
    assert [piece.id for piece in result.kept] == ["aaaa", "cccc"]
    assert result.to_report()["exact_groups"] == 1
    assert result.to_report()["dropped_by_source"] == {"linkedin_post": 1}


def test_keeper_prefers_specific_source_then_date_then_metadata():
    groups = find_duplicate_groups(
        [
            _piece("aaaa", source="linkedin_comment"),
            _piece("bbbb", source="linkedin_article"),
            _piece("cccc", source="linkedin_post"),
        ]
    )
    assert [groups[0].keeper.id, *sorted(piece.id for piece in groups[0].dropped)] == [
        "bbbb",
        "aaaa",
        "cccc",
    ]
    assert groups[0].cross_source is True
    assert groups[0].sources == ["linkedin_article", "linkedin_comment", "linkedin_post"]

    dated = find_duplicate_groups([_piece("aaaa", date=None), _piece("bbbb")])
    assert dated[0].keeper.id == "bbbb"

    titled = find_duplicate_groups(
        [_piece("aaaa"), _piece("bbbb", title="Contoso pricing test")]
    )
    assert titled[0].keeper.id == "bbbb"

    richer_meta = find_duplicate_groups(
        [_piece("bbbb"), _piece("aaaa", meta={"file": "shares.csv"})]
    )
    assert richer_meta[0].keeper.id == "aaaa"


def test_identical_pieces_collapse_to_the_lowest_id():
    groups = find_duplicate_groups([_piece("cccc"), _piece("aaaa"), _piece("bbbb")])
    assert groups[0].keeper.id == "aaaa"
    assert [piece.id for piece in groups[0].dropped] == ["bbbb", "cccc"]


def test_near_identical_recapture_needs_the_ratio_and_keeps_fuller_text():
    scraped = _piece("aaaa", text=CONTOSO_POST, path="scrape/post.txt")
    exported = _piece(
        "bbbb",
        text=CONTOSO_POST + " #contoso #pricing",
        path="exports/shares.csv",
    )

    exact_only = dedupe_pieces([scraped, exported], near_ratio=None)
    assert exact_only.dropped_ids == []

    near = dedupe_pieces([scraped, exported], near_ratio=DEFAULT_NEAR_RATIO)
    assert near.dropped_ids == ["aaaa"]
    assert near.groups[0].keeper.id == "bbbb"
    assert near.groups[0].exact is False


def test_distinct_pieces_sharing_an_opening_are_not_near_duplicates():
    tail = CONTOSO_POST.rsplit("\n\n", 1)[0]
    result = dedupe_pieces(
        [
            _piece("aaaa", text=tail + "\n\nThe renewal cohort barely moved this month."),
            _piece("bbbb", text=tail + "\n\nSupport tickets dropped by a third instead."),
        ],
        near_ratio=DEFAULT_NEAR_RATIO,
    )
    assert result.dropped_ids == []


def test_holdout_id_always_becomes_the_keeper():
    result = dedupe_pieces(
        [
            _piece("aaaa", source="linkedin_article", title="Contoso pricing"),
            _piece("zzzz", source="linkedin_comment", date=None),
        ],
        holdout_ids={"zzzz"},
    )

    assert result.groups[0].keeper.id == "zzzz"
    assert result.dropped_ids == ["aaaa"]
    assert "zzzz" in {piece.id for piece in result.kept}


def test_dedupe_refuses_when_a_holdout_would_be_dropped():
    groups = find_duplicate_groups([_piece("aaaa"), _piece("zzzz")])
    assert {piece.id for piece in groups[0].dropped} == {"zzzz"}

    with pytest.raises(AssertionError, match="zzzz"):
        dedupe_pieces([_piece("aaaa"), _piece("zzzz")], holdout_ids={"aaaa", "zzzz"})


def test_empty_text_pieces_never_group_together():
    result = dedupe_pieces([_piece("aaaa", text=""), _piece("bbbb", text="   ")])
    assert result.dropped_ids == []


def test_cli_dedupe_index_reports_then_applies_with_backup(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    paths.root.joinpath("dogfood_holdout_ids.json").write_text(
        json.dumps({"holdout_ids": ["zzzz"]}), encoding="utf-8"
    )
    save_index(
        paths.index_path,
        [
            _piece("aaaa", source="linkedin_article"),
            _piece("zzzz", source="linkedin_comment"),
            _piece("cccc", text="Contoso platform migrations need boring rollback plans."),
        ],
    )

    base_args = ["--logo", "off", "dedupe-index", "--profile", "contoso", "--home", str(tmp_path)]
    report = runner.invoke(app, [*base_args, "--json"])
    assert report.exit_code == 0, report.output
    payload = json.loads(report.output)
    assert payload["applied"] is False
    assert payload["dropped"] == 1
    assert payload["group_reports"][0]["keeper"] == "zzzz"
    assert len(load_index(paths.index_path)) == 3

    applied = runner.invoke(app, [*base_args, "--apply", "--json"])
    assert applied.exit_code == 0, applied.output
    payload = json.loads(applied.output)
    assert payload["applied"] is True
    assert payload["written"] == 2
    assert payload["after"]["pieces"] == 2
    assert {piece.id for piece in load_index(paths.index_path)} == {"zzzz", "cccc"}
    assert Path(payload["backup"]).is_file()
    assert len(load_index(Path(payload["backup"]))) == 3

    again = runner.invoke(app, [*base_args, "--apply", "--json"])
    assert json.loads(again.output)["dropped"] == 0


def test_cli_dedupe_index_fails_without_an_index(tmp_path: Path):
    init_profile("contoso", home=tmp_path)
    result = runner.invoke(
        app,
        ["--logo", "off", "dedupe-index", "--profile", "contoso", "--home", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "No corpus index" in result.output
