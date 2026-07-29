"""Contoso-safe tests for the private pair-factory orchestration."""

from __future__ import annotations

import base64
import csv
import importlib.util
import io
import re
import sys
import tarfile
import tempfile
from pathlib import Path


def _load_installer_pair_factory():
    """Load the Contoso-safe private helper from the committed installer."""
    installer = (
        Path(__file__).parents[1]
        / "scripts"
        / "mac-install-private-studio.sh"
    ).read_text(encoding="utf-8")
    match = re.search(r"b64 = r'''(.*?)'''", installer, flags=re.DOTALL)
    assert match is not None
    with tarfile.open(
        fileobj=io.BytesIO(base64.b64decode(match.group(1))),
        mode="r:gz",
    ) as archive:
        member = archive.getmember("private/pair_factory.py")
        extracted = archive.extractfile(member)
        assert extracted is not None
        source = extracted.read()
    root = Path(tempfile.mkdtemp(prefix="contoso-pair-factory-"))
    module_path = root / "private" / "pair_factory.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(
        "installer_pair_factory", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pair_factory = _load_installer_pair_factory()


def _write_shares(path: Path, word_counts: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "Date",
                "ShareLink",
                "ShareCommentary",
                "SharedUrl",
                "MediaUrl",
                "Visibility",
            ],
        )
        writer.writeheader()
        for index, words in enumerate(word_counts, start=1):
            writer.writerow(
                {
                    "Date": f"2026-01-{index:02d}",
                    "ShareLink": f"https://contoso.example/posts/{index}",
                    "ShareCommentary": " ".join(
                        f"word{word}" for word in range(words)
                    ),
                    "Visibility": "MEMBER_NETWORK",
                }
            )


def _article_text(paragraphs: int = 8) -> str:
    return "\n\n".join(
        (
            f"Contoso section {index} explains the reconciliation workflow "
            + "with deterministic synthetic evidence " * 12
        )
        for index in range(paragraphs)
    )


def test_discover_sources_reads_shares_and_articles(tmp_path: Path):
    shares = tmp_path / "Shares_64568138.csv"
    articles = tmp_path / "Articles"
    articles.mkdir()
    _write_shares(shares, [39, 40, 65])
    (articles / "contoso-one.html").write_text(
        f"<html><body><p>{_article_text(2)}</p></body></html>",
        encoding="utf-8",
    )

    pieces = pair_factory.discover_source_pieces([shares, articles])

    posts = [piece for piece in pieces if piece.source_type == "post"]
    article_pieces = [
        piece for piece in pieces if piece.source_type == "article"
    ]
    assert len(posts) == 2
    assert len(article_pieces) == 1
    assert all(piece.source_id for piece in pieces)
    assert all(piece.channel in {"post", "article"} for piece in pieces)


def test_plan_sections_balances_before_flattening(tmp_path: Path):
    shares = tmp_path / "Shares_64568138.csv"
    articles = tmp_path / "Articles"
    articles.mkdir()
    _write_shares(shares, [40, 41, 42, 43])
    for index in range(2):
        (articles / f"contoso-{index}.md").write_text(
            _article_text(),
            encoding="utf-8",
        )

    pieces = pair_factory.discover_source_pieces([shares, articles])
    plan = pair_factory.plan_source_sections(pieces, max_seq_length=400)

    assert plan["sources"] == {"post": 4, "article": 2}
    assert plan["before"]["posts_whole"] == 4
    assert plan["before"]["article_openers"] == 2
    assert plan["before"]["article_closers"] == 2
    assert plan["before"]["article_middles"] > 0
    assert plan["after"]["posts_whole"] == 4
    assert plan["after"]["article_openers"] == 2
    assert plan["after"]["article_closers"] == 2
    assert plan["after"]["article_middles"] == 0
    assert plan["flatten_calls_saved"] == plan["before"]["article_middles"]
    assert len(plan["selected"]) == sum(plan["after"].values())
    assert all(section.section_role for section in plan["selected"])


def test_pair_rows_carry_channel_metadata():
    row = pair_factory.emit_pair_row(
        flat="Flat Contoso text.",
        original="Voiced Contoso text.",
        pair_id="contoso.part-01",
        source_type="article",
        source_id="contoso",
        section_role="opener",
    )

    assert row["source_type"] == "article"
    assert row["channel"] == "article"
    assert row["source_id"] == "contoso"
    assert row["section_role"] == "opener"


def test_train_passes_force_retrain_and_resume(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pair_factory,
        "run_pp",
        lambda args: calls.append(args) or 0,
    )
    parser = pair_factory.build_parser()

    force = parser.parse_args(
        ["train", "--pairs", "pairs.jsonl", "--force-retrain"]
    )
    resume = parser.parse_args(["train", "--pairs", "pairs.jsonl", "--resume"])
    pair_factory.cmd_train(force)
    pair_factory.cmd_train(resume)

    assert "--force-retrain" in calls[0]
    assert "--resume" in calls[1]


def test_train_rejects_force_retrain_with_resume():
    parser = pair_factory.build_parser()

    try:
        parser.parse_args(
            [
                "train",
                "--pairs",
                "pairs.jsonl",
                "--force-retrain",
                "--resume",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected mutually exclusive train flags")
