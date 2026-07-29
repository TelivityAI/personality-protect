"""Contoso-safe Phase 1 tests: gated pairs → translator SFT; train --pairs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app

runner = CliRunner()

# Flat Contoso press-release tone (sterile input) — sized for SFT seq budget.
FLAT_CONTOSO = (
    "The category twelve reconciliation workflow will ship to enterprise "
    "customers after the readiness assessment. Partner integrations remain "
    "on the published roadmap for operations teams."
)

# Voiced Contoso output (author target).
VOICED_CONTOSO = (
    "Contoso Ledger.\n\n"
    "Category 12 is not a feature.\n\n"
    "You ship the reconciliation or you own the outage."
)


def test_pair_to_translator_example_uses_translator_prompt():
    from personality_protect.sft import pair_to_translator_example

    row = pair_to_translator_example(FLAT_CONTOSO, VOICED_CONTOSO)
    assert row is not None
    messages = row["messages"]
    assert len(messages) == 3
    system = messages[0]["content"].lower()
    assert "author" in system or "voice" in system
    assert "do not leave unchanged" in system or "don't leave unchanged" in system
    assert "leave alone" not in system
    user = messages[1]["content"]
    assert "### Draft" in user
    assert FLAT_CONTOSO.strip() in user
    assert "### Rewritten" in user
    assert messages[2]["content"].strip() == VOICED_CONTOSO.strip()
    assert row["meta"]["pair_kind"] == "translator"


def test_build_sft_from_pairs_jsonl_accepts_key_aliases(tmp_path: Path):
    from personality_protect.sft import build_sft_from_pairs

    pairs = tmp_path / "pairs.kept.jsonl"
    pairs.write_text(
        "\n".join(
            [
                json.dumps({"input": FLAT_CONTOSO, "output": VOICED_CONTOSO}),
                json.dumps({"not_you": FLAT_CONTOSO, "author": VOICED_CONTOSO}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sft.jsonl"
    dest, n = build_sft_from_pairs(pairs, out)
    assert dest == out
    assert n == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["meta"]["pair_kind"] == "translator"
        assert row["messages"][0]["content"] == rows[0]["messages"][0]["content"]
        assert "do not leave unchanged" in row["messages"][0]["content"].lower()
        assert row["messages"][-1]["content"].strip() == VOICED_CONTOSO.strip()
        kinds_forbidden = {"leave_alone", "identity", "copy_through", "copy-through"}
        assert row["meta"]["pair_kind"] not in kinds_forbidden


def test_train_pairs_mode_writes_translator_sft_only(tmp_path: Path):
    """--pairs voice mode: translator rows only; no leave_alone/identity minting."""
    from personality_protect.config import init_profile
    from personality_protect.train import run_train

    pairs = tmp_path / "pairs.kept.jsonl"
    pairs.write_text(
        json.dumps({"input": FLAT_CONTOSO, "output": VOICED_CONTOSO}) + "\n",
        encoding="utf-8",
    )
    paths, _, _ = init_profile("voice", home=tmp_path)
    result = run_train(
        paths,
        backend="mock",
        mock=True,
        smoke=True,
        sft_only=True,
        pairs=pairs,
    )
    assert result.status == "sft_ready"
    assert result.examples == 1
    rows = [
        json.loads(line)
        for line in paths.sft_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["meta"]["pair_kind"] == "translator"
    kinds = {r["meta"]["pair_kind"] for r in rows}
    assert "leave_alone" not in kinds
    assert "identity" not in kinds
    system = rows[0]["messages"][0]["content"].lower()
    assert "leave alone" not in system
    assert "do not leave unchanged" in system


def test_cli_train_pairs_sft_only(tmp_path: Path):
    home = str(tmp_path)
    pairs = tmp_path / "pairs.kept.jsonl"
    pairs.write_text(
        json.dumps({"not_you": FLAT_CONTOSO, "author": VOICED_CONTOSO}) + "\n",
        encoding="utf-8",
    )
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
            "train",
            "--home",
            home,
            "--profile",
            "t",
            "--pairs",
            str(pairs),
            "--sft-only",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["status"] == "sft_ready"
    assert data["examples"] == 1
    from personality_protect.config import get_paths

    paths = get_paths("t", home=tmp_path)
    rows = [
        json.loads(line)
        for line in paths.sft_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(r["meta"]["pair_kind"] == "translator" for r in rows)
    assert not any(r["meta"]["pair_kind"] == "leave_alone" for r in rows)


def test_default_train_path_still_mints_leave_alone_without_pairs():
    """Absent --pairs: existing piece SFT path still generates leave_alone."""
    from personality_protect.models import Piece
    from personality_protect.sft import piece_to_examples

    text = (
        "I absolutely get why Contoso Labs would get into this. They have 0 knowledge "
        "about the vertical.\n\n"
        "Northwind? Keep the spine of the argument.\n\n"
        "Say something with a bite — or don't post."
    )
    examples = piece_to_examples(
        Piece(id="x", source="linkedin_post", text=text, year=2026, word_count=40)
    )
    kinds = {ex["meta"]["pair_kind"] for ex in examples}
    assert "leave_alone" in kinds
