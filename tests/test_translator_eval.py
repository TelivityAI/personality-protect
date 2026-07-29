"""Contoso-safe Phase 4: translator holdout eval (axes toward author; no echo)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app

runner = CliRunner()

# Foreign holdout — press-release band (text the Contoso "author" did not write).
FOREIGN_STERILE = (
    "A quarterly operations bulletin outlined warehouse throughput improvements "
    "and supplier onboarding timelines for regional distribution centers across "
    "three continents. Leadership stated that fulfillment service levels remain "
    "aligned with published targets after extended consultation with logistics "
    "partners and store managers responsible for inventory continuity during "
    "peak seasonal demand windows."
)

# Held-out Contoso author post (voice band reference — not training target copy).
AUTHOR_HOLDOUT = (
    "Contoso Ledger.\n"
    "\n"
    "Category 12 is not a feature.\n"
    "\n"
    "You ship the reconciliation or you own the outage.\n"
    "\n"
    "Partners already know.\n"
    "\n"
    "Stop pretending the roadmap is the work.\n"
)

# Successful translator rewrite of foreign → Contoso voice band.
SUCCESS_REWRITE = (
    "Warehouse throughput is not a Contoso Ledger bulletin.\n"
    "\n"
    "You either hit fulfillment or you own the miss.\n"
    "\n"
    "Partners already know the suppliers are late.\n"
    "\n"
    "Stop dressing the timeline as a strategy.\n"
)

# Still press-release: paraphrased sterile, no fragment punch / you·I move.
PRESS_RELEASE_ECHO = (
    "An operations bulletin described warehouse throughput improvements and "
    "supplier onboarding timelines for regional distribution centers. Leadership "
    "confirmed that fulfillment service levels stay aligned with published targets "
    "following consultation with logistics partners and store managers during "
    "peak seasonal demand windows."
)


def test_packaged_translator_holdout_fixtures_are_contoso_safe():
    from personality_protect.translator_eval import (
        load_packaged_author_holdout,
        load_packaged_foreign_holdout,
    )

    foreign = load_packaged_foreign_holdout()
    author = load_packaged_author_holdout()
    assert "Contoso" not in foreign  # foreign = not authored Contoso voice
    assert "Contoso" in author
    assert "you" in author.lower()
    # No personal / private markers in public fixtures.
    banned = ("linkedin.com", "@gmail", "dusan", "telivity.ai/private")
    blob = (foreign + "\n" + author).lower()
    for token in banned:
        assert token not in blob


def test_score_success_moves_axes_toward_author_band():
    from personality_protect.translator_eval import score_translator_holdout

    result = score_translator_holdout(
        FOREIGN_STERILE, SUCCESS_REWRITE, AUTHOR_HOLDOUT
    )
    assert result["pass"] is True
    assert result["failed"] == []
    assert result["echo"] is False
    assert result["axes_moved"]["frag"] is True
    assert result["axes_moved"]["you_i"] is True


def test_score_fails_byte_identical_echo():
    from personality_protect.translator_eval import score_translator_holdout

    result = score_translator_holdout(
        FOREIGN_STERILE, FOREIGN_STERILE, AUTHOR_HOLDOUT
    )
    assert result["pass"] is False
    assert "byte_identical_echo" in result["failed"]
    assert result["echo"] is True


def test_score_fails_still_press_release_band():
    from personality_protect.translator_eval import score_translator_holdout

    result = score_translator_holdout(
        FOREIGN_STERILE, PRESS_RELEASE_ECHO, AUTHOR_HOLDOUT
    )
    assert result["pass"] is False
    assert "still_press_release" in result["failed"] or any(
        code.startswith("frag_") or code.startswith("you_i_")
        for code in result["failed"]
    )


def test_score_fixture_packaged_paths_match_loaders():
    from personality_protect.translator_eval import (
        load_packaged_author_holdout,
        load_packaged_foreign_holdout,
        score_translator_holdout,
    )

    result = score_translator_holdout(
        load_packaged_foreign_holdout(),
        SUCCESS_REWRITE,
        load_packaged_author_holdout(),
    )
    assert result["pass"] is True


def test_cli_translator_eval_scores_files(tmp_path: Path):
    inp = tmp_path / "foreign.txt"
    out = tmp_path / "rewrite.txt"
    author = tmp_path / "author.txt"
    inp.write_text(FOREIGN_STERILE, encoding="utf-8")
    out.write_text(SUCCESS_REWRITE, encoding="utf-8")
    author.write_text(AUTHOR_HOLDOUT, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "translator-eval",
            "--input-file",
            str(inp),
            "--output-file",
            str(out),
            "--author-band",
            str(author),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"pass": true' in result.output.lower() or '"pass": true' in result.output


def test_cli_translator_eval_packaged_fixtures_echo_fails(tmp_path: Path):
    out = tmp_path / "echo.txt"
    # Echo the packaged foreign text — must fail.
    root = resources.files("personality_protect").joinpath("data/evals")
    foreign = (root / "translator_foreign.txt").read_text(encoding="utf-8")
    out.write_text(foreign, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "translator-eval",
            "--packaged",
            "--output-file",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "byte_identical_echo" in result.output
