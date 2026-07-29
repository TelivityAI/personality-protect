"""Contoso-safe tests for voice-pair gate and sterile-flattener preflight."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.pair_gate import (
    gate_jsonl,
    gate_pair,
    sterile_flattener_check,
    text_axes,
)

runner = CliRunner()

# Flat professional prose: long sentences, no short-line punch, no you>i,
# proper/1k under the input ceiling (names stripped / lowercased).
FLAT_CONTOSO = (
    "The reconciliation workflow for category twelve will ship to enterprise "
    "customers during the next planning cycle after internal review boards "
    "completed a multi-week readiness assessment. The program office stated "
    "that partner integrations remain on the published roadmap and that "
    "documentation updates will accompany the release so operations teams "
    "can prepare change-management materials without interrupting current "
    "settlement windows across regional offices."
)

# Voiced Contoso output: short lines, punch, second person.
VOICED_CONTOSO = (
    "Contoso Ledger.\n"
    "Category 12 is not a feature.\n"
    "You ship the reconciliation or you own the outage.\n"
    "Partners already know.\n"
    "Stop pretending the roadmap is the work.\n"
)

# Poisoned "flatten": kept you-address and short lines.
POISONED_FLAT = (
    "The ledger.\n"
    "You still own category twelve.\n"
    "You cannot hide behind the roadmap.\n"
    "Partners already know.\n"
)

FOREIGN_FLAT = (
    "A quarterly operations note described warehouse throughput improvements "
    "and supplier onboarding timelines for regional distribution centers "
    "across three continents after extended consultation with logistics partners "
    "and store managers responsible for fulfillment service levels."
)


def test_text_axes_you_gt_i():
    axes = text_axes(VOICED_CONTOSO)
    assert axes["you_gt_i"] is True
    assert axes["you_count"] >= 2
    flat = text_axes(FLAT_CONTOSO)
    assert flat["you_gt_i"] is False
    assert flat["proper_per_1k"] <= 30.0


def test_gate_pair_passes_clean_flatten_to_voice():
    result = gate_pair(FLAT_CONTOSO, VOICED_CONTOSO)
    assert result["pass"] is True
    assert result["failed"] == []
    assert result["input"]["you_gt_i"] is False
    assert result["median_sentence_gap"] >= 2.0
    assert result["frag_gap_ratio"] >= 0.08


def test_gate_pair_fails_input_you_gt_i():
    result = gate_pair(POISONED_FLAT, VOICED_CONTOSO)
    assert result["pass"] is False
    assert "input_you_gt_i" in result["failed"]


def test_gate_pair_fails_voiced_input_proper_density():
    dense = (
        "Contoso Fabrikam Northwind AdventureWorks Contoso Ledger ships "
        "Azure Contoso Hub with Contoso Graph and Contoso Mesh for Contoso "
        "Partners across Contoso Regions while Contoso Board reviews Contoso "
        "Roadmap items with Contoso Ops and Contoso Legal each quarter."
    )
    # Pair with a voiced-shaped output so frag/median are not the only fails.
    result = gate_pair(dense, VOICED_CONTOSO)
    assert result["input"]["proper_per_1k"] > 30.0
    assert result["pass"] is False
    assert "max_input_proper_1k" in result["failed"]


def test_gate_pair_fails_missing_frag_gap():
    twin = FLAT_CONTOSO
    result = gate_pair(twin, twin)
    assert result["pass"] is False
    assert "min_frag_gap_ratio" in result["failed"]


def test_gate_pair_fails_short_input_median():
    short_in = (
        "Contoso ships.\n"
        "Ledger updates.\n"
        "Partners wait.\n"
        "Roadmap slips.\n"
    )
    long_out = (
        "Contoso Ledger will continue shipping Category 12 updates while "
        "partners wait on documentation that never quite matches the "
        "roadmap that leadership keeps presenting as finished work."
    )
    result = gate_pair(short_in, long_out)
    assert result["pass"] is False
    assert "min_median_sentence_gap" in result["failed"]


def test_sterile_check_passes_same_band():
    result = sterile_flattener_check(FLAT_CONTOSO, FOREIGN_FLAT)
    assert result["pass"] is True


def test_sterile_check_fails_author_voice_leak():
    result = sterile_flattener_check(POISONED_FLAT, FLAT_CONTOSO)
    assert result["pass"] is False
    assert (
        "author_flat_you_gt_i" in result["failed"]
        or "you_delta" in result["failed"]
        or "frag_delta" in result["failed"]
    )


def test_gate_jsonl_keeps_and_drops(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"input": FLAT_CONTOSO, "output": VOICED_CONTOSO, "id": "ok"},
        {"input": POISONED_FLAT, "output": VOICED_CONTOSO, "id": "bad"},
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    report = gate_jsonl(path)
    assert report["total"] == 2
    assert report["kept"] == 1
    assert report["dropped"] == 1
    assert report["kept_rows"][0]["row"]["id"] == "ok"
    assert "input_you_gt_i" in report["dropped_rows"][0]["failed"]


def test_cli_pair_gate_single_json():
    result = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "pair-gate",
            "--input-text",
            FLAT_CONTOSO,
            "--output-text",
            VOICED_CONTOSO,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["pass"] is True


def test_cli_pair_gate_batch_drop_log(tmp_path: Path):
    pairs = tmp_path / "pairs.jsonl"
    keep = tmp_path / "kept.jsonl"
    drop = tmp_path / "dropped.json"
    pairs.write_text(
        json.dumps({"not_you": FLAT_CONTOSO, "author": VOICED_CONTOSO})
        + "\n"
        + json.dumps({"not_you": POISONED_FLAT, "author": VOICED_CONTOSO})
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "pair-gate",
            "--pairs",
            str(pairs),
            "--keep-out",
            str(keep),
            "--drop-log",
            str(drop),
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    summary = json.loads(result.stdout)
    assert summary["kept"] == 1
    assert summary["dropped"] == 1
    assert keep.is_file()
    assert "input_you_gt_i" in drop.read_text(encoding="utf-8")


def test_cli_sterile_check(tmp_path: Path):
    a = tmp_path / "author.md"
    f = tmp_path / "foreign.md"
    a.write_text(FLAT_CONTOSO, encoding="utf-8")
    f.write_text(FOREIGN_FLAT, encoding="utf-8")
    # Prefer subcommand-only argv; parent --logo is optional for JSON mode.
    result = runner.invoke(
        app,
        [
            "sterile-check",
            "--author-flat",
            str(a),
            "--foreign-flat",
            str(f),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["pass"] is True
