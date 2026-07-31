"""Contoso-safe tests for the writer-LoRA ship gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from personality_protect.config import init_profile
from personality_protect.eval_write_holdout import assert_receipt_contoso_safe
from personality_protect.eval_writer_adapter import (
    decide_ship,
    run_writer_adapter_gate,
    sign_test_p_value,
)
from personality_protect.models import Piece, save_index
from personality_protect.style_profile import build_style_profile, save_style_profile

CONTOSO_POST = (
    "Contoso Ledger keeps the reconciliation queue boring on purpose.\n\n"
    "You ship the reconciliation on the day it lands, or you own the outage "
    "that follows it.\n\n"
    "You name one owner before the packaging change starts, and you don't "
    "pretend the roadmap is the work of the quarter.\n\n"
    "Partners already know which one you picked this quarter — 40% of them "
    "said so in the survey.\n\n"
    "Northwind Traders tried the clever version of this and spent a year "
    "rebuilding what they had already shipped once.\n\n"
    "Boring beats clever every single time that Contoso ships Ledger.\n\n"
    "You keep the ledger boring and the partners stay calm about it."
)

ADAPTER_DRAFT = (
    "The queue stays dull because someone decided it should.\n\n"
    "Name the owner first.\n\n"
    "Ship on the day, or answer for the night that follows.\n\n"
    "Everyone downstream already knows which choice got made.\n\n"
    "Dull wins. Every quarter, without exception, it wins again."
)

RAG_DRAFT = (
    "In today's rapidly evolving operational landscape, organizations must "
    "carefully consider the strategic implications of their reconciliation "
    "processes, ensuring that ownership is clearly delineated across all "
    "relevant stakeholders and that exceptions are documented thoroughly "
    "before any packaging modification is permitted to proceed through the "
    "established review pipeline."
)


def _profile(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = [
        Piece(id="hold1", source="linkedin_post", text=CONTOSO_POST, year=2024),
        Piece(
            id="hold2",
            source="linkedin_post",
            text=CONTOSO_POST
            + "\n\nThe second release of the ledger shipped on the same day "
            "that the review closed, and nobody had to stay late for it.",
            year=2024,
        ),
    ]
    save_index(paths.index_path, pieces)
    save_style_profile(paths, build_style_profile(pieces))
    adapter = paths.adapters_dir / "latest"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapters.safetensors").write_text("stub", encoding="utf-8")
    return paths


def _fixed(text: str):
    def _generate(messages, **kwargs):  # noqa: ANN001, ANN003
        return text

    return _generate


def test_sign_test_matches_the_binomial_tail():
    assert sign_test_p_value(3, 0) == 0.125  # why n=3 could never clear a bar
    assert sign_test_p_value(0, 0) == 1.0
    assert sign_test_p_value(15, 5) == pytest.approx(0.0207, abs=1e-3)
    assert sign_test_p_value(10, 10) == pytest.approx(0.588, abs=1e-3)


def test_decide_ship_requires_a_margin_beyond_chance():
    verdict = decide_ship(
        {"adapter": 2, "rag": 1, "tie": 0}, adapter_disqualified=0, rag_disqualified=0
    )
    assert verdict["decision"] == "archive"
    assert verdict["blocking_reasons"] == ["margin_within_chance"]


def test_decide_ship_keeps_a_clear_win():
    verdict = decide_ship(
        {"adapter": 15, "rag": 5, "tie": 0}, adapter_disqualified=1, rag_disqualified=2
    )
    assert verdict["decision"] == "keep"
    assert verdict["blocking_reasons"] == []


def test_decide_ship_blocks_an_adapter_that_fabricates_more():
    verdict = decide_ship(
        {"adapter": 15, "rag": 5, "tie": 0}, adapter_disqualified=6, rag_disqualified=1
    )
    assert verdict["decision"] == "archive"
    assert "adapter_disqualified_more_often" in verdict["blocking_reasons"]


def test_decide_ship_blocks_a_minority_adapter():
    verdict = decide_ship(
        {"adapter": 1, "rag": 2, "tie": 0}, adapter_disqualified=0, rag_disqualified=0
    )
    assert verdict["decision"] == "archive"
    assert "adapter_did_not_win_majority" in verdict["blocking_reasons"]


def test_gate_runs_both_arms_and_returns_a_safe_receipt(tmp_path: Path):
    paths = _profile(tmp_path)
    receipt = run_writer_adapter_gate(
        paths,
        ["hold1", "hold2"],
        generate_fn_adapter=_fixed(ADAPTER_DRAFT),
        generate_fn_rag=_fixed(RAG_DRAFT),
        k=0,
    )
    assert receipt["kind"] == "eval_writer_adapter_gate"
    assert receipt["n_holdouts"] == 2
    assert sum(receipt["wins"].values()) == 2
    assert receipt["decision"] in {"keep", "archive"}
    assert_receipt_contoso_safe(receipt)
    assert "reconciliation" not in json.dumps(receipt)


def test_gate_records_the_pair_quality_of_every_holdout(tmp_path: Path):
    paths = _profile(tmp_path)
    receipt = run_writer_adapter_gate(
        paths,
        ["hold1"],
        generate_fn_adapter=_fixed(ADAPTER_DRAFT),
        generate_fn_rag=_fixed(RAG_DRAFT),
        k=0,
    )
    item = receipt["items"][0]
    # The gate briefs its holdouts the same way training built its pairs, so a
    # verbatim-extract brief would show up here as well.
    assert item["brief_copy_ratio"] <= 0.35


def test_gate_refuses_a_holdout_that_leaked_into_retrieval(tmp_path: Path, monkeypatch):
    paths = _profile(tmp_path)
    monkeypatch.setattr(
        "personality_protect.eval_writer_adapter.verify_holdouts_never_indexed",
        lambda *_args, **_kwargs: {"ok": False, "indexed_holdout_ids": ["hold1"]},
    )
    with pytest.raises(ValueError, match="retrieval leak"):
        run_writer_adapter_gate(
            paths,
            ["hold1"],
            generate_fn_adapter=_fixed(ADAPTER_DRAFT),
            generate_fn_rag=_fixed(RAG_DRAFT),
            k=0,
        )


def test_checkpoint_sweep_keeps_the_first_shipping_step(tmp_path: Path):
    from personality_protect.eval_writer_adapter import run_checkpoint_gate_sweep
    from personality_protect.mlx_train import persist_step_checkpoint

    paths = _profile(tmp_path)
    latest = paths.adapters_dir / "latest"
    (latest / "adapters.safetensors").write_bytes(b"early")
    persist_step_checkpoint(latest, 50)
    (latest / "adapters.safetensors").write_bytes(b"late")
    persist_step_checkpoint(latest, 100)

    calls: list[str] = []

    def _make(adapter_path: str):
        calls.append(Path(adapter_path).name)
        # Early checkpoint wins the distance game; late one mirrors the RAG loser.
        if Path(adapter_path).name == "step_000050":
            return _fixed(ADAPTER_DRAFT)
        return _fixed(RAG_DRAFT)

    sweep = run_checkpoint_gate_sweep(
        paths,
        ["hold1", "hold2"],
        make_adapter_generate=_make,
        generate_fn_rag=_fixed(RAG_DRAFT),
        k=0,
        alpha=1.0,  # majority alone is enough for Contoso stub n=2
    )
    assert sweep["decision"] == "keep"
    assert sweep["kept_checkpoint"] == "step_000050"
    assert sweep["evaluated"] == 1  # stop at first keep
    assert calls == ["step_000050"]
    assert (latest / "adapters.safetensors").read_bytes() == b"early"
    assert_receipt_contoso_safe(sweep)


def test_writer_epochs_default_is_short_enough_to_avoid_overfit():
    from personality_protect.mlx_train import WRITER_EPOCHS
    from personality_protect.train import auto_max_steps, writer_train_settings

    assert WRITER_EPOCHS == 3
    assert writer_train_settings()["epochs"] == 3
    assert auto_max_steps(60, epochs=WRITER_EPOCHS) == 180
