"""Contoso-safe tests for the widened writer holdout carve."""

from __future__ import annotations

from personality_protect.models import Piece
from personality_protect.writer_holdout import (
    is_briefable,
    resolve_holdout_n,
    select_writer_holdouts,
)

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


def _pieces(n: int) -> list[Piece]:
    return [
        Piece(
            id=f"c{index:03d}",
            source="linkedin_post",
            # Vary the tail so ids differ without changing brief-ability.
            text=CONTOSO_POST + f"\n\nRelease {index} shipped on the same day.",
            year=2024,
        )
        for index in range(n)
    ]


def test_fixture_pieces_are_briefable():
    assert is_briefable(_pieces(1)[0])


def test_short_or_wrong_source_pieces_are_not_briefable():
    assert not is_briefable(Piece(id="s", source="linkedin_post", text="Too short.", year=2024))
    assert not is_briefable(
        Piece(id="a", source="linkedin_article", text=CONTOSO_POST, year=2024)
    )


def test_resolve_holdout_n_clamps_to_a_usable_band():
    assert resolve_holdout_n(0) == 0
    assert resolve_holdout_n(8) == 8  # never more than the pool
    assert resolve_holdout_n(40) == 12  # floor beats a 25% share this small
    assert resolve_holdout_n(80) == 20
    assert resolve_holdout_n(400) == 24  # ceiling


def test_selection_is_deterministic_and_order_independent():
    pieces = _pieces(40)
    first = select_writer_holdouts(pieces)
    again = select_writer_holdouts(list(reversed(pieces)))
    assert first["holdout_ids"] == again["holdout_ids"]
    assert first["n_holdouts"] == 12


def test_widened_carve_is_far_larger_than_the_failed_gate():
    receipt = select_writer_holdouts(_pieces(80))
    assert receipt["n_holdouts"] >= 12
    assert receipt["train_pairs_remaining"] > receipt["n_holdouts"]


def test_pinned_ids_are_always_kept():
    pieces = _pieces(40)
    receipt = select_writer_holdouts(pieces, pinned_ids=["c039", "c000"])
    assert {"c039", "c000"} <= set(receipt["holdout_ids"])
    assert receipt["pinned_ids"] == ["c000", "c039"]


def test_pinned_ids_absent_from_the_corpus_are_reported_not_carved():
    receipt = select_writer_holdouts(_pieces(20), pinned_ids=["gone"])
    assert receipt["pinned_ids_missing_from_corpus"] == ["gone"]
    assert "gone" not in receipt["holdout_ids"]


def test_receipt_carries_no_piece_text():
    blob = repr(select_writer_holdouts(_pieces(20)))
    assert "reconciliation" not in blob
