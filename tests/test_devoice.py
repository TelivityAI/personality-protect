"""Contoso-safe tests for the de-voicing operator.

The regression these lock down is the one that sank the first writer adapter:
SFT rows whose input was a verbatim extract of their own target.
"""

from __future__ import annotations

import pytest

from personality_protect.devoice import (
    DevoiceRejected,
    devoice_report,
    devoice_sentences,
    devoice_text,
    mine_writer_brief,
    pair_copy_ratio,
)
from personality_protect.eval_write_holdout import mine_brief_from_holdout
from personality_protect.writer_guards import (
    extract_entity_keys,
    extract_named_entity_keys,
)

CONTOSO_POST = (
    "Contoso Ledger keeps the reconciliation queue boring on purpose.\n\n"
    "You ship the reconciliation on the day it lands, or you own the outage "
    "that follows it.\n\n"
    "And that is the whole point.\n\n"
    "You name one owner before the packaging change starts, and you don't "
    "pretend the roadmap is the work of the quarter.\n\n"
    "Partners already know which one you picked this quarter — 40% of them "
    "said so in the survey.\n\n"
    "You cut the exceptions, or you explain every one of them in writing. "
    "#boring @contoso\n\n"
    "Northwind Traders tried the clever version of this and spent a year "
    "rebuilding what they had already shipped once.\n\n"
    "Boring beats clever every single time that Contoso ships Ledger.\n\n"
    "You keep the ledger boring and the partners stay calm about it."
)


def test_devoice_strips_second_person_and_register():
    flat = devoice_text(CONTOSO_POST)
    assert "you" not in flat.lower()
    assert "don't" not in flat.lower()
    assert "#boring" not in flat
    assert "@contoso" not in flat


def test_devoice_keeps_entities_and_figures():
    flat = devoice_text(CONTOSO_POST)
    assert "Contoso" in flat
    assert "Ledger" in flat
    assert "40%" in flat


def test_devoice_invents_no_entity():
    flat = devoice_text(CONTOSO_POST)
    tokens = {
        token
        for key in extract_named_entity_keys(flat)
        for token in key.split(" ")
        if token
    }
    assert not tokens - extract_entity_keys(CONTOSO_POST)


def test_devoice_flattens_the_cadence_axes():
    report = devoice_report(CONTOSO_POST, devoice_text(CONTOSO_POST))
    # Author writes standalone short lines; the note is one unbroken block.
    assert report["input_axes"]["short_line_ratio"] < report["output_axes"]["short_line_ratio"]
    assert report["median_sentence_gap"] > 0
    assert report["input_axes"]["you_count"] == 0


def test_devoice_drops_cadence_only_lines():
    clauses = devoice_sentences(CONTOSO_POST)
    assert not any("whole point" in clause.lower() for clause in clauses)


def test_pair_copy_ratio_is_total_for_an_identity_pair():
    assert pair_copy_ratio(CONTOSO_POST, CONTOSO_POST) == 1.0


def test_devoice_report_rejects_an_identity_pair():
    report = devoice_report(CONTOSO_POST, CONTOSO_POST)
    assert not report["pass"]
    assert "pair_copy_ratio" in report["failed"]


def test_mined_writer_brief_is_not_an_extract_of_the_post():
    """The headline fix: the trained input no longer sits inside its target."""
    devoiced_brief, report = mine_writer_brief(CONTOSO_POST, holdout_id="c1")
    verbatim = mine_brief_from_holdout(CONTOSO_POST, holdout_id="c1")

    devoiced_ratio = pair_copy_ratio(
        f"{devoiced_brief['topic']}\n{devoiced_brief['points']}", CONTOSO_POST
    )
    verbatim_ratio = pair_copy_ratio(
        f"{verbatim['topic']}\n{verbatim['points']}", CONTOSO_POST
    )
    assert verbatim_ratio > 0.9, "shipped mining hands the post back nearly whole"
    assert devoiced_ratio <= report["max_copy_ratio"]
    assert devoiced_ratio < verbatim_ratio


def test_mined_writer_brief_guards_against_the_post_not_the_note():
    brief, _ = mine_writer_brief(CONTOSO_POST, holdout_id="c1")
    # Invention is judged against what the author actually wrote, so a figure
    # dropped by the operator must not become "invented" in a draft.
    assert "40%" in brief["guard_facts"]


def test_mine_writer_brief_rejects_a_pair_it_cannot_move():
    with pytest.raises(DevoiceRejected):
        mine_writer_brief("Ledger. Ledger. Ledger. Ledger.", holdout_id="c2")
