"""Contoso-safe draft trimming: drop restated tails, stop at the length target."""

from __future__ import annotations

from personality_protect.draft_trim import (
    drop_repeated_paragraphs,
    drop_restated_sections,
    trim_draft,
    trim_to_word_target,
    word_count,
)

CONTOSO_DEGENERATE = (
    "Contoso Ledger is billing for its own homework.\n\n"
    "They wrote the reconciliation standard.\n\n"
    "They set the deadlines.\n\n"
    "Say no.\n\n"
    "Say no.\n\n"
    "Say no.\n"
)


def test_drop_repeated_paragraphs_removes_degenerate_refrain():
    trimmed = drop_repeated_paragraphs(CONTOSO_DEGENERATE)
    assert trimmed.count("Say no.") == 1
    assert trimmed.startswith("Contoso Ledger is billing")
    assert "They set the deadlines." in trimmed


def test_drop_repeated_paragraphs_catches_near_duplicates():
    text = (
        "You own the reconciliation outage.\n\n"
        "You own the reconciliation outage now.\n\n"
        "Partners already know the Ledger timeline.\n"
    )
    trimmed = drop_repeated_paragraphs(text)
    assert trimmed.count("You own the reconciliation outage") == 1
    assert "Partners already know" in trimmed


def test_drop_repeated_paragraphs_keeps_distinct_short_lines():
    text = "Ship the ledger.\n\nOwn the outage.\n\nName one owner.\n"
    assert len(drop_repeated_paragraphs(text).split("\n\n")) == 3


def test_drop_restated_sections_drops_paraphrase_of_earlier_section():
    first = (
        "Contoso Ledger names one owner before a packaging tier opens and "
        "keeps the renewal test boring so the signal stays readable."
    )
    paraphrase = (
        "Before a packaging tier opens Contoso Ledger names one owner and "
        "keeps the renewal test boring so its signal stays readable."
    )
    third = (
        "Exceptions are where a published price list quietly stops describing "
        "anyone who still renews Contoso."
    )
    kept = drop_restated_sections([first, paraphrase, third])
    assert kept == [first, third]


def test_trim_to_word_target_cuts_on_paragraph_boundaries():
    text = "One two three four.\n\nFive six seven eight.\n\nNine ten eleven twelve.\n"
    trimmed = trim_to_word_target(text, 8)
    assert trimmed == "One two three four.\n\nFive six seven eight."
    assert word_count(trimmed) <= 8


def test_trim_to_word_target_never_returns_empty_draft():
    text = "Contoso Ledger reconciliation is the entire first paragraph here."
    assert trim_to_word_target(text, 3) == text


def test_trim_draft_enforces_target_after_dropping_repeats():
    trimmed = trim_draft(CONTOSO_DEGENERATE, max_words=20)
    assert word_count(trimmed) <= 20
    assert trimmed.count("Say no.") <= 1
    assert trimmed.startswith("Contoso Ledger is billing")


def test_trim_draft_leaves_a_clean_draft_untouched():
    text = "Contoso named one owner.\n\nThe Ledger rollback stayed boring."
    assert trim_draft(text, max_words=100) == text
