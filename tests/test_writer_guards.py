"""Contoso-safe tests for RAG writer entity, parrot, and invention guards."""

from __future__ import annotations

from personality_protect.writer_guards import (
    check_invention,
    find_parroted_ngrams,
    mask_exemplar_entities,
    parrot_reject,
)


def test_mask_exemplar_entities_absent_from_brief():
    brief = "Contoso is launching Ledger for operations teams."
    exemplar = (
        "Fabrikam rebuilt its queue with Northwind Traders. Contoso kept Ledger deliberately small."
    )

    masked = mask_exemplar_entities(exemplar, brief)

    assert "Fabrikam" not in masked
    assert "Northwind Traders" not in masked
    assert masked.count("[ENTITY]") == 2
    assert "Contoso" in masked
    assert "Ledger" in masked


def test_mask_exemplar_entities_is_case_insensitive_and_preserves_layout():
    brief = "The CONTOSO team uses LEDGER."
    exemplar = "Contoso ships.\n\nLedger stays boring."

    assert mask_exemplar_entities(exemplar, brief) == exemplar


def test_parrot_rejects_normalized_exemplar_ngram():
    exemplar = "Contoso made the hard part boring by naming one owner for every queue."
    draft = "A useful lesson: made the hard part boring, by naming one owner for every queue."

    matches = find_parroted_ngrams(draft, [exemplar], n=8)

    assert "made the hard part boring by naming one" in matches
    assert parrot_reject(draft, [exemplar], n=8) is True


def test_parrot_allows_overlap_shorter_than_threshold():
    exemplar = "Contoso made the hard part boring by naming one owner for every queue."
    draft = "We made the hard part boring, then stopped."

    assert find_parroted_ngrams(draft, [exemplar], n=8) == set()
    assert parrot_reject(draft, [exemplar], n=8) is False


def test_invention_guard_accepts_entities_and_figures_from_brief():
    brief = "Contoso cut Ledger exceptions by 12% over seven weeks."
    draft = "Seven weeks. Contoso cut Ledger exceptions by 12%."

    result = check_invention(brief, draft)

    assert result.passed is True
    assert result.invented_entities == frozenset()
    assert result.invented_numbers == frozenset()


def test_invention_guard_rejects_entity_absent_from_brief():
    brief = "Contoso is simplifying Ledger exceptions."
    draft = "Contoso should copy Fabrikam and simplify Ledger exceptions."

    result = check_invention(brief, draft)

    assert result.passed is False
    assert result.invented_entities == frozenset({"fabrikam"})
    assert result.invented_numbers == frozenset()


def test_invention_guard_rejects_figure_absent_from_brief():
    brief = "Contoso is simplifying Ledger exceptions."
    draft = "Contoso cut Ledger exceptions by 18%."

    result = check_invention(brief, draft)

    assert result.passed is False
    assert result.invented_numbers == frozenset({"18%"})


def test_invention_guard_uses_brief_not_exemplar_entities():
    brief = "Contoso is simplifying Ledger exceptions."
    exemplar = "Fabrikam gave every queue one owner."
    draft = "Fabrikam should simplify Ledger exceptions."

    # Exemplar text is deliberately not an input to the invention guard.
    assert "Fabrikam" in exemplar
    assert check_invention(brief, draft).passed is False


def test_invention_guard_ignores_common_nouns_and_prompt_scaffolding():
    """Real signal only: ordinary words and section labels are not entities.

    The dogfood smoke run flagged ``brief``, ``examples``, ``entity``,
    ``match``, and ``ai`` as invented companies — every draft failed, so the
    single regen carried no information. Those must pass cleanly.
    """
    brief = "Topic: Contoso pricing\nPoints: Keep the test boring; name one owner."
    draft = (
        "Keep the Match boring.\n"
        "\n"
        "Examples help, but AI is not the Brief.\n"
        "\n"
        "Name one owner before you ship Contoso."
    )

    result = check_invention(brief, draft)

    assert result.passed is True
    assert result.invented_entities == frozenset()


def test_invention_guard_still_catches_real_proper_names_and_numbers():
    brief = "Topic: Contoso pricing\nPoints: Name one owner; keep Ledger boring."
    draft = "Contoso should copy Fabrikam and Northwind Traders, then cut exceptions by 18%."

    result = check_invention(brief, draft)

    assert result.passed is False
    assert "fabrikam" in result.invented_entities
    # Ordinary second token ("Traders") is shed; the unfamiliar stem remains.
    assert "northwind" in result.invented_entities
    assert "18%" in result.invented_numbers


def test_invention_guard_ignores_sentence_initial_verbs():
    brief = "Topic: Contoso ops\nPoints: Name one owner."
    draft = "Stop pretending the roadmap is the work.\nKeep Contoso boring."

    assert check_invention(brief, draft).passed is True
