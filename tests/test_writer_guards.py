"""Contoso-safe tests for RAG writer entity, parrot, and invention guards."""

from __future__ import annotations

from personality_protect.writer_guards import (
    brief_echo_reject,
    check_invention,
    copied_token_ratio,
    find_parroted_ngrams,
    find_scaffold_markers,
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
    assert "Contoso" in masked
    assert "Ledger" in masked


def test_mask_exemplar_entities_leaves_no_placeholder_token():
    """A visible slot in the prompt comes back out in the draft — emit none."""
    brief = "Topic: Contoso pricing"
    exemplar = (
        "Fabrikam's queue was a mess.\n"
        "\n"
        "Northwind Traders fixed it with Ledger Pro — one owner, one metric.\n"
        "\n"
        "#Fabrikam #Northwind"
    )

    masked = mask_exemplar_entities(exemplar, brief)

    assert "[" not in masked and "]" not in masked
    assert "ENTITY" not in masked
    assert "Fabrikam" not in masked
    assert "Northwind" not in masked
    # Prose survives redaction without doubled spaces or dangling punctuation.
    assert "  " not in masked
    assert "queue was a mess." in masked
    assert " ." not in masked


def test_mask_exemplar_entities_drops_possessive_after_redacted_name():
    masked = mask_exemplar_entities("Fabrikam's board asked for one metric.", "")

    assert masked == "board asked for one metric."


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


def _contoso_exemplars() -> list[str]:
    return [
        (
            "Fabrikam is testing queue ads right now.\n"
            "\n"
            "The industry is drowning in noise and starving for owners.\n"
            "\n"
            "So we named one owner per queue and shipped it.\n"
            "\n"
            "#Fabrikam #Queues"
        ),
        (
            "Northwind Traders shipped Ledger Pro in seven weeks.\n"
            "\n"
            "Rate plans, night audit, billing, reporting — all of it.\n"
            "\n"
            "Most operators pay for closed systems that treat data as an afterthought."
        ),
    ]


def test_parrot_rejects_exemplar_dump_after_masking():
    """The failure mode from the dogfood run: paste the redacted exemplars back."""
    exemplars = _contoso_exemplars()
    dump = "\n\n===\n\n".join(mask_exemplar_entities(text, "") for text in exemplars)

    assert copied_token_ratio(dump, exemplars) > 0.9
    assert parrot_reject(dump, exemplars) is True


def test_parrot_rejects_dump_that_kept_placeholder_tokens():
    exemplars = _contoso_exemplars()
    dump = "[ENTITY] is testing queue ads right now.\n\n[ENTITY] named one owner."

    assert "placeholder" in find_scaffold_markers(dump)
    assert parrot_reject(dump, exemplars) is True
    # A placeholder is a scaffolding echo even with no exemplars to compare to.
    assert parrot_reject(dump, ()) is True


def test_parrot_rejects_echoed_prompt_scaffolding():
    draft = (
        "EXAMPLES (rhythm reference only):\n"
        "\n"
        "One owner per queue.\n"
        "\n"
        "---\n"
        "\n"
        "BRIEF:\n"
        "Topic: Contoso queues"
    )

    markers = find_scaffold_markers(draft)

    assert {"examples_header", "brief_header", "block_separator"} <= markers
    assert parrot_reject(draft, ()) is True


def test_parrot_allows_a_real_post_that_shares_one_phrase():
    exemplars = _contoso_exemplars()
    draft = (
        "Contoso named one owner per queue this quarter.\n"
        "\n"
        "Not a committee. Not a dashboard. One person who answers for the "
        "backlog and gets to say no.\n"
        "\n"
        "The industry is drowning in noise, and every extra reviewer buys "
        "another week of delay.\n"
        "\n"
        "So we picked one metric, watched it for a week, and left everything "
        "else alone.\n"
        "\n"
        "Boring beats clever when the queue is on fire."
    )

    assert copied_token_ratio(draft, exemplars) < 0.3
    assert parrot_reject(draft, exemplars) is False


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


def test_mask_exemplar_entities_keeps_ordinary_possessives():
    """Only a redacted name loses its ``'s``; ordinary contractions survive."""
    exemplar = "That's it. The industry's habit is confusing noise with progress."

    assert mask_exemplar_entities(exemplar, "") == exemplar


def test_brief_echo_allows_a_post_built_from_the_brief():
    """Reusing the brief's facts and wording is the job, not a failure."""
    brief = (
        "Topic: Contoso queue ownership\n"
        "Points: - Name one owner per queue\n"
        "- Watch one metric for a week\n"
        "- Keep the rollback boring"
    )
    draft = (
        "We named one owner per queue last month.\n"
        "\n"
        "Not a committee. One person who answers for the backlog and gets to "
        "say no when the list grows.\n"
        "\n"
        "Then we watched one metric for a week and ignored the rest, because "
        "every extra dashboard buys another argument.\n"
        "\n"
        "The rollback stayed boring on purpose. Boring is what you want at "
        "two in the morning."
    )

    assert brief_echo_reject(draft, brief) is False


def test_brief_echo_rejects_the_bullets_handed_back():
    brief = (
        "Topic: Contoso queue ownership\n"
        "Points: - Name one owner per queue\n"
        "- Watch one metric for a week\n"
        "- Keep the rollback boring and reversible for the on-call engineer"
    )
    echo = (
        "Name one owner per queue\n"
        "Watch one metric for a week\n"
        "Keep the rollback boring and reversible for the on-call engineer"
    )

    assert brief_echo_reject(echo, brief) is True
