"""Contoso-safe longform fixtures for the article channel.

Article code paths need sources that are actually article shaped: the brief
miner segments a piece into an outline, and the length targets are percentiles
over article word counts. A 40-word stub exercises neither, so these fixtures
are built to the length band the real corpus sits in.
"""

from __future__ import annotations

from personality_protect.models import Piece

_SECTIONS: tuple[tuple[str, ...], ...] = (
    (
        "Contoso Ledger shipped a packaging change last spring and nobody could "
        "say who owned it.",
        "The pricing page listed nine tiers and the sales team quoted four of them.",
        "You cannot run a pricing experiment when the price list is already a "
        "guess.",
        "Name one owner for the packaging before anyone writes a new tier.",
        "That owner answers for the tier in writing and keeps the record where "
        "the next team can read it.",
    ),
    (
        "The review took three weeks and removed two tiers nobody had bought.",
        "Removing a tier is unglamorous work and it moved renewals more than the "
        "roadmap did.",
        "Boring tests are readable tests, and a readable test is the only kind "
        "worth running twice.",
        "Keep the experiment boring on purpose so the renewal signal stays legible.",
        "A narrow question beats a wide deck in every quarter I have worked "
        "through.",
    ),
    (
        "Exceptions are where a price list goes to die.",
        "Contoso approved forty exceptions in one quarter and each of them was "
        "reasonable on its own.",
        "Together they meant the published price described almost nobody.",
        "Cut exceptions by twelve percent before adding a discount program on top "
        "of them.",
        "Track the cut weekly, in one number, and put that number where the "
        "packaging owner has to look at it.",
    ),
    (
        "Migrations fail on boundaries, not on code.",
        "The Ledger team drew its service boundaries after the first cutover and "
        "paid for it twice.",
        "Draw the boundary first and write down what crosses it.",
        "Keep the rollback plan small enough to explain on one page to somebody "
        "who was not in the room.",
        "If the rollback needs a diagram, the change is too large to ship this "
        "week.",
    ),
    (
        "Operations queues improve when a team picks one metric and ignores the "
        "rest for a week.",
        "Queue length is usually the right one because everybody can see it "
        "without a dashboard.",
        "Bring the other metrics back only after the first one has moved twice.",
        "Write down what changed between those two moves, or the next team "
        "repeats the experiment.",
        "The record is the deliverable; the queue is just where you noticed.",
    ),
    (
        "Every packaging change needs a person who answers for it in writing.",
        "Contoso learned that after the second migration and forgot it before the "
        "third.",
        "Stop when customer value moves the wrong way and reopen the old package.",
        "Reopening is cheap in the first month and expensive in the sixth.",
        "Write down what Contoso learned before the next experiment starts, "
        "because nobody remembers the reasoning by then.",
    ),
)


def contoso_article_text(seed: int, *, sections: int = 4) -> str:
    """Deterministic longform body around 300–400 words."""
    chosen = [_SECTIONS[(seed + offset) % len(_SECTIONS)] for offset in range(sections)]
    return "\n\n".join("\n".join(block) for block in chosen)


def contoso_article(seed: int, *, sections: int = 4, year: int = 2024) -> Piece:
    """One ``linkedin_article`` piece with a stable id."""
    return Piece(
        id=f"contoso-article-{seed:02d}",
        source="linkedin_article",
        text=contoso_article_text(seed, sections=sections),
        year=year,
    )


def contoso_articles(count: int, *, sections: int = 4) -> list[Piece]:
    """``count`` distinct article pieces."""
    return [contoso_article(seed, sections=sections) for seed in range(count)]


def contoso_post(seed: int = 0) -> Piece:
    """A short post so article-only paths can prove they filter by source."""
    return Piece(
        id=f"contoso-post-{seed:02d}",
        source="linkedin_post",
        text=(
            "Contoso keeps the queue boring.\n"
            "\n"
            "You name one owner before the packaging change starts.\n"
            "\n"
            "Write the result down before anybody opens a new tier."
        ),
        year=2024,
    )
