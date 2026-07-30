"""Build the locked prompt for RAG-backed LinkedIn writing."""

from __future__ import annotations

from collections.abc import Sequence


def build_write_prompt(*, topic: str, points: str, examples: Sequence[str]) -> str:
    """Render the locked writing prompt with retrieved examples and a brief."""
    examples_text = "\n\n".join(examples)
    return (
        "System: Write a LinkedIn post in the author's voice. "
        "Match rhythm/lineation of EXAMPLES.\n"
        "Do not copy facts or names from EXAMPLES unless they appear in the BRIEF.\n"
        "No AI filler (leverage, delve, moreover, tapestry).\n"
        "\n"
        "EXAMPLES:\n"
        f"{examples_text}\n"
        "\n"
        "BRIEF:\n"
        f"Topic: {topic}\n"
        f"Points: {points}\n"
        "\n"
        "POST:"
    )
