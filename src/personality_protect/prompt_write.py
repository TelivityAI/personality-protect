"""Build the locked prompt for RAG-backed LinkedIn writing.

The prompt is a chat exchange, not a text block. Qwen3.5-instruct handed a bare
"EXAMPLES: … BRIEF: … POST:" block continues the *document* and echoes the
section headers back as if completing a form; wrapped as system + user it
writes the post. :func:`build_write_prompt` keeps a flat rendering for backends
with no chat template and for human-readable raw artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence

from personality_protect.chat_prompt import flatten_chat_messages

WRITE_SYSTEM_PROMPT = (
    "You write LinkedIn posts in the author's voice.\n"
    "\n"
    "Rules:\n"
    "- Write ONE new post about the BRIEF. The BRIEF is the only source of "
    "content.\n"
    "- The EXAMPLES are rhythm reference only: match their line lengths, "
    "paragraph breaks, and sentence cadence.\n"
    "- Never copy, quote, continue, summarize, or list the EXAMPLES. Reuse none "
    "of their words, sentences, facts, names, numbers, or links.\n"
    "- Names have been removed from the EXAMPLES. Do not guess them, and never "
    "output bracketed placeholders such as [ENTITY] or [NAME].\n"
    "- Do not invent companies, people, products, or figures the BRIEF did not "
    "give you.\n"
    "- No AI filler (leverage, delve, moreover, tapestry).\n"
    "- Output the post text and nothing else: no title, no preamble, "
    "no commentary, no markdown headings, no section labels, no separator "
    "lines, no hashtags copied from the EXAMPLES."
)

_EXAMPLES_HEADER = (
    "EXAMPLES (rhythm reference only — names removed; never reuse their words, "
    "facts, or names):"
)
_EXAMPLE_SEPARATOR = "\n\n===\n\n"
_WRITE_INSTRUCTION = "Write one new post from the BRIEF now."
_WRITE_INSTRUCTION_WITH_EXAMPLES = (
    "Write one new post from the BRIEF now, in the rhythm of the EXAMPLES. "
    "Do not repeat any EXAMPLE."
)


def build_write_user_content(
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
) -> str:
    """User turn: optional exemplars, then the brief, then the instruction.

    The bare-base arm omits the EXAMPLES section entirely — and every mention of
    EXAMPLES, down to the closing instruction — rather than emitting an empty
    header. Dangling scaffolding is exactly what the model echoes, and it would
    confound the RAG-vs-base comparison.
    """
    blocks: list[str] = []
    kept = [example.strip() for example in examples if example and example.strip()]
    if kept:
        blocks.append(_EXAMPLES_HEADER + "\n\n" + _EXAMPLE_SEPARATOR.join(kept))
    blocks.append(f"BRIEF:\nTopic: {topic}\nPoints:\n{points}")
    blocks.append(_WRITE_INSTRUCTION_WITH_EXAMPLES if kept else _WRITE_INSTRUCTION)
    return "\n\n".join(blocks)


def build_write_messages(
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
) -> list[dict[str, str]]:
    """Locked writing prompt as chat turns (system + user)."""
    return [
        {"role": "system", "content": WRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_write_user_content(topic=topic, points=points, examples=examples),
        },
    ]


def build_write_prompt(*, topic: str, points: str, examples: Sequence[str]) -> str:
    """Flat rendering of the locked prompt (no chat template available)."""
    return flatten_chat_messages(
        build_write_messages(topic=topic, points=points, examples=examples)
    )
