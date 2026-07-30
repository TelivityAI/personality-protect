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
    "- Match the rhythm and lineation of the EXAMPLES: same line lengths, "
    "same paragraph breaks, same sentence cadence.\n"
    "- Do not copy facts, names, numbers, or phrasing from the EXAMPLES. "
    "They are voice reference only.\n"
    "- Use only what the BRIEF gives you. Do not invent companies, people, "
    "products, or figures.\n"
    "- No AI filler (leverage, delve, moreover, tapestry).\n"
    "- Output the post text and nothing else: no title, no preamble, "
    "no commentary, no markdown headings, no section labels."
)

_EXAMPLES_HEADER = "EXAMPLES (voice reference only — never reuse their facts or names):"
_EXAMPLE_SEPARATOR = "\n\n---\n\n"
_WRITE_INSTRUCTION = "Write the post now."


def build_write_user_content(
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
) -> str:
    """User turn: optional exemplars, then the brief, then the instruction.

    The bare-base arm omits the EXAMPLES section entirely rather than emitting
    an empty header — dangling scaffolding is exactly what the model echoes,
    and it would confound the RAG-vs-base comparison.
    """
    blocks: list[str] = []
    kept = [example.strip() for example in examples if example and example.strip()]
    if kept:
        blocks.append(_EXAMPLES_HEADER + "\n\n" + _EXAMPLE_SEPARATOR.join(kept))
    blocks.append(f"BRIEF:\nTopic: {topic}\nPoints:\n{points}")
    blocks.append(_WRITE_INSTRUCTION)
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
