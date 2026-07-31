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
    "- Write exactly one post and then stop. Never write a second post, a "
    "variant, an alternative version, or a separator line between posts.\n"
    "- Follow the VOICE cadence targets, including the word count target.\n"
    "- The EXAMPLES are rhythm reference only: match their line lengths, "
    "paragraph breaks, and sentence cadence.\n"
    "- Never copy, quote, continue, summarize, or list the EXAMPLES. Reuse none "
    "of their words, sentences, facts, names, numbers, or links.\n"
    # Naming a placeholder here would put the very token we do not want in
    # front of the model; the rule stays generic on purpose.
    "- Names have been removed from the EXAMPLES. Do not guess them, and never "
    "output bracketed or capitalized placeholders of any kind.\n"
    "- Do not invent companies, people, products, or figures the BRIEF did not "
    "give you.\n"
    "- No AI filler (leverage, delve, moreover, tapestry).\n"
    "- Output the post text and nothing else: no title, no preamble, "
    "no commentary, no markdown headings, no section labels, no separator "
    "lines, no hashtags copied from the EXAMPLES."
)

WRITE_ARTICLE_SYSTEM_PROMPT = (
    "You write one section of a LinkedIn article in the author's voice.\n"
    "\n"
    "Rules:\n"
    "- Write ONLY the current section about the BRIEF. The BRIEF is the only "
    "source of facts.\n"
    "- Expand claims already in the BRIEF. Prefer a shorter true section over "
    "padding that invents companies, people, products, places, or figures.\n"
    "- Do not invent companies, people, products, or figures the BRIEF did not "
    "give you. If the BRIEF has no number, write no number.\n"
    "- Follow the VOICE cadence targets. Treat the section word aim as a soft "
    "ceiling: stop early rather than invent.\n"
    "- The EXAMPLES are rhythm reference only: match their line lengths, "
    "paragraph breaks, and sentence cadence.\n"
    "- Never copy, quote, continue, summarize, or list the EXAMPLES. Reuse none "
    "of their words, sentences, facts, names, numbers, or links.\n"
    "- Names have been removed from the EXAMPLES. Do not guess them, and never "
    "output bracketed or capitalized placeholders of any kind.\n"
    "- No AI filler (leverage, delve, moreover, tapestry).\n"
    "- Output the section text and nothing else: no title, no preamble, "
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
_WRITE_ARTICLE_INSTRUCTION = "Write this article section from the BRIEF now."
_WRITE_ARTICLE_INSTRUCTION_WITH_EXAMPLES = (
    "Write this article section from the BRIEF now, in the rhythm of the "
    "EXAMPLES. Do not repeat any EXAMPLE."
)


_STYLE_HEADER = "VOICE (cadence targets measured from the author's own posts):"


def build_write_user_content(
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
    style_directives: Sequence[str] = (),
    channel: str = "post",
) -> str:
    """User turn: voice card, optional exemplars, the brief, then the instruction.

    The bare-base arm omits the EXAMPLES section entirely — and every mention of
    EXAMPLES, down to the closing instruction — rather than emitting an empty
    header. Dangling scaffolding is exactly what the model echoes, and it would
    confound the RAG-vs-base comparison.

    ``style_directives`` carry voice as measured numbers rather than copyable
    prose, so the cadence signal survives even when few or no exemplars are
    supplied.
    """
    blocks: list[str] = []
    directives = [line.strip() for line in style_directives if line and line.strip()]
    if directives:
        blocks.append(
            _STYLE_HEADER + "\n" + "\n".join(f"- {line}" for line in directives)
        )
    kept = [example.strip() for example in examples if example and example.strip()]
    if kept:
        blocks.append(_EXAMPLES_HEADER + "\n\n" + _EXAMPLE_SEPARATOR.join(kept))
    blocks.append(f"BRIEF:\nTopic: {topic}\nPoints:\n{points}")
    if channel == "article":
        blocks.append(
            _WRITE_ARTICLE_INSTRUCTION_WITH_EXAMPLES
            if kept
            else _WRITE_ARTICLE_INSTRUCTION
        )
    else:
        blocks.append(_WRITE_INSTRUCTION_WITH_EXAMPLES if kept else _WRITE_INSTRUCTION)
    return "\n\n".join(blocks)


def build_write_messages(
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
    style_directives: Sequence[str] = (),
    channel: str = "post",
) -> list[dict[str, str]]:
    """Locked writing prompt as chat turns (system + user)."""
    system = (
        WRITE_ARTICLE_SYSTEM_PROMPT if channel == "article" else WRITE_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_write_user_content(
                topic=topic,
                points=points,
                examples=examples,
                style_directives=style_directives,
                channel=channel,
            ),
        },
    ]


def build_write_prompt(
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
    style_directives: Sequence[str] = (),
    channel: str = "post",
) -> str:
    """Flat rendering of the locked prompt (no chat template available)."""
    return flatten_chat_messages(
        build_write_messages(
            topic=topic,
            points=points,
            examples=examples,
            style_directives=style_directives,
            channel=channel,
        )
    )
