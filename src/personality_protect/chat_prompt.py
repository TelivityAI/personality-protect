"""Shared chat-template rendering for instruct models (filter + write paths).

Qwen3.5-instruct only behaves like an assistant when the prompt is wrapped in
its chat template. Handing it a bare text block makes it continue the document
instead — it echoes section headers ("EXAMPLES", "BRIEF", "Task") back as if it
were completing a form. Every MLX entrypoint therefore renders through
:func:`render_chat_prompt`.

This module never imports mlx; it only touches a tokenizer object supplied by
the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Message = Mapping[str, str]


def flatten_chat_messages(messages: Sequence[Message]) -> str:
    """Plain-text rendering for backends without a chat template.

    Used as the fallback prompt and as the human-readable artifact when a
    tokenizer is unavailable (tests, receipts, raw dumps).
    """
    parts: list[str] = []
    for message in messages:
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(content)
    text = "\n\n".join(parts)
    return text + "\n" if text and not text.endswith("\n") else text


def tokenizer_has_chat_template(tokenizer: Any) -> bool:
    """True when the tokenizer can render chat turns."""
    return bool(
        getattr(tokenizer, "has_chat_template", False) or getattr(tokenizer, "chat_template", None)
    )


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    fallback: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Render ``messages`` through the tokenizer chat template.

    ``enable_thinking=False`` matters for Qwen3: the default burns the token
    budget on a "Thinking Process" preamble and never returns a clean answer.
    Tokenizers that reject the kwarg are retried without it, and tokenizers
    with no template at all fall back to :func:`flatten_chat_messages`.
    """
    turns = [dict(message) for message in messages]
    if not tokenizer_has_chat_template(tokenizer):
        return fallback if fallback is not None else flatten_chat_messages(turns)
    try:
        return str(
            tokenizer.apply_chat_template(
                turns,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                turns,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
