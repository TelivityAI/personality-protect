"""Tests for the shared chat-template rendering helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from personality_protect.chat_prompt import (
    flatten_chat_messages,
    render_chat_prompt,
    tokenizer_has_chat_template,
)


def test_flatten_chat_messages_joins_system_and_user():
    text = flatten_chat_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Write a Contoso note."},
        ]
    )
    assert text == "Be brief.\n\nWrite a Contoso note.\n"


def test_tokenizer_has_chat_template_checks_both_flags():
    tok = MagicMock()
    tok.has_chat_template = False
    tok.chat_template = None
    assert tokenizer_has_chat_template(tok) is False

    tok.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    assert tokenizer_has_chat_template(tok) is True


def test_render_chat_prompt_applies_template_with_thinking_disabled():
    tok = MagicMock()
    tok.has_chat_template = True
    tok.chat_template = "unused"
    tok.apply_chat_template.return_value = "<|im_start|>user\nHi<|im_end|>\n"

    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    out = render_chat_prompt(tok, messages)

    assert out == "<|im_start|>user\nHi<|im_end|>\n"
    tok.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def test_render_chat_prompt_retries_without_thinking_kwarg():
    tok = MagicMock()
    tok.has_chat_template = True
    tok.chat_template = "unused"

    def apply(*_args, **kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword")
        return "templated"

    tok.apply_chat_template.side_effect = apply

    out = render_chat_prompt(
        tok,
        [{"role": "user", "content": "Hi"}],
    )
    assert out == "templated"
    assert tok.apply_chat_template.call_count == 2


def test_render_chat_prompt_falls_back_when_no_template():
    tok = MagicMock()
    tok.has_chat_template = False
    tok.chat_template = None

    out = render_chat_prompt(
        tok,
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Write Contoso."},
        ],
        fallback="CUSTOM FALLBACK\n",
    )
    assert out == "CUSTOM FALLBACK\n"
    tok.apply_chat_template.assert_not_called()
