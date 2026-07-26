"""Filter must return a clean rewrite, not SFT template echo; MLX must stay capped."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from personality_protect.filter import (
    build_filter_messages,
    build_filter_user_content,
    extract_rewrite,
)
from personality_protect.sft import SYSTEM_PROMPT, USER_TEMPLATE


def test_extract_rewrite_strips_template_echo_loops():
    garbage = (
        "I keep the spine of the argument.\n\n"
        "### My voice (reference)\n"
        "In today's fast-paced world we must leverage synergies.\n\n"
        "### Rewritten\n"
        "More slop leverage.\n\n"
        "### Draft\n"
        "Again.\n"
    )
    clean = extract_rewrite(garbage)
    assert clean == "I keep the spine of the argument."
    assert "###" not in clean
    assert "leverage" not in clean


def test_extract_rewrite_strips_thinking_process():
    text = (
        "Thinking Process:\n"
        "1. Strip leverage and synergies from the draft.\n"
        "2. Match the reference voice.\n\n"
        "</think>\n"
        "I cut the fog and keep the branding honest.\n\n"
        "### Draft\nignore"
    )
    clean = extract_rewrite(text)
    assert "Thinking Process" not in clean
    assert "leverage" not in clean
    assert clean == "I cut the fog and keep the branding honest."


def test_extract_rewrite_handles_rewritten_header_prefix():
    text = "### Rewritten\nPlain voice rewrite only.\n### Draft\nignore"
    assert extract_rewrite(text) == "Plain voice rewrite only."


def test_filter_user_content_matches_sft_shape_with_reference():
    user = build_filter_user_content(
        "We must leverage synergies.",
        reference="I cut the fog and keep the spine.",
    )
    assert user == USER_TEMPLATE.format(
        draft="We must leverage synergies.",
        reference="I cut the fog and keep the spine.",
    )
    assert "### Draft" in user
    assert "### My voice (reference)" in user
    assert user.rstrip().endswith("### Rewritten")


def test_filter_user_content_without_reference_still_ends_at_rewritten():
    user = build_filter_user_content("Draft with leverage.")
    assert "### Draft" in user
    assert "### My voice (reference)" not in user
    assert user.rstrip().endswith("### Rewritten")
    assert "leverage" in user  # draft preserved; rewrite is model's job


def test_filter_messages_are_chat_roles():
    messages = build_filter_messages(
        "We must leverage synergies.",
        reference="I keep it plain.",
    )
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert "### Rewritten" in messages[1]["content"]
    assert "I keep it plain." in messages[1]["content"]


def test_finalize_rewrite_strips_slop_after_template_cut():
    from personality_protect.filter import finalize_rewrite

    text = (
        "In today's fast-paced world we must leverage robust synergies.\n\n"
        "### Draft\nmore"
    )
    out = finalize_rewrite(text)
    assert "###" not in out
    assert "leverage" not in out.lower()
    assert "synergies" not in out.lower()
    """mlx_lm.generate.wired_limit requests max_recommended (~40GB); we must clamp."""
    calls: list[int] = []

    def fake_set(requested=None):
        calls.append(int(requested))
        return requested

    fake_mx = MagicMock()
    fake_mx.set_wired_limit = fake_set

    with patch.dict("sys.modules", {"mlx.core": fake_mx}):
        import personality_protect.mlx_runtime as rt

        rt._CAP_INSTALLED_FOR = None
        limit = rt.install_wired_cap(16 * 10**9)
        assert limit == 16 * 10**9
        # Simulate mlx_lm.generate.wired_limit requesting ~40 GB
        fake_mx.set_wired_limit(40 * 10**9)
        assert calls[-1] == 16 * 10**9
