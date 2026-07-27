"""Filter must return a clean rewrite, not SFT template echo; MLX must stay capped."""

from __future__ import annotations

from unittest.mock import patch

from personality_protect.filter import (
    build_filter_messages,
    build_filter_user_content,
    extract_rewrite,
    finalize_rewrite,
    similarity_guard,
    strip_ai_tells,
)
from personality_protect.sft import SYSTEM_PROMPT, USER_TEMPLATE, USER_TEMPLATE_INFER


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
    # Default inference shape must match SFT (no reference dump to regurgitate)
    assert user == USER_TEMPLATE_INFER.format(draft="Draft with leverage.")
    assert "cadence" in user.lower()


def test_prompts_preserve_paragraphs_and_allow_leave_alone():
    """Voice filter must keep paragraph rhythm and not force fidget rewrites."""
    sys_l = SYSTEM_PROMPT.lower()
    assert "paragraph" in sys_l
    assert "unchanged" in sys_l or "leave alone" in sys_l
    assert "opener" in sys_l or "openings" in sys_l
    assert "thesaurus" in sys_l or "multi-paragraph" in sys_l or "blank lines" in sys_l
    assert "soulless" in sys_l or "clean" in sys_l
    user = USER_TEMPLATE_INFER.format(draft="Companies need a clear point of view.")
    user_l = user.lower()
    assert "paragraph" in user_l
    assert "unchanged" in user_l
    assert "opener" in user_l or "openings" in user_l
    assert "multi-paragraph" in user_l or "thesaurus" in user_l or "blank lines" in user_l
    assert "soulless" in user_l or "clean" in user_l


def test_strip_ai_tells_preserves_paragraph_breaks():
    text = (
        "In today's fast-paced world we must leverage synergies.\n\n"
        "Second punch stays on its own line.\n\n"
        "Third short one."
    )
    out = strip_ai_tells(text)
    assert "\n\n" in out
    assert out.count("\n\n") >= 2
    assert "leverage" not in out.lower()
    assert "Second punch" in out


def test_strip_ai_tells_does_not_mint_thesaurus_mush():
    """Phrase-level cleanup — not unlock→open / nestled→in nonsense."""
    text = (
        "We must leverage robust synergies to delve into authentic personal branding.\n\n"
        "Unlocking nestled opportunities is a testament to vibrant innovation."
    )
    out = strip_ai_tells(text)
    assert "\n\n" in out or "branding" in out.lower()
    assert "open in" not in out.lower()
    assert "solid strengths" not in out.lower()
    assert "leverage" not in out.lower()
    assert "nestled" not in out.lower()
    assert "testament" not in out.lower()
    assert "branding" in out.lower()
    # Whole mush clause deleted — not "find real opportunities is innovation."
    assert "is innovation" not in out.lower()


def test_prefer_multipara_on_slop_paragraphizes_flat_rewrite():
    from personality_protect.filter import prefer_multipara_on_slop

    draft = (
        "In today's fast-paced world, Contoso must leverage robust synergies. "
        "Moreover, unlocking nestled opportunities is a testament to vibrant innovation."
    )
    flat = (
        "Contoso doesn't need another synonym parade. Who's saying something real. "
        "Ship the take."
    )
    out = prefer_multipara_on_slop(draft, flat)
    assert "\n\n" in out
    assert out.count("\n\n") >= 1
    # Already-in-voice drafts are left alone (leave-alone path).
    good = (
        "These questions keep popping up every time I read another take.\n\n"
        "How much is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders."
    )
    assert prefer_multipara_on_slop(good, flat) == flat
    soulless = (
        "Personal branding is increasingly essential as AI systems appear across "
        "every channel. Organizations require a distinct perspective."
    )
    assert "\n\n" in prefer_multipara_on_slop(soulless, flat)


def test_similarity_guard_does_not_freeze_multipara_slop():
    """Multi-para Contoso mush must still be rewritten — never leave-alone."""
    draft = (
        "In today's fast-paced world, Contoso must leverage robust synergies.\n\n"
        "Moreover, unlocking nestled opportunities is a testament to vibrant innovation."
    )
    voiced = (
        "Let's be real.\n\n"
        "Contoso doesn't need a synonym parade.\n\n"
        "Ship the take."
    )
    assert similarity_guard(draft, voiced) == voiced


def test_similarity_guard_returns_draft_when_near_identity():
    draft = (
        "These questions keep popping up every time I read another take.\n\n"
        "How much is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders."
    )
    fidget = (
        "These questions keep coming up every time I read another take.\n\n"
        "How much is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders."
    )
    assert similarity_guard(draft, fidget) == draft
    different = "I cut the fog and keep the branding honest."
    assert similarity_guard(draft, different) == different
    flat = "Personal branding matters more than ever as AI tools flood every channel."
    voiced = "Personal branding matters. AI floods every channel — say something real."
    assert similarity_guard(flat, voiced) == voiced


def test_similarity_guard_keeps_draft_when_only_reparagraphed():
    """Leave-alone: same substance + new blank lines must not count as a rewrite."""
    draft = (
        "These questions keep popping up every time I read another vibe coding take.\n\n"
        "How much of the labs' revenue is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders. Drucker said create a customer."
    )
    fidget = (
        "These questions keep popping up every time I read another vibe coding take.\n\n"
        "How much of the labs' revenue is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders.\n\n"
        "Drucker said create a customer."
    )
    assert similarity_guard(draft, fidget) == draft


def test_similarity_guard_does_not_freeze_soulless_clean():
    from personality_protect.filter import (
        draft_already_in_voice,
        draft_looks_soulless,
        similarity_guard,
    )

    draft = (
        "Personal branding is increasingly essential as AI systems appear across "
        "every channel. Organizations require a distinct perspective rather than "
        "a templated statement regarding genuine positioning."
    )
    voiced = (
        "Personal branding matters more than ever.\n\n"
        "AI floods every channel.\n\n"
        "Who's saying something real?"
    )
    assert draft_looks_soulless(draft)
    assert not draft_already_in_voice(draft)
    assert similarity_guard(draft, voiced) == voiced
    blank_only = (
        "Personal branding is increasingly essential as AI systems appear across "
        "every channel.\n\n"
        "Organizations require a distinct perspective rather than a templated "
        "statement regarding genuine positioning."
    )
    assert similarity_guard(draft, blank_only) == blank_only


def test_draft_already_in_voice_detects_vibe_shaped():
    from personality_protect.filter import draft_already_in_voice, draft_looks_soulless

    vibe = (
        "These questions keep popping up every time I read another vibe coding take.\n\n"
        "How much of the labs' revenue is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders."
    )
    assert draft_already_in_voice(vibe)
    assert not draft_looks_soulless(vibe)


def test_dedupe_repetition_collapse_cuts_loops():
    from personality_protect.filter import _dedupe_repetition_collapse, finalize_rewrite

    spam = "\n\n".join(
        [f"AI will never say “{w}.”" for w in (
            "connections", "Fast-paced", "Nested", "real", "Testament",
            "Dynamic", "Multifaceted", "Catalyst", "Confluence", "Disruption",
        )]
        + ["AI will never say"] * 20
    )
    out = _dedupe_repetition_collapse(spam)
    assert out.count("AI will never") <= 6
    block = (
        "So why settle for hype when the AI floods the channel?\n\n"
        "The point is authenticity.\n\n"
        "When everyone sounds the same, authenticity cuts through.\n\n"
    )
    collapsed = _dedupe_repetition_collapse(block * 6)
    assert collapsed.count("The point is authenticity") == 1
    assert finalize_rewrite(spam).count("AI will never") <= 6


def test_filter_messages_default_omits_long_reference():
    """LoRA path: cadence via weights, not pasted corpus blocks."""
    messages = build_filter_messages("We must leverage synergies.")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "### My voice" not in messages[1]["content"]
    assert messages[1]["content"] == USER_TEMPLATE_INFER.format(
        draft="We must leverage synergies."
    )


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
    text = (
        "In today's fast-paced world we must leverage robust synergies.\n\n"
        "### Draft\nmore"
    )
    out = finalize_rewrite(text)
    assert "###" not in out
    assert "leverage" not in out.lower()
    assert "synergies" not in out.lower()


def test_mlx_wired_cap_clamps_generate_requests(monkeypatch):
    """mlx_lm.generate.wired_limit requests max_recommended (~40GB); we must clamp."""
    import types

    monkeypatch.setenv("PP_MLX_DISABLE", "0")
    calls: list[int] = []

    def fake_set(requested=None, *args, **kwargs):
        calls.append(int(requested))
        return requested

    fake_core = types.ModuleType("mlx.core")
    fake_core.set_wired_limit = fake_set  # type: ignore[attr-defined]
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core  # type: ignore[attr-defined]

    # Stub mlx.core BEFORE install_wired_cap imports it — never load real Metal.
    with patch.dict("sys.modules", {"mlx": fake_mlx, "mlx.core": fake_core}):
        import personality_protect.mlx_runtime as rt

        monkeypatch.setattr(rt, "assert_mlx_import_allowed", lambda: None)
        rt._CAP_INSTALLED_FOR = None
        limit = rt.install_wired_cap(16 * 10**9)
        assert limit == 16 * 10**9
        # Simulate mlx_lm.generate.wired_limit requesting ~40 GB
        fake_core.set_wired_limit(40 * 10**9)
        assert calls[-1] == 16 * 10**9
