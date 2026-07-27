"""Filter must return a clean rewrite, not SFT template echo; MLX must stay capped."""

from __future__ import annotations

from unittest.mock import patch

from personality_protect.filter import (
    FILTER_SYSTEM_PROMPT,
    FILTER_USER_TEMPLATE_INFER,
    build_filter_messages,
    build_filter_user_content,
    extract_rewrite,
    finalize_rewrite,
    rewrite_quality_flags,
    similarity_guard,
    strip_ai_tells,
    suggest_max_tokens,
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
    # Inference shape (stronger than train leave-alone) still ends at ### Rewritten
    assert user == FILTER_USER_TEMPLATE_INFER.format(draft="Draft with leverage.")
    assert "cadence" in user.lower()


def test_prompts_preserve_paragraphs_and_allow_leave_alone():
    """Training prompt keeps leave-alone; filter prompt still allows it sparingly."""
    sys_l = SYSTEM_PROMPT.lower()
    assert "paragraph" in sys_l
    assert "unchanged" in sys_l
    assert "opener" in sys_l or "openings" in sys_l
    # Flat/slop must be rewritten into multi-para cadence, not a thesaurus line.
    assert "thesaurus" in sys_l or "multi-paragraph" in sys_l or "blank lines" in sys_l
    user = USER_TEMPLATE_INFER.format(draft="Companies need a clear point of view.")
    user_l = user.lower()
    assert "paragraph" in user_l
    assert "unchanged" in user_l
    assert "opener" in user_l or "openings" in user_l
    assert "multi-paragraph" in user_l or "thesaurus" in user_l or "blank lines" in user_l


def test_filter_prompt_pushes_past_polished_generic():
    """Inference must not treat clean Claude drafts as automatic leave-alone."""
    sys_l = FILTER_SYSTEM_PROMPT.lower()
    assert "polished" in sys_l or "frontier" in sys_l or "generic" in sys_l
    assert "truncate" in sys_l or "full draft" in sys_l
    assert "header" in sys_l
    user_l = FILTER_USER_TEMPLATE_INFER.lower()
    assert "throat-clearing" in user_l or "only if" in user_l
    assert "whole draft" in user_l or "mid-piece" in user_l


def test_force_prompt_forbids_leave_alone():
    from personality_protect.filter import (
        FILTER_SYSTEM_PROMPT_FORCE,
        FILTER_USER_TEMPLATE_FORCE,
        build_filter_messages,
    )

    assert "always rewrite" in FILTER_SYSTEM_PROMPT_FORCE.lower()
    assert "do not copy" in FILTER_USER_TEMPLATE_FORCE.lower()
    msgs = build_filter_messages("Clean Claude draft.", force=True)
    assert msgs[0]["content"] == FILTER_SYSTEM_PROMPT_FORCE
    assert "ALWAYS rewrite" in msgs[1]["content"]
    assert "Clean Claude draft." in msgs[1]["content"]


def test_suggest_max_tokens_scales_for_articles():
    short = "Short post.\n\nSecond beat."
    assert suggest_max_tokens(short) == 512
    article = "x" * 7000  # ~1k-word article size
    budget = suggest_max_tokens(article)
    assert budget >= 2500
    assert budget <= 4096
    assert suggest_max_tokens(article, override=900) == 900
    assert suggest_max_tokens(article, override=99999) == 4096


def test_rewrite_quality_flags_detect_noop_and_truncation():
    draft = "One.\n\nTwo.\n\nThree complete sentences here."
    assert rewrite_quality_flags(draft, draft)["unchanged"] is True
    cut = "One.\n\nTwo mid"
    flags = rewrite_quality_flags(draft * 20, cut)
    assert flags["likely_truncated"] is True


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
    assert "\n\n" in out
    assert "open in" not in out.lower()
    assert "solid strengths" not in out.lower()
    assert "leverage" not in out.lower()
    assert "nestled" not in out.lower()
    assert "testament" not in out.lower()
    assert "branding" in out.lower()


def test_similarity_guard_returns_draft_when_near_identity():
    draft = (
        "These questions keep popping up every time I read another take.\n\n"
        "How much is the first pass, and how much is cleanup?"
    )
    fidget = (
        "These questions keep coming up every time I read another take.\n\n"
        "How much is the first pass, and how much is cleanup?"
    )
    assert similarity_guard(draft, fidget) == draft
    different = "I cut the fog and keep the branding honest."
    assert similarity_guard(draft, different) == different
    # Flat single-block drafts must still be allowed to rewrite (clean→voice).
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


def test_filter_messages_default_omits_long_reference():
    """LoRA path: cadence via weights, not pasted corpus blocks."""
    messages = build_filter_messages("We must leverage synergies.")
    assert messages[0] == {"role": "system", "content": FILTER_SYSTEM_PROMPT}
    assert "### My voice" not in messages[1]["content"]
    assert messages[1]["content"] == FILTER_USER_TEMPLATE_INFER.format(
        draft="We must leverage synergies."
    )


def test_filter_messages_are_chat_roles():
    messages = build_filter_messages(
        "We must leverage synergies.",
        reference="I keep it plain.",
    )
    assert messages[0] == {"role": "system", "content": FILTER_SYSTEM_PROMPT}
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
