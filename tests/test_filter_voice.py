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


def test_filter_prompt_pushes_past_polished_generic():
    """Inference must not treat clean polished drafts as automatic leave-alone."""
    sys_l = FILTER_SYSTEM_PROMPT.lower()
    assert "polished" in sys_l or "frontier" in sys_l or "generic" in sys_l
    assert "truncate" in sys_l or "full draft" in sys_l
    assert "header" in sys_l
    user_l = FILTER_USER_TEMPLATE_INFER.lower()
    assert "throat-clearing" in user_l or "only if" in user_l
    assert "whole draft" in user_l or "mid-piece" in user_l


def test_force_prompt_keeps_substance_over_delete():
    from personality_protect.filter import (
        FILTER_SYSTEM_PROMPT_FORCE,
        FILTER_USER_TEMPLATE_FORCE,
        build_filter_messages,
    )

    sys_l = FILTER_SYSTEM_PROMPT_FORCE.lower()
    assert "never delete" in sys_l or "worse than" in sys_l
    assert "deterministic" in sys_l
    assert "parallel" in sys_l or "blank lines" in sys_l
    user_l = FILTER_USER_TEMPLATE_FORCE.lower()
    assert "load-bearing" in user_l or "argument" in user_l
    msgs = build_filter_messages("Clean polished draft.", force=True)
    assert msgs[0]["content"] == FILTER_SYSTEM_PROMPT_FORCE
    assert "Clean polished draft." in msgs[1]["content"]


def test_strip_voice_scaffolding_cuts_known_patterns():
    from personality_protect.filter import strip_voice_scaffolding

    text = (
        "14 meetings. 6 weeks. 0 decisions.\n\n"
        "Here's what those 6 weeks actually looked like:\n\n"
        "Eight people in every session.\n\n"
        "Nobody was aligning on quality. That's the part worth being honest about.\n\n"
        "The loop existed to protect two things.\n\n"
        "That's the part people miss about disagree and commit."
    )
    out = strip_voice_scaffolding(text)
    assert "Here's what" not in out
    assert "worth being honest" not in out
    assert "That's what people miss" in out
    assert "Eight people in every session." in out
    # Admission and analysis stay separate paragraphs (not welded).
    assert "quality.The" not in out
    assert "quality.\n\nThe loop" in out


def test_force_echo_must_not_write_voiced_file(tmp_path):
    """Under --force, byte-identical rewrite is a hard fail (no fake voiced file)."""
    from typer.testing import CliRunner

    from personality_protect.cli import app

    home = str(tmp_path)
    runner = CliRunner()
    assert (
        runner.invoke(
            app, ["--logo", "off", "init", "--home", home, "--json"]
        ).exit_code
        == 0
    )
    draft = tmp_path / "draft.md"
    draft.write_text("Plain polished paragraph with no voice punch.\n", encoding="utf-8")
    out = tmp_path / "voiced.md"

    def _echo(draft_text, _paths, **kwargs):
        return draft_text.strip(), "mock"

    with patch("personality_protect.cli.filter_draft", side_effect=_echo):
        result = runner.invoke(
            app,
            [
                "--logo",
                "off",
                "filter",
                "--file",
                str(draft),
                "--out",
                str(out),
                "--force",
                "--home",
                home,
                "--json",
            ],
        )
    assert result.exit_code == 1, result.output
    assert not out.is_file()
    assert "force_echo_reject" in result.stdout


def test_strip_voice_scaffolding_cuts_article_throat_clearers():
    from personality_protect.filter import strip_voice_scaffolding

    text = (
        "Contoso audits prose texture.\n\n"
        "Here's the problem: the people most likely to fail are not low-effort.\n\n"
        "Northwind keeps shipping.\n\n"
        "Here's the part that should genuinely bother anyone running these audits.\n\n"
        "The hours of circling the problem."
    )
    out = strip_voice_scaffolding(text)
    assert "Here's the problem:" not in out
    assert "Here's the part that should genuinely bother" not in out
    assert "people most likely to fail" in out
    assert "hours of circling" in out


def test_strip_keeps_load_bearing_didnt_see_hinge():
    from personality_protect.filter import strip_voice_scaffolding

    text = (
        '"This is clearly ChatGPT slop."\n\n'
        "The comment appeared under a message.\n\n"
        "Here's what the commenter didn't see:\n\n"
        "The hours of circling the problem."
    )
    out = strip_voice_scaffolding(text)
    assert "Here's what the commenter didn't see:" in out
    assert "This is clearly ChatGPT slop." in out
    assert "hours of circling" in out


def test_restore_structural_openers_puts_back_quote_and_hinge():
    from personality_protect.filter import apply_voice_postprocess

    draft = (
        '"This is clearly ChatGPT slop."\n\n'
        "The comment appeared under a message.\n\n"
        "Here's what the commenter didn't see:\n\n"
        "The hours of circling the problem."
    )
    # Model dropped the premise quote and the hinge.
    model_out = (
        "The comment appeared under a message.\n\n"
        "The hours of circling the problem."
    )
    out = apply_voice_postprocess(model_out, draft=draft)
    assert out.startswith('"This is clearly ChatGPT slop."')
    assert "Here's what the commenter didn't see:" in out
    assert "hours of circling" in out


def test_substance_guard_rejects_catastrophic_drops_only():
    from personality_protect.filter import substance_guard

    draft = (
        "Nothing about the analysis was wrong. Nothing about it was even questioned.\n\n"
        "What got audited was the texture of the prose.\n\n"
        "In a previous piece, I argued that compressed thinking gets misread as slop."
    )
    skinny = "What got audited was the texture of the prose."
    assert substance_guard(draft, skinny) == draft
    # Reparagraphing alone must not revert (overcorrection).
    split = (
        "Nothing about the analysis was wrong. Nothing about it was even questioned.\n\n"
        "What got audited was the texture of the prose.\n\n"
        "In a previous piece, I argued that compressed thinking gets misread as slop.\n\n"
        "Extra beat."
    )
    assert substance_guard(draft, split) == split


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
        "Unlocking nestled opportunities is a testament to vibrant innovation.\n\n"
        "Keep this real beat about the brand."
    )
    out = strip_ai_tells(text)
    assert "\n\n" in out
    assert "open in" not in out.lower()
    assert "solid strengths" not in out.lower()
    assert "leverage" not in out.lower()
    assert "nestled" not in out.lower()
    assert "testament" not in out.lower()
    assert "branding" in out.lower() or "brand" in out.lower()
    # Whole mush clause deleted — not "find real opportunities is innovation."
    assert "is innovation" not in out.lower()
    assert "Keep this real beat" in out


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


def test_filter_draft_leave_alone_skips_model_for_vibe(tmp_path):
    """Already-in-voice short drafts must not fidget or mode-collapse."""
    from personality_protect.config import get_paths
    from personality_protect.demo import run_demo
    from personality_protect.filter import filter_draft

    run_demo(home=tmp_path)
    paths = get_paths("demo", home=tmp_path)
    draft = (
        "These questions keep popping up every time I read another vibe coding take.\n\n"
        "How much of the labs' revenue is the first pass, and how much is cleanup?\n\n"
        "Maybe we are too hard on vibe coders. Who's saying the model finishes?"
    )
    calls: list[str] = []

    def boom(d: str, *_a, **_k):
        calls.append(d)
        raise AssertionError("model must not run for leave-alone voice drafts")

    with patch("personality_protect.filter._filter_mock", side_effect=boom):
        out, used = filter_draft(draft, paths, backend="mock")
    assert used == "mock"
    assert calls == []
    assert out == draft.strip()
    # --force still runs the model, but invented vocabulary is rejected.
    with patch(
        "personality_protect.filter._filter_mock",
        side_effect=lambda d, *_a, **_k: "Cut the fog.\n\nKeep the spine.",
    ):
        forced, _ = filter_draft(draft, paths, backend="mock", force=True)
    assert "Cut the fog" not in forced
    assert "spine" not in forced
    # Subtractive force rewrite (no new vocabulary) is kept.
    with patch(
        "personality_protect.filter._filter_mock",
        side_effect=lambda d, *_a, **_k: d.replace("Maybe we are too hard", "We are hard"),
    ):
        forced2, _ = filter_draft(draft, paths, backend="mock", force=True)
    assert "Maybe we are too hard" not in forced2
    assert "We are hard on vibe coders" in forced2


def test_draft_already_in_voice_rejects_polished_long_claude():
    from personality_protect.filter import draft_already_in_voice

    # Long multipara with ?, contractions, and I — not user cadence by itself.
    paras = []
    for i in range(20):
        paras.append(
            f"I'm noticing that Contoso's audit {i} doesn't capture substance. "
            f"Why does Northwind still treat texture as the verdict?"
        )
    long_polished = "\n\n".join(paras)
    assert len(long_polished) > 1600
    assert not draft_already_in_voice(long_polished)


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


def test_paragraph_windows_packs_under_budget_and_preserves_short():
    from personality_protect.filter import (
        FILTER_CHUNK_MAX_CHARS,
        paragraph_windows,
        stitch_filter_chunks,
    )

    short = "Contoso needs a distinct voice, not a template."
    assert paragraph_windows(short) == [short]

    paras = [
        "Contoso Labs ships another memo on workplace communication quality.",
        "Here's the problem: auditors punish compressed thinking as low effort.",
        "Northwind Analytics scores warmth and brand alignment like substance.",
        "Here's the part that should genuinely bother anyone running these audits.",
        "If Contoso keeps scoring texture, it trains polished indecision.",
    ]
    # Inflate so packing needs multiple windows under the infer budget.
    long = "\n\n".join(p + (" Extra clause." * 8) for p in paras)
    assert len(long) > FILTER_CHUNK_MAX_CHARS * 2
    windows = paragraph_windows(long, max_chars=FILTER_CHUNK_MAX_CHARS)
    assert len(windows) >= 2
    assert all(len(w) <= FILTER_CHUNK_MAX_CHARS + 5 for w in windows)
    # Stitch restores blank-line paragraph rhythm.
    stitched = stitch_filter_chunks(windows)
    assert "\n\n" in stitched
    assert "Contoso Labs" in stitched
    assert "polished indecision" in stitched


def test_should_chunk_filter_skips_short_probes():
    from personality_protect.filter import FILTER_CHUNK_THRESHOLD, should_chunk_filter

    assert FILTER_CHUNK_THRESHOLD >= 1600
    assert not should_chunk_filter("x" * 670)  # slop_multipara-sized
    assert not should_chunk_filter("x" * 1283)  # pending_vibe-sized
    assert should_chunk_filter("x" * 1601)
    assert should_chunk_filter("y" * 5000)


def test_filter_draft_short_stays_singleshot(tmp_path):
    """Short drafts must not enter the chunked path (probe grades depend on it)."""
    from personality_protect.config import get_paths
    from personality_protect.demo import run_demo
    from personality_protect.filter import filter_draft

    run_demo(home=tmp_path)
    paths = get_paths("demo", home=tmp_path)
    draft = (
        "In today's fast-paced digital world, Contoso must leverage robust synergies.\n\n"
        "Furthermore, leaders should delve into authentic storytelling."
    )
    calls: list[str] = []

    def fake_mock(d: str, _adapter_dir):
        calls.append(d)
        # Subtractive only — no invented vocabulary.
        return (
            d.replace("In today's fast-paced digital world, ", "")
            .replace("leverage ", "")
            .replace("Furthermore, ", "")
            .replace("delve into ", "")
        )

    with patch("personality_protect.filter._filter_mock", side_effect=fake_mock):
        out, used = filter_draft(draft, paths, backend="mock")
    assert used == "mock"
    assert len(calls) == 1
    assert calls[0] == draft.strip()
    assert "leverage" not in out.lower()
    assert "delve" not in out.lower()


def test_novelty_guard_rejects_invented_vocabulary():
    from personality_protect.filter import (
        introduced_vocabulary,
        novelty_guard,
        novelty_too_high,
        strip_ai_tells,
    )

    draft = (
        "Contoso audits punish compressed thinking as low effort.\n\n"
        "Northwind scores warmth like substance."
    )
    generative = (
        draft
        + "\n\nWrite with some spine instead of packaging your thoughts in bandages.\n"
        "Whoeres in the fog climb ladders toward the chatbot mirror."
    )
    new = introduced_vocabulary(draft, generative)
    assert "bandages" in new
    assert "whoeres" in new
    assert novelty_too_high(draft, generative) is True
    assert novelty_guard(draft, generative) == draft.strip()

    # Deterministic AI-tell synonym swaps are subtractive cleanup, not generation.
    slop = "In today's fast-paced world we must leverage synergies."
    cleaned = strip_ai_tells(slop)
    assert "leverage" not in cleaned.lower()
    assert novelty_too_high(slop, cleaned) is False
    assert introduced_vocabulary(slop, cleaned) == []


def test_strip_capitalizes_orphan_after_problem_cut():
    from personality_protect.filter import strip_voice_scaffolding

    text = (
        "Here's the problem: the people most likely to fail are not low-effort.\n\n"
        "Northwind keeps the rest."
    )
    out = strip_voice_scaffolding(text)
    assert "Here's the problem:" not in out
    assert out.startswith("The people most likely")


def test_filter_draft_long_auto_chunks_strips_scaffolding(tmp_path):
    """Article-length drafts auto-chunk; generative novelty falls back to strip-only."""
    from personality_protect.config import get_paths
    from personality_protect.demo import run_demo
    from personality_protect.eval_compare import longform_metrics
    from personality_protect.filter import FILTER_CHUNK_MAX_CHARS, filter_draft

    run_demo(home=tmp_path)
    paths = get_paths("demo", home=tmp_path)

    body = []
    for i in range(12):
        body.append(
            f"Contoso Labs memo section {i} argues that Northwind must evaluate "
            f"texture as carefully as substance when reviewing workplace writing."
        )
    draft = (
        "Here's the problem: Contoso audits punish compressed thinking.\n\n"
        + "\n\n".join(body)
        + "\n\nHere's the part that should genuinely bother anyone running these audits.\n\n"
        "Northwind will keep publishing polished indecision at greater volume."
    )
    assert len(draft) > 1600

    calls: list[str] = []

    def fake_mock(d: str, _adapter_dir):
        calls.append(d)
        # Invented vocabulary — must be rejected by novelty_guard.
        return d + "\n\nCut the fog with bandages and chatbots."

    with patch("personality_protect.filter._filter_mock", side_effect=fake_mock):
        out, used = filter_draft(draft, paths, backend="mock", force=True)

    assert used == "mock"
    assert len(calls) >= 2
    assert all(len(c) <= FILTER_CHUNK_MAX_CHARS + 40 for c in calls)
    assert "Here's the problem" not in out
    assert "Here's the part that should genuinely bother" not in out
    assert "bandages" not in out
    assert "chatbots" not in out
    assert out.startswith("Contoso audits") or "Contoso audits" in out
    metrics = longform_metrics(draft, out)
    assert metrics["scaffolding_after"] == 0
    assert "\n\n" in out


def test_filter_draft_short_leave_alone_still_single_shot(tmp_path):
    """Short already-voice drafts stay leave-alone; chunk reject/retry must not fire."""
    from personality_protect.config import get_paths
    from personality_protect.demo import run_demo
    from personality_protect.filter import filter_draft

    run_demo(home=tmp_path)
    paths = get_paths("demo", home=tmp_path)
    draft = (
        "These questions keep popping up every time Contoso ships another take.\n\n"
        "How much of the lab revenue is the first pass?\n\n"
        "How much is the next forty prompts cleaning up what that first pass produced?\n\n"
        "Who owns the mess when the demo looks clean and the codebase does not?\n\n"
        "What happens when the manager wants polish and the engineer wants signal?\n\n"
        "Where does the bill land when everyone calls it productivity?\n\n"
        "And why does the write-up always sound finished before the thinking does?"
    )
    calls: list[str] = []

    def fake_mock(d: str, _adapter_dir):
        calls.append(d)
        return d + "\n\n(should not run)"

    with patch("personality_protect.filter._filter_mock", side_effect=fake_mock):
        out, used = filter_draft(draft, paths, backend="mock")

    assert used == "mock"
    assert calls == []  # leave-alone before model
    assert out.strip() == draft.strip()
