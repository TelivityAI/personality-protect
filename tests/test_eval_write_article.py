"""Contoso-safe article eval: carve → outline brief → two arms → receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from contoso_articles import contoso_articles, contoso_post
from typer.testing import CliRunner

from personality_protect.article_holdout import (
    save_article_holdout_ids,
    select_article_holdouts,
)
from personality_protect.cli import app
from personality_protect.config import init_profile
from personality_protect.eval_write_article import (
    ARTICLE_ALPHA,
    article_word_budget,
    decide_article_voice,
    run_bare_base_article,
    run_eval_write_article,
    write_article_eval_receipt,
)
from personality_protect.eval_write_holdout import assert_receipt_contoso_safe, raw_artifacts_dir
from personality_protect.models import save_index
from personality_protect.style_profile import build_style_profile, save_style_profile
from personality_protect.voice_index import build_voice_index

runner = CliRunner()

N_ARTICLES = 12

ARTICLE_DRAFT = (
    "Contoso Ledger names one owner before anybody opens a new tier.\n"
    "\n"
    "The owner answers for the packaging in writing and keeps the record where "
    "the next team can find it without asking.\n"
    "\n"
    "Exceptions are where a published price list quietly stops describing "
    "anyone at all.\n"
    "\n"
    "Cut them weekly, in one number, and put that number in front of whoever "
    "owns the packaging."
)
BASE_DRAFT = (
    "In today's fast-paced landscape, organizations must leverage robust "
    "packaging frameworks to unlock synergies across the pricing lifecycle. "
    "Furthermore, a paradigm of continuous exception governance is a testament "
    "to operational excellence and stakeholder alignment throughout the "
    "enterprise value chain."
)


def _seed(tmp_path: Path, *, holdouts: list[str] | None = None) -> tuple:
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = [*contoso_articles(N_ARTICLES), contoso_post()]
    save_index(paths.index_path, pieces)
    carved = set(holdouts or [])
    build_voice_index(paths, holdout_ids=carved)
    save_style_profile(paths, build_style_profile(pieces))
    return paths, pieces


def _carve(tmp_path: Path) -> tuple:
    paths, pieces = _seed(tmp_path)
    receipt = select_article_holdouts(pieces)
    paths, _ = _seed(tmp_path, holdouts=receipt["holdout_ids"])
    save_article_holdout_ids(paths, receipt)
    return paths, receipt["holdout_ids"]


def _arms(article: str = ARTICLE_DRAFT, base: str = BASE_DRAFT):
    """Two generators that answer with fixed drafts, per arm."""
    return (lambda _m, **_k: article), (lambda _m, **_k: base)


def test_word_budget_is_shared_by_both_arms(tmp_path: Path):
    paths, _ = _seed(tmp_path)
    budget = article_word_budget(paths, "Contoso packaging", "- One\n- Two\n- Three")
    assert budget["section_count"] == 3
    assert budget["sections"] == ["One", "Two", "Three"]
    assert budget["word_ceiling"] == budget["section_trim_words"] * 3
    assert budget["section_trim_words"] > budget["section_words"]
    assert budget["max_tokens"] >= budget["section_trim_words"]


def test_control_arm_writes_an_article_without_the_voice_machinery(tmp_path: Path):
    """The control must be a bare article, not a bare post."""
    paths, _ = _seed(tmp_path)
    budget = article_word_budget(paths, "Contoso packaging", "- One\n- Two\n- Three")
    seen: list[str] = []

    def fake_generate(messages, **_kwargs: object) -> str:
        seen.append(messages[1]["content"])
        # Stay inside the brief — Contoso is allowed; invented vendors are not.
        return f"Contoso packaging section {len(seen)} covers the one claim."

    result = run_bare_base_article(
        "Contoso packaging",
        "- One\n- Two\n- Three",
        budget=budget,
        generate_fn=fake_generate,
        base_model="contoso-local",
    )
    assert result["mode"] == "bare_base_article"
    assert result["section_count"] == 3
    assert len(seen) >= 3
    # Same structure as the product arm...
    assert "section 1 of 3" in seen[0]
    assert f"Aim for about {budget['section_words']} words" in seen[0]
    assert "ALLOWED names from the BRIEF only" in seen[0]
    # ...and none of the voice machinery.
    assert "EXAMPLES" not in seen[0]
    assert "Sentence length varies" not in seen[0]
    assert "Never use these words" not in seen[0]


def test_control_arm_repairs_and_scrubs_like_the_product_arm(tmp_path: Path):
    """A fact-lock only one arm has to survive would decide the comparison."""
    paths, _ = _seed(tmp_path)
    budget = article_word_budget(paths, "Contoso packaging", "- One\n- Two\n- Three")
    seen: list[str] = []

    def fake_generate(messages, **_kwargs: object) -> str:
        prompt = messages[1]["content"]
        seen.append(prompt)
        if "REPAIR:" not in prompt:
            return "The packaging work ran at Fabrikam Northwind for nine weeks."
        return "Contoso packaging keeps one owner and one published list."

    result = run_bare_base_article(
        "Contoso packaging",
        "- One\n- Two\n- Three",
        budget=budget,
        generate_fn=fake_generate,
        base_model="contoso-local",
    )
    assert result["section_count"] == 3
    assert len(result["repaired_sections"]) == 3
    assert result["dropped_sections"] == []
    assert "Fabrikam" not in result["text"]
    repairs = [prompt for prompt in seen if "REPAIR:" in prompt]
    assert len(repairs) == 3
    assert "northwind" in repairs[0].lower()
    # Still no voice machinery, repair or not.
    assert "EXAMPLES" not in repairs[0]


def test_control_arm_that_cannot_be_repaired_still_disqualifies(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    invented = "The rollout ran at Fabrikam Northwind across two Tailspin Toys sites."
    receipt = run_eval_write_article(
        paths,
        [holdouts[0]],
        k=1,
        generate_fn=lambda _m, **_k: ARTICLE_DRAFT,
        generate_fn_base=lambda _m, **_k: invented,
    )
    item = receipt["items"][0]
    assert item["base_dropped_sections"] == item["section_count"]
    assert item["base_invent_reject"]
    assert item["base_disqualified"]
    assert item["winner"] != "base"


def test_both_arms_see_the_same_outline_and_budget(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    article_prompts: list[str] = []
    base_prompts: list[str] = []

    def article_fn(messages, **_kwargs: object) -> str:
        article_prompts.append(messages[1]["content"])
        return ARTICLE_DRAFT

    def base_fn(messages, **_kwargs: object) -> str:
        base_prompts.append(messages[1]["content"])
        return BASE_DRAFT

    receipt = run_eval_write_article(
        paths, holdouts[:1], k=1, generate_fn=article_fn, generate_fn_base=base_fn
    )
    assert len(base_prompts) == receipt["items"][0]["section_count"]
    assert "Never use these words" in article_prompts[0]
    assert "Never use these words" not in base_prompts[0]
    assert "EXAMPLES" not in base_prompts[0]
    for line in ("section 1 of", "Aim for about", "Write only the section about:"):
        assert line in article_prompts[0]
        assert line in base_prompts[0]


def test_eval_scores_every_holdout_and_reports_a_verdict(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    article_fn, base_fn = _arms()

    receipt = run_eval_write_article(
        paths, holdouts, k=2, generate_fn=article_fn, generate_fn_base=base_fn
    )
    assert receipt["kind"] == "eval_write_article"
    assert receipt["channel"] == "article"
    assert receipt["n_holdouts"] == len(holdouts)
    assert receipt["carve"]["ok"]
    assert sum(receipt["wins"].values()) == len(holdouts)
    assert receipt["verdict"] in {"voice_supported", "not_supported"}
    assert receipt["alpha"] == ARTICLE_ALPHA


def test_receipt_never_carries_draft_or_holdout_bodies(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    article_fn, base_fn = _arms()
    receipt = run_eval_write_article(
        paths, holdouts, k=1, generate_fn=article_fn, generate_fn_base=base_fn
    )
    assert_receipt_contoso_safe(receipt)
    blob = json.dumps(receipt)
    assert "Contoso Ledger names one owner" not in blob
    assert "leverage" not in blob


def test_eval_refuses_to_run_when_a_holdout_is_indexed(tmp_path: Path):
    paths, pieces = _seed(tmp_path)
    leaked = pieces[0].id
    article_fn, base_fn = _arms()
    with pytest.raises(ValueError, match="retrieval leak"):
        run_eval_write_article(
            paths, [leaked], generate_fn=article_fn, generate_fn_base=base_fn
        )


def test_unbriefable_holdouts_are_skipped_not_scored(tmp_path: Path):
    paths, pieces = _seed(tmp_path)
    stub = pieces[0]
    stub.text = "Contoso ships. You own it."
    save_index(paths.index_path, pieces)
    build_voice_index(paths, holdout_ids={stub.id})
    article_fn, base_fn = _arms()

    receipt = run_eval_write_article(
        paths, [stub.id], generate_fn=article_fn, generate_fn_base=base_fn
    )
    assert receipt["skipped_unbriefable"] == [stub.id]
    assert receipt["n_holdouts"] == 0
    assert receipt["items"] == []


def test_a_draft_that_hands_the_brief_back_cannot_win(tmp_path: Path):
    """Echoing the mined bullets scores flattering rhythm and writes nothing."""
    paths, holdouts = _carve(tmp_path)
    from personality_protect.article_brief import mine_article_brief
    from personality_protect.corpus_text import normalize_corpus_text
    from personality_protect.models import load_index

    by_id = {piece.id: piece for piece in load_index(paths.index_path)}
    brief, _ = mine_article_brief(normalize_corpus_text(by_id[holdouts[0]].text))
    echo = f"{brief['topic']}\n\n{brief['points']}"

    receipt = run_eval_write_article(
        paths,
        [holdouts[0]],
        k=1,
        generate_fn=lambda _m, **_k: echo,
        generate_fn_base=lambda _m, **_k: BASE_DRAFT,
    )
    item = receipt["items"][0]
    assert item["article_disqualified"]
    assert item["winner"] != "article"


def test_invented_entities_disqualify_the_article_arm(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    invented = (
        "Fabrikam Northwind shipped the Tailspin Toys packaging with Wingtip "
        "Partners and Adventure Works oversight across nine regions.\n"
        "\n"
        "Litware Proseware also confirmed the Woodgrove Bank exception "
        "programme before the review closed."
    )
    receipt = run_eval_write_article(
        paths,
        [holdouts[0]],
        k=1,
        generate_fn=lambda _m, **_k: invented,
        generate_fn_base=lambda _m, **_k: BASE_DRAFT,
    )
    item = receipt["items"][0]
    # Inventing sections are dropped from the stitch; the arm still DQs.
    assert item["article_disqualified"]
    assert item["article_invent_reject"] or item["article_invented_entities_count"] > 0
    assert item["winner"] != "article"


def test_both_arms_are_held_to_the_same_length_ceiling(tmp_path: Path):
    """A trim applied to one side only would decide the comparison by itself."""
    paths, holdouts = _carve(tmp_path)
    runaway = "\n\n".join(
        f"Contoso Ledger paragraph {n} names one owner and cuts one exception."
        for n in range(400)
    )
    receipt = run_eval_write_article(
        paths,
        [holdouts[0]],
        k=1,
        generate_fn=lambda _m, **_k: runaway,
        generate_fn_base=lambda _m, **_k: runaway,
    )
    item = receipt["items"][0]
    assert item["article_draft_words"] <= item["word_ceiling"]
    assert item["base_draft_words"] <= item["word_ceiling"]


def test_verdict_requires_a_margin_a_coin_would_not_produce():
    swept = decide_article_voice(
        {"article": 4, "base": 0, "tie": 0}, article_disqualified=0, base_disqualified=0
    )
    assert swept["verdict"] == "voice_supported"

    narrow = decide_article_voice(
        {"article": 2, "base": 1, "tie": 0}, article_disqualified=0, base_disqualified=0
    )
    assert narrow["verdict"] == "not_supported"
    assert "margin_within_chance" in narrow["blocking_reasons"]

    fabricating = decide_article_voice(
        {"article": 4, "base": 0, "tie": 0}, article_disqualified=2, base_disqualified=0
    )
    assert fabricating["verdict"] == "not_supported"
    assert "article_disqualified_more_often" in fabricating["blocking_reasons"]


def test_a_run_that_never_compared_cadence_says_so():
    """Losing on distance and never reaching distance are different failures.

    Both arms disqualified scores a tie, so an all-disqualified run and a run
    the voice arm genuinely lost both report zero wins. Only one of them is
    evidence about voice, and the receipt has to name which.
    """
    stalled = decide_article_voice(
        {"article": 0, "base": 0, "tie": 4},
        article_disqualified=4,
        base_disqualified=4,
        both_disqualified=4,
        n_items=4,
    )
    assert stalled["verdict"] == "not_supported"
    assert stalled["distance_ever_decided"] is False
    assert "every_item_disqualified_in_both_arms" in stalled["blocking_reasons"]

    measured = decide_article_voice(
        {"article": 1, "base": 3, "tie": 0},
        article_disqualified=0,
        base_disqualified=0,
        both_disqualified=0,
        n_items=4,
    )
    assert measured["verdict"] == "not_supported"
    assert measured["distance_ever_decided"] is True
    assert "every_item_disqualified_in_both_arms" not in measured["blocking_reasons"]


def test_raw_artifacts_stay_out_of_the_receipt(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    article_fn, base_fn = _arms()
    receipt = run_eval_write_article(
        paths,
        holdouts[:1],
        k=1,
        generate_fn=article_fn,
        generate_fn_base=base_fn,
        save_raw=True,
    )
    written = sorted(path.name for path in raw_artifacts_dir(paths).glob("*"))
    assert any(name.endswith(".article.draft.txt") for name in written)
    assert any(name.endswith(".bare_base.draft.txt") for name in written)
    assert str(raw_artifacts_dir(paths)) not in json.dumps(receipt)


def test_receipt_file_round_trips(tmp_path: Path):
    paths, holdouts = _carve(tmp_path)
    article_fn, base_fn = _arms()
    receipt = run_eval_write_article(
        paths, holdouts[:1], k=1, generate_fn=article_fn, generate_fn_base=base_fn
    )
    out = tmp_path / "evals" / "article.json"
    write_article_eval_receipt(receipt, out)
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == "eval_write_article"


def test_cli_requires_a_carve_before_it_will_run(tmp_path: Path):
    _seed(tmp_path)
    result = runner.invoke(
        app,
        ["eval-write-article", "--profile", "contoso", "--home", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "select-article-holdouts" in result.output
