"""Tests for eval/compare CLI and synthetic drafts."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.demo import run_demo
from personality_protect.eval_compare import (
    SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE,
    SCORECARD_MIN_NUMBERS_PER_1K_POST,
    SCORECARD_MIN_PROPER_PER_1K_ARTICLE,
    SCORECARD_MIN_PROPER_PER_1K_POST,
    count_numbers,
    list_synthetic_drafts,
    longform_metrics,
    resolve_numbers_floor,
    resolve_proper_floor,
    run_compare,
    run_eval,
    scaffolding_count,
    slop_score,
    specificity_scorecard,
)
from personality_protect.filter import FILTER_TEMPERATURE, build_filter_prompt, filter_system_prompt
from personality_protect.models import Piece
from personality_protect.sft import SYSTEM_PROMPT, piece_to_example

runner = CliRunner()


def test_synthetic_evals_packaged():
    drafts = list_synthetic_drafts()
    assert len(drafts) >= 3
    stems = {p.stem for p in drafts}
    assert "slop_branding" in stems
    assert "clean_neutral" in stems
    assert "clean_article" in stems
    article = next(p for p in drafts if p.stem == "clean_article").read_text(encoding="utf-8")
    assert 1500 <= len(article) <= 6000
    assert "Here's the problem:" in article
    assert "Here's the part that should genuinely bother" in article
    assert "contoso" in article.lower()
    assert "travelport" not in article.lower()


def test_slop_score_detects_tells():
    dirty = "In today's fast-paced world we must leverage robust synergies."
    clean = "I cut the fog and keep the spine of the argument."
    assert slop_score(dirty) >= 3
    assert slop_score(clean) == 0


def test_resolve_proper_floor_channel_p10s():
    assert resolve_proper_floor("linkedin") == SCORECARD_MIN_PROPER_PER_1K_POST
    assert resolve_proper_floor("post") == 20.0
    assert resolve_proper_floor("article") == SCORECARD_MIN_PROPER_PER_1K_ARTICLE
    assert resolve_proper_floor("article") == 42.0
    assert resolve_proper_floor(None, words=180) == 20.0
    assert resolve_proper_floor(None, words=1100) == 42.0


def test_resolve_numbers_floor_advisory_both_channels():
    assert resolve_numbers_floor("linkedin") == SCORECARD_MIN_NUMBERS_PER_1K_POST
    assert resolve_numbers_floor("linkedin") == 0.0
    assert resolve_numbers_floor("article") == SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE
    assert resolve_numbers_floor("article") == 0.0
    assert resolve_numbers_floor(None, words=180) == 0.0
    assert resolve_numbers_floor(None, words=1100) == 0.0


def test_count_numbers_evidence_not_labels_or_timeline():
    labels = (
        "Category 12 and Category 21 via Contoso Ledger rules. Cat 12/21. "
        "Version 21 schema."
    )
    assert count_numbers(labels) == 0
    timeline = "T+0. T+2 to T+5. T+5 to T+10. T+10 to T+20."
    assert count_numbers(timeline) == 0
    # Queue color without a unit noun — not evidence.
    assert count_numbers("Waiting at position 12, then position 1,100.") == 0
    evidence = (
        "75% of participants. 93.1% vs 91.1%. After 12 years. "
        "3,000 invoices cleared. $2.5m at risk."
    )
    assert count_numbers(evidence) >= 6
    # Bare calendar years / version ids are not evidence without a unit.
    assert count_numbers("Shipped Contoso Ledger in 2024. Version 21.") == 0
    # Identifiers mixed with one real claim — only the claim counts.
    mixed = "A Contoso Category 12 rule broke settlement for 180 cases."
    assert count_numbers(mixed) == 1
    # Timeline scaffolding + real claims → only the claims.
    anatomy = (
        "T+0 incident start. 64% utilization. 3,000+ invoices. "
        "T+2 to T+5. position 12. T+5 to T+10. 12 years in the role."
    )
    assert count_numbers(anatomy) == 3  # 64%, 3,000+ invoices, 12 years


def test_count_numbers_word_numerals_and_distinct_values():
    # Spelled-out counts with units are evidence.
    words = (
        "After twelve long haul flights and forty times the load, "
        "seven years of ops, five complex rebookings, two quarters, "
        "hundreds of bookings, a hundred delayed travelers, "
        "three channels, six hours."
    )
    assert count_numbers(words) >= 8
    # Repeats of one fact count once.
    assert count_numbers("3,000 invoices. Again 3,000 invoices. Still 3,000 invoices.") == 1
    # Digit and spelled form of the same claim collapse.
    assert count_numbers("Seven years. Also 7 years.") == 1
    # Bare spelled numeral without a unit still does not count.
    assert count_numbers("Twelve alone is not evidence.") == 0


def test_specificity_scorecard_gates_parable_vs_named():
    # Soft Contoso-free article: proper-noun floor is the only hard gate.
    soft = (
        "Meetings and decks aren't neutral overhead. They're how companies avoid "
        "committing. You screen for culture fit and filter out operators. "
        "Ask for a decision. Decode what it tests. Process narratives pass. "
        "The curve is years of judgment, not weeks of polish. You staff for it "
        "or you don't. Outcomes live outside the deck. Tell the truth about "
        "constraints nobody approved. Can the room kill a bad call early? "
        "Can it own the follow-through? Here is the part that matters: names "
        "and numbers are optional only when the argument stays generic. "
    ) * 4
    card = specificity_scorecard(soft, channel="article")
    assert card["pass"] is False
    assert "proper_nouns_per_1k" in card["failed"]
    assert "numbers_per_1k" not in card["failed"]
    assert card["thresholds"]["min_proper_per_1k"] == 42.0
    assert card["thresholds"]["min_numbers_per_1k"] == 0.0
    # Numbers / formatting / you>I are advisory — never hard-fail.
    assert "median_sentence" not in card["failed"]
    assert "short_line_ratio" not in card["failed"]
    assert "you_gt_i" not in card["failed"]
    assert "advisory" in card
    assert card["advisory"]["numbers_hard_gate"] is False

    # Named article with zero numbers still passes (numbers advisory).
    named_article = next(p for p in list_synthetic_drafts() if p.stem == "clean_article")
    named = specificity_scorecard(
        named_article.read_text(encoding="utf-8"), channel="article"
    )
    assert named["thresholds"]["min_numbers_per_1k"] == 0.0
    assert "numbers_per_1k" not in named["checks"]
    assert named["proper_nouns_per_1k"] >= 42.0
    assert named["pass"] is True

    # Named LinkedIn post with zero numbers still passes (numbers advisory).
    named_no_nums = (
        "Contoso Ledger, Northwind Billing, and Fabrikam Settlement make the curve years. "
        "Hotel platforms do not have this machinery. You staff for it or you don't."
    )
    nn = specificity_scorecard(named_no_nums, channel="linkedin")
    assert nn["numbers"] == 0
    assert nn["thresholds"]["min_numbers_per_1k"] == 0.0
    assert "numbers_per_1k" not in nn["checks"]
    assert nn["pass"] is True
    assert nn["proper_nouns_per_1k"] >= 20

    # First-person heavy piece still passes if names clear the post floor.
    first_person = (
        "I asked Contoso in 2024. I got 12 answers. Northwind and Fabrikam "
        "showed up in Contoso Ledger. API. SDK. I still ship the take."
    )
    fp = specificity_scorecard(first_person, channel="linkedin")
    assert fp["pass"] is True
    assert fp["thresholds"]["min_proper_per_1k"] == 20.0
    assert fp["i_count"] >= fp["you_count"]

    dense = (
        "SDK. API. CLI.\n\n"
        "Contoso said no in 2024. Northwind said wait. Fabrikam took 75%.\n\n"
        "You own the call. You ship the risk.\n\n"
        "Contoso Labs cut 12 pilots. Northwind kept GPT out of Contoso Ledger.\n\n"
        "You. Not the deck.\n"
    )
    good = specificity_scorecard(dense, channel="linkedin")
    assert good["pass"] is True
    assert good["proper_nouns_per_1k"] >= 20

    # Thin soft post fails the post floor (not only the old 48 gate).
    thin = (
        "Meetings and decks aren't neutral overhead. They're how companies "
        "avoid committing. You screen for culture fit and filter out operators."
    )
    thin_card = specificity_scorecard(thin, channel="linkedin")
    assert thin_card["pass"] is False
    assert "proper_nouns_per_1k" in thin_card["failed"]
    assert "numbers_per_1k" not in thin_card["failed"]


def test_cli_scorecard_fails_soft_article(tmp_path: Path):
    soft = tmp_path / "soft.md"
    soft.write_text(
        (
            "Meetings and decks aren't neutral overhead. They're how companies avoid "
            "committing. You screen for culture fit and filter out operators. "
            "Ask for a decision. Decode what it tests. Process narratives pass.\n\n"
        )
        * 8,
        encoding="utf-8",
    )
    res = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "scorecard",
            "--file",
            str(soft),
            "--channel",
            "article",
            "--json",
        ],
    )
    assert res.exit_code == 1, res.output
    data = json.loads(res.output)
    assert data["pass"] is False
    assert data["failed"] == ["proper_nouns_per_1k"]
    assert data["thresholds"]["min_numbers_per_1k"] == 0.0


def test_cli_scorecard_named_article_passes_with_zero_number_floor():
    article = next(p for p in list_synthetic_drafts() if p.stem == "clean_article")
    res = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "scorecard",
            "--file",
            str(article),
            "--channel",
            "article",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["pass"] is True
    assert data["thresholds"]["min_numbers_per_1k"] == 0.0
    assert "numbers_per_1k" not in data["checks"]


def test_longform_metrics_flags_near_copy_and_scaffolding():
    draft = (
        "Contoso Labs published guidance.\n\n"
        "Here's the problem: reviewers punish short punches.\n\n"
        "Here's the part that should genuinely bother anyone running these audits.\n\n"
        "Northwind keeps shipping polished indecision."
    )
    near = draft  # noop
    metrics = longform_metrics(draft, near)
    assert metrics["scaffolding_before"] >= 2
    assert metrics["scaffolding_after"] >= 2
    assert metrics["near_copy"] is True
    assert metrics["pipeline_pass"] is False

    stripped = (
        "Contoso Labs published guidance.\n\n"
        "Reviewers punish short punches.\n\n"
        "Northwind keeps shipping polished indecision.\n\n"
        "Say the tradeoff early or stay quiet."
    )
    good = longform_metrics(draft, stripped)
    assert good["scaffolding_after"] == 0
    assert good["near_copy"] is False
    assert scaffolding_count(stripped) == 0
    assert good["pipeline_pass"] is True


def test_sft_templates_stronger():
    assert "AI tells" in SYSTEM_PROMPT or "leverage" in SYSTEM_PROMPT.lower()
    p = Piece(id="x", source="demo", text="I keep the spine of the argument.", year=2023)
    ex = piece_to_example(p)
    user = ex["messages"][1]["content"]
    assert "### Rewritten" in user
    assert "cadence" in user.lower() or "voice" in user.lower()


def test_filter_prompt_stable():
    prompt = build_filter_prompt("We must leverage synergies.")
    assert "### Draft" in prompt
    assert "### Rewritten" in prompt
    # Inference prompt is stronger than train leave-alone; still shares voice goals.
    assert filter_system_prompt() != SYSTEM_PROMPT
    assert "paragraph" in filter_system_prompt().lower()
    assert FILTER_TEMPERATURE <= 0.5


def test_eval_and_compare_write_receipts(tmp_path: Path):
    run_demo(home=tmp_path)
    from personality_protect.config import get_paths

    paths = get_paths("demo", home=tmp_path)
    draft = list_synthetic_drafts()[0].read_text(encoding="utf-8")
    ev = run_eval(paths, draft, backend="mock", label="t")
    assert Path(ev["dir"]).is_dir()
    assert (Path(ev["dir"]) / "before.txt").is_file()
    assert (Path(ev["dir"]) / "after.txt").is_file()
    assert (Path(ev["dir"]) / "receipt.json").is_file()
    assert ev["slop_after"] <= ev["slop_before"]

    cmp = run_compare(paths, draft, backend="mock", label="t")
    assert (Path(cmp["dir"]) / "raw.txt").is_file()
    assert (Path(cmp["dir"]) / "prompt_baseline.txt").is_file()
    assert (Path(cmp["dir"]) / "lora.txt").is_file()
    assert cmp["slop"]["raw"] >= cmp["slop"]["lora"]


def test_cli_compare_json(tmp_path: Path):
    run_demo(home=tmp_path)
    r = runner.invoke(
        app,
        [
            "--logo", "off", "compare",
            "--home", str(tmp_path),
            "--profile", "demo",
            "--synthetic", "slop_branding",
            "--backend", "mock",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "raw" in data and "prompt_baseline" in data and "lora" in data
    assert data["slop"]["raw"] > 0
