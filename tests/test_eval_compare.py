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


def test_resolve_numbers_floor_posts_advisory():
    assert resolve_numbers_floor("linkedin") == SCORECARD_MIN_NUMBERS_PER_1K_POST
    assert resolve_numbers_floor("linkedin") == 0.0
    assert resolve_numbers_floor("article") == SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE
    assert resolve_numbers_floor("article") == 6.6
    assert resolve_numbers_floor(None, words=180) == 0.0
    assert resolve_numbers_floor(None, words=1100) == 6.6


def test_count_numbers_evidence_not_labels_or_timeline():
    labels = (
        "Category 15 and Category 35 via Contoso fare rules. Cat 15/35. "
        "Version 21 schema."
    )
    assert count_numbers(labels) == 0
    timeline = "T+0. T+5 to T+15. T+15 to T+30. T+30 to T+45. T+45 to T+60."
    assert count_numbers(timeline) == 0
    # Queue color without a unit noun — not evidence.
    assert count_numbers("Waiting at position 340, then position 1,100.") == 0
    evidence = (
        "75% of participants. 93.1% vs 91.1%. After 18 years. "
        "3,000 seats in 2024. $2.5m at risk."
    )
    assert count_numbers(evidence) >= 6
    # Identifiers mixed with one real claim — only the claim counts.
    mixed = "A Category 35 rule broke settlement for 180 seats."
    assert count_numbers(mixed) == 1
    # Timeline scaffolding + real claims → only the claims.
    anatomy = (
        "T+0 incident start. 81% utilization. 3,000+ seats. "
        "T+15 to T+30. position 340. T+45 to T+60. 18 years in the role."
    )
    assert count_numbers(anatomy) == 3  # 81%, 3,000+ seats, 18 years


def test_specificity_scorecard_gates_parable_vs_named():
    parable = next(p for p in list_synthetic_drafts() if p.stem == "clean_article")
    card = specificity_scorecard(parable.read_text(encoding="utf-8"), channel="article")
    assert card["pass"] is False
    assert "numbers_per_1k" in card["failed"]
    assert card["thresholds"]["min_proper_per_1k"] == 42.0
    assert card["thresholds"]["min_numbers_per_1k"] == 6.6
    # Formatting / you>I are advisory — never hard-fail.
    assert "median_sentence" not in card["failed"]
    assert "short_line_ratio" not in card["failed"]
    assert "you_gt_i" not in card["failed"]
    assert "advisory" in card

    # Named LinkedIn post with zero numbers still passes (numbers advisory).
    named_no_nums = (
        "PNR state, ATPCO filings, and GDS settlement make air's curve years. "
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
        "showed up in the PNR. GDS. NDC. API. I still ship the take."
    )
    fp = specificity_scorecard(first_person, channel="linkedin")
    assert fp["pass"] is True
    assert fp["thresholds"]["min_proper_per_1k"] == 20.0
    assert fp["i_count"] >= fp["you_count"]

    dense = (
        "GDS. NDC. API.\n\n"
        "Contoso said no in 2024. Northwind said wait. Fabrikam took 75%.\n\n"
        "You own the call. You ship the risk.\n\n"
        "Contoso Labs cut 12 pilots. Northwind kept GPT out of the PNR.\n\n"
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


def test_cli_scorecard_fails_parable(tmp_path: Path):
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
    assert res.exit_code == 1, res.output
    data = json.loads(res.output)
    assert data["pass"] is False
    assert set(data["failed"]) <= {"proper_nouns_per_1k", "numbers_per_1k"}
    assert data["thresholds"]["min_numbers_per_1k"] == 6.6


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
