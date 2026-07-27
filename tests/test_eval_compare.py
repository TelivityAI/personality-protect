"""Tests for eval/compare CLI and synthetic drafts."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.demo import run_demo
from personality_protect.eval_compare import (
    list_synthetic_drafts,
    run_compare,
    run_eval,
    slop_score,
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


def test_slop_score_detects_tells():
    dirty = "In today's fast-paced world we must leverage robust synergies."
    clean = "I cut the fog and keep the spine of the argument."
    assert slop_score(dirty) >= 3
    assert slop_score(clean) == 0


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
