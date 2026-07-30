"""Contoso-safe Lane G: holdout carve → briefs → RAG vs bare-base → receipt."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import init_profile
from personality_protect.eval_write_holdout import (
    assert_brief_is_not_the_post,
    assert_receipt_contoso_safe,
    brief_leakage_ratio,
    load_holdout_pieces,
    mine_brief_from_holdout,
    raw_artifacts_dir,
    run_bare_base_write,
    run_eval_write_holdout,
    score_rag_vs_base,
    verify_holdouts_never_indexed,
)
from personality_protect.models import Piece, save_index
from personality_protect.voice_index import build_voice_index

runner = CliRunner()


def _contoso_pieces() -> list[Piece]:
    return [
        Piece(
            id="contoso-pricing",
            source="linkedin_post",
            text=(
                "Pricing experiments work when teams compare customer value "
                "and renewal signals.\n"
                "\n"
                "Contoso kept the test boring on purpose."
            ),
        ),
        Piece(
            id="contoso-platform",
            source="linkedin_post",
            text=(
                "Platform migrations need careful service boundaries.\n"
                "\n"
                "Name one owner before you start."
            ),
        ),
        Piece(
            id="contoso-ops",
            source="linkedin_post",
            text=(
                "Operations queues improve when Contoso picks one metric.\n"
                "\n"
                "Ignore the rest for a week."
            ),
        ),
        Piece(
            id="contoso-holdout",
            source="linkedin_post",
            text=(
                "Customer pricing and packaging lessons belong in the private holdout.\n"
                "\n"
                "You name one owner.\n"
                "\n"
                "Keep Contoso Ledger boring.\n"
                "\n"
                "Cut exceptions by 12% over seven weeks or stop pretending "
                "the roadmap is the work."
            ),
        ),
    ]


def _seed_contoso_index(tmp_path: Path) -> None:
    paths, _, _ = init_profile("contoso", home=tmp_path)
    save_index(paths.index_path, _contoso_pieces())
    build_voice_index(paths, holdout_ids={"contoso-holdout"})


@pytest.mark.parametrize("module", ["mlx", "mlx.nn", "mlx.core", "mlx_lm"])
def test_conftest_blocks_real_mlx_imports(module: str):
    with pytest.raises(ImportError, match="MLX is blocked under pytest"):
        importlib.import_module(module)


def test_eval_module_import_does_not_import_mlx():
    from personality_protect import eval_write_holdout as mod

    assert mod.run_eval_write_holdout is not None
    assert not [name for name in sys.modules if name.split(".", 1)[0] == "mlx"]


def test_g1_holdout_carve_never_indexed(tmp_path: Path):
    _seed_contoso_index(tmp_path)
    paths, _, _ = init_profile("contoso", home=tmp_path)

    carve = verify_holdouts_never_indexed(paths, ["contoso-holdout"])
    assert carve["ok"] is True
    assert carve["indexed_holdout_ids"] == []
    assert carve["holdout_ids"] == ["contoso-holdout"]

    pieces = load_holdout_pieces(paths, ["contoso-holdout"])
    assert len(pieces) == 1
    assert pieces[0].id == "contoso-holdout"
    assert "private holdout" in pieces[0].text


def test_g1_fails_when_holdout_leaked_into_index(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    save_index(paths.index_path, _contoso_pieces())
    # Intentionally index without holdout skip — should be caught by eval.
    build_voice_index(paths, holdout_ids=())

    carve = verify_holdouts_never_indexed(paths, ["contoso-holdout"])
    assert carve["ok"] is False
    assert "contoso-holdout" in carve["indexed_holdout_ids"]

    with pytest.raises(ValueError, match="voice_index"):
        run_eval_write_holdout(
            paths,
            ["contoso-holdout"],
            k=3,
            generate_fn=lambda *_a, **_k: "noop",
        )


def test_g2_mine_brief_is_terse_bullets_not_the_post():
    text = (
        "Customer pricing and packaging lessons belong in the private holdout.\n"
        "\n"
        "You name one owner.\n"
        "\n"
        "Keep Contoso Ledger boring.\n"
        "\n"
        "Cut exceptions by 12% over seven weeks or stop pretending "
        "the roadmap is the work."
    )
    brief = mine_brief_from_holdout(text, holdout_id="contoso-holdout")

    assert brief["holdout_id"] == "contoso-holdout"
    assert brief["topic"].startswith("Customer pricing")
    # Points are bullets, not the body. Short Contoso posts may yield only one
    # bullet under the leakage budget; that is fine — the fail-closed check is
    # "not the post", not "at least N bullets".
    assert brief["points"] != text
    assert brief["points"].startswith("- ")
    assert "12%" in brief["points"] or "Contoso" in brief["points"]
    # Cap: a brief that hands back the whole post is an answer key.
    assert_brief_is_not_the_post(brief, text)
    assert brief_leakage_ratio(brief, text) <= 0.6
    # Guard facts are the same brief the model sees — never the raw body.
    assert brief["guard_facts"] == (f"Topic: {brief['topic']}\nPoints: {brief['points']}")
    assert text not in brief["guard_facts"]
    # Contoso-safe: no personal markers in mined brief.
    blob = (brief["topic"] + "\n" + brief["points"]).lower()
    for token in ("linkedin.com", "@gmail", "dusan"):
        assert token not in blob


def test_g2_mine_brief_is_deterministic():
    text = (
        "Platform migrations need careful service boundaries.\n"
        "\n"
        "Name one owner before you start Contoso Ledger."
    )
    a = mine_brief_from_holdout(text, holdout_id="x")
    b = mine_brief_from_holdout(text, holdout_id="x")
    assert a == b


def test_g3_bare_base_prompt_has_no_exemplars():
    calls: list = []

    def fake_generate(messages, **_kwargs: object) -> str:
        calls.append(messages)
        return "Contoso names one owner and keeps Ledger boring."

    result = run_bare_base_write(
        "Contoso pricing",
        "- Name one owner\n- Keep Ledger boring",
        generate_fn=fake_generate,
        base_model="mlx-community/Qwen3.5-9B-4bit",
    )

    assert len(calls) == 1
    user = calls[0][1]["content"]
    # Empty EXAMPLES header is scaffolding — it must be absent entirely.
    assert "EXAMPLES" not in user
    assert user.startswith("BRIEF:\n")
    assert "Pricing experiments work" not in user
    assert result["mode"] == "bare_base"
    assert result["adapter"] == "none"
    assert result["k"] == 0
    assert result["exemplar_ids"] == []


def test_g4_score_schema_prefers_rag_closer_to_holdout():
    holdout = (
        "Contoso keeps the queue boring.\n"
        "\n"
        "You ship the reconciliation or you own the outage.\n"
        "\n"
        "Stop pretending the roadmap is the work."
    )
    brief = (
        "Topic: Contoso ops\n"
        "Points: Contoso keeps the queue boring; ship reconciliation; own the outage."
    )
    # Fragmented, you-forward — closer to holdout. Contoso only sentence-initial
    # (normalized away for invent-guard) so invent flags stay clean.
    rag = (
        "Contoso keeps the queue boring.\n"
        "\n"
        "You ship the fix or you own the miss.\n"
        "\n"
        "Stop dressing the roadmap as the work."
    )
    # Long press-release prose — farther from holdout.
    base = (
        "A Contoso operations bulletin outlined reconciliation improvements and "
        "roadmap alignment for partners across regional distribution centers "
        "following extended consultation with operations leadership."
    )

    score = score_rag_vs_base(holdout, rag, base, brief)
    assert score["winner"] == "rag"
    assert score["rag"]["distance"] < score["base"]["distance"]
    assert score["rag"]["invent_reject"] is False
    assert "invented_entities_count" in score["rag"]
    assert "invented_numbers_count" in score["base"]


def test_g4_invent_flags_fire_on_entities_beyond_brief():
    holdout = "Contoso keeps pricing tests boring."
    brief = "Topic: Contoso pricing\nPoints: Keep tests boring."
    rag = "Contoso keeps pricing tests boring."
    base = "Contoso should copy Fabrikam and cut exceptions by 18%."

    score = score_rag_vs_base(holdout, rag, base, brief)
    assert score["base"]["invent_reject"] is True
    assert score["base"]["invented_entities_count"] >= 1
    assert "fabrikam" in score["base"]["invented_entities"]
    assert score["rag"]["invent_reject"] is False


def test_g5_run_eval_write_holdout_receipt_contoso_safe(tmp_path: Path):
    _seed_contoso_index(tmp_path)
    paths, _, _ = init_profile("contoso", home=tmp_path)

    def rag_generate(messages, **_kwargs: object) -> str:
        user = messages[1]["content"]
        assert "EXAMPLES (voice reference only" in user
        examples_section = user.split("EXAMPLES", 1)[1].split("\n\nBRIEF:", 1)[0]
        # Holdout body must never appear as an exemplar.
        assert "private holdout" not in examples_section
        assert "contoso-holdout" not in examples_section
        # Brief points must be bullets, not the full holdout.
        brief_section = user.split("BRIEF:\n", 1)[1]
        assert "Cut exceptions by 12%" in brief_section or "owner" in brief_section
        assert brief_section.count("\n") >= 1
        return "Customer pricing needs one owner.\n\nYou keep Contoso Ledger boring."

    def base_generate(messages, **_kwargs: object) -> str:
        user = messages[1]["content"]
        assert "EXAMPLES" not in user
        return (
            "A quarterly Contoso pricing bulletin outlined packaging lessons "
            "and private holdout alignment for regional customer segments "
            "after extended leadership consultation."
        )

    receipt = run_eval_write_holdout(
        paths,
        ["contoso-holdout"],
        k=3,
        generate_fn=rag_generate,
        generate_fn_base=base_generate,
        save_raw=True,
    )

    assert receipt["kind"] == "eval_write_holdout"
    assert receipt["adapter"] == "none"
    assert receipt["n_holdouts"] == 1
    assert receipt["holdout_ids"] == ["contoso-holdout"]
    assert receipt["carve"]["ok"] is True
    assert receipt["carve"]["voice_index"] == "voice_index"
    assert receipt["wins"]["rag"] + receipt["wins"]["base"] + receipt["wins"]["tie"] == 1
    assert "contoso-holdout" not in (receipt["items"][0].get("exemplar_ids") or [])
    assert receipt["items"][0]["brief_leakage_ratio"] < 0.6
    assert receipt["raw_artifacts_saved"] is True
    assert_receipt_contoso_safe(receipt)
    blob = json.dumps(receipt).lower()
    # Receipt omits draft/holdout bodies (word counts only).
    assert "private holdout" not in blob
    assert "text" not in receipt["items"][0]
    assert "prompt" not in receipt["items"][0]

    raw_dir = raw_artifacts_dir(paths)
    for arm in ("rag", "bare_base"):
        prompt_path = raw_dir / f"contoso-holdout.{arm}.prompt.txt"
        draft_path = raw_dir / f"contoso-holdout.{arm}.draft.txt"
        assert prompt_path.is_file()
        assert draft_path.is_file()
        assert prompt_path.read_text(encoding="utf-8").strip()
        assert draft_path.read_text(encoding="utf-8").strip()


def test_cli_eval_write_holdout_json_receipt(tmp_path: Path):
    _seed_contoso_index(tmp_path)

    rag = "Customer pricing needs one owner.\n\nYou keep Contoso Ledger boring."
    base = (
        "A Contoso pricing bulletin described packaging lessons for regional "
        "customer segments after leadership consultation across three continents."
    )
    calls: list = []

    def fake_mlx(messages, **_kwargs: object) -> str:
        calls.append(messages)
        user = messages[1]["content"]
        if "EXAMPLES" not in user:
            return base
        return rag

    with patch(
        "personality_protect.eval_write_holdout.mlx_generate_no_adapter",
        side_effect=fake_mlx,
    ):
        result = runner.invoke(
            app,
            [
                "--logo",
                "off",
                "eval-write-holdout",
                "--holdout-id",
                "contoso-holdout",
                "--k",
                "3",
                "--profile",
                "contoso",
                "--home",
                str(tmp_path),
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["kind"] == "eval_write_holdout"
    assert payload["adapter"] == "none"
    assert payload["carve"]["ok"] is True
    assert "private holdout" not in result.output.lower()
    assert len(calls) >= 2
