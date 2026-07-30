"""Contoso-safe writer SFT: brief→post rows, holdouts excluded."""

from __future__ import annotations

import json
from pathlib import Path

from personality_protect.config import init_profile
from personality_protect.models import Piece, save_index
from personality_protect.select import Selection
from personality_protect.style_profile import build_style_profile, save_style_profile
from personality_protect.writer_sft import (
    build_writer_sft,
    piece_to_writer_example,
    run_build_writer_sft,
    writer_sft_path,
)

CONTOSO_LONG = (
    "Contoso Ledger keeps the queue boring on purpose.\n\n"
    "You ship the reconciliation or you own the outage.\n\n"
    "You name one owner before the packaging change starts.\n\n"
    "You cut exceptions or you explain them in writing.\n\n"
    "Partners already know which one you picked this quarter.\n\n"
    "Stop pretending the roadmap is the work.\n\n"
    "You own the queue you refuse to look at.\n\n"
    "Boring beats clever every single time Contoso ships Ledger.\n\n"
    "You keep Contoso boring and the partners stay calm."
)


def test_piece_to_writer_example_builds_chat_row():
    piece = Piece(id="c1", source="linkedin_post", text=CONTOSO_LONG, year=2024)
    row = piece_to_writer_example(piece)
    assert row is not None
    assert row["meta"]["pair_kind"] == "writer"
    assert row["messages"][-1]["role"] == "assistant"
    assert "Contoso Ledger" in row["messages"][-1]["content"]
    assert "BRIEF:" in row["messages"][1]["content"]


def test_build_writer_sft_excludes_holdouts(tmp_path: Path):
    pieces = [
        Piece(id="keep", source="linkedin_post", text=CONTOSO_LONG, year=2024),
        Piece(id="hold", source="linkedin_post", text=CONTOSO_LONG + " Extra.", year=2024),
    ]
    out = tmp_path / "writer.jsonl"
    receipt = build_writer_sft(pieces, out, holdout_ids={"hold"})
    assert receipt["examples"] == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["meta"]["piece_id"] == "keep"


def test_run_build_writer_sft_writes_profile_file(tmp_path: Path):
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = [
        Piece(id="c1", source="linkedin_post", text=CONTOSO_LONG, year=2024),
        Piece(id="c2", source="linkedin_post", text=CONTOSO_LONG + " Again.", year=2024),
    ]
    save_index(paths.index_path, pieces)
    save_style_profile(paths, build_style_profile(pieces))
    paths.selection_path.write_text(
        json.dumps(
            Selection(
                piece_ids=["c1", "c2"],
                min_words=10,
                through_year=2024,
                include_undated=True,
                summary={"pieces": 2},
            ).to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.root.joinpath("dogfood_holdout_ids.json").write_text(
        json.dumps({"holdout_ids": ["c2"]}) + "\n",
        encoding="utf-8",
    )

    receipt = run_build_writer_sft(paths)
    assert receipt["examples"] == 1
    assert writer_sft_path(paths).is_file()
