"""Tests for train defaults, anti-mock guard, and corpus gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import (
    CORPUS_BLOCK_BELOW,
    CORPUS_WARN_BELOW,
    SMOKE_MAX_STEPS,
    init_profile,
)
from personality_protect.demo import run_demo
from personality_protect.ingest import run_ingest
from personality_protect.select import run_select
from personality_protect.train import (
    MockFallbackError,
    auto_max_steps,
    check_corpus_size,
    detect_backend,
    run_train,
)

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_auto_max_steps_scales_and_clamps():
    assert auto_max_steps(1, smoke=True) == SMOKE_MAX_STEPS
    assert auto_max_steps(100, smoke=False) == 300  # 100 * 3 epochs
    assert auto_max_steps(10) == 50  # min clamp
    assert auto_max_steps(5000) == 2000  # max clamp
    assert auto_max_steps(100, max_steps=42) == 42


def test_corpus_gates_warn_and_block():
    warn = check_corpus_size(CORPUS_WARN_BELOW - 1, force=False, smoke=False)
    assert warn is not None
    assert str(CORPUS_WARN_BELOW) in warn

    with pytest.raises(RuntimeError, match="too small"):
        check_corpus_size(CORPUS_BLOCK_BELOW - 1, force=False, smoke=False)

    note = check_corpus_size(5, force=True, smoke=False)
    assert note is not None
    assert check_corpus_size(5, smoke=True) is not None
    assert check_corpus_size(CORPUS_WARN_BELOW + 10) is None


def test_detect_backend_refuses_silent_mock():
    # Explicit mock is fine
    assert detect_backend("mock") == "mock"
    from personality_protect.train import _has_mlx

    if not _has_mlx():
        with pytest.raises(MockFallbackError):
            detect_backend("mlx", allow_mock=False)
        assert detect_backend("mlx", allow_mock=True) == "mock"
    else:
        assert detect_backend("mlx", allow_mock=False) == "mlx"

def test_train_persists_metadata(tmp_path: Path):
    run_demo(home=tmp_path)
    paths, _, _ = init_profile("demo", home=tmp_path)
    # demo already trained; re-train mock and check meta
    result = run_train(
        paths, backend="mock", mock=True, smoke=True, force=True, max_steps=1
    )
    assert result.status == "ok"
    meta_path = paths.adapter_meta
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["backend"] == "mock"
    assert meta["examples"] >= 1
    assert "steps" in meta
    assert meta["sft_path"]
    train_result = paths.adapters_dir / "latest" / "train_result.json"
    assert train_result.is_file()


def test_train_blocks_tiny_corpus_without_force(tmp_path: Path):
    paths, _, _ = init_profile("t", home=tmp_path)
    run_ingest(
        paths,
        linkedin=FIXTURES / "linkedin",
        local=[FIXTURES / "local_docs"],
        source_hint="note",
    )
    run_select(paths, min_words=20, through_year=2024, include_undated=True)
    with pytest.raises(RuntimeError, match="too small|Corpus"):
        run_train(paths, backend="cpu", force=False, smoke=False, mock=False)


def test_cli_select_force_and_train_smoke(tmp_path: Path):
    home = str(tmp_path)
    r = runner.invoke(app, ["--logo", "off", "init", "--home", home, "--profile", "t", "--json"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        [
            "--logo", "off", "ingest",
            "--home", home, "--profile", "t",
            "--path", str(FIXTURES / "local_docs"),
            "--source", "note",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        [
            "--logo", "off", "select",
            "--home", home, "--profile", "t",
            "--min-words", "10",
            "--include-undated",
            "--json",
        ],
    )
    # tiny corpus → exit 2 without --force
    assert r.exit_code == 2, r.output
    r = runner.invoke(
        app,
        [
            "--logo", "off", "select",
            "--home", home, "--profile", "t",
            "--min-words", "10",
            "--include-undated",
            "--force",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        app,
        [
            "--logo", "off", "train",
            "--home", home, "--profile", "t",
            "--backend", "mock",
            "--smoke",
            "--force",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["status"] == "ok"
    assert data["smoke"] is True
    assert data["steps"] <= SMOKE_MAX_STEPS or data["steps"] == 1
