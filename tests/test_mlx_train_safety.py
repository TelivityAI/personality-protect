"""Tests for memory-safe chunked MLX training (no silent machine melt)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from personality_protect.config import init_profile
from personality_protect.mlx_train import (
    DEFAULT_CHUNK_STEPS,
    DEFAULT_MAX_SEQ_LENGTH,
    build_mlx_lora_argv,
    parse_iter_from_line,
    plan_train_chunks,
    resolve_wired_limit_bytes,
)
from personality_protect.train import run_train


def test_plan_train_chunks_splits_and_covers_total():
    assert plan_train_chunks(0, 50) == []
    assert plan_train_chunks(40, 50) == [40]
    assert plan_train_chunks(100, 50) == [50, 50]
    assert plan_train_chunks(120, 50) == [50, 50, 20]
    assert plan_train_chunks(50, 0) == [50]  # chunk_size<=0 → one shot
    assert sum(plan_train_chunks(517 * 3, DEFAULT_CHUNK_STEPS)) == 517 * 3


def test_resolve_wired_limit_leaves_headroom():
    # 48 GB machine → must not wire ~40 GB (mlx-lm default)
    limit = resolve_wired_limit_bytes(
        memory_size=48 * 10**9,
        max_recommended=40 * 10**9,
    )
    assert limit <= 24 * 10**9
    assert limit < 40 * 10**9
    # explicit override wins
    assert (
        resolve_wired_limit_bytes(
            memory_size=48 * 10**9,
            max_recommended=40 * 10**9,
            memory_gb=12.0,
        )
        == 12 * 10**9
    )


def test_build_mlx_lora_argv_is_memory_safe():
    argv = build_mlx_lora_argv(
        model="mlx-community/Qwen3.5-9B-4bit",
        data_dir=Path("/tmp/data"),
        adapter_dir=Path("/tmp/adapters"),
        iters=50,
        resume_adapter=Path("/tmp/adapters/adapters.safetensors"),
    )
    assert "--train" in argv
    assert "--batch-size" in argv and argv[argv.index("--batch-size") + 1] == "1"
    assert "--grad-checkpoint" in argv
    assert argv[argv.index("--max-seq-length") + 1] == str(DEFAULT_MAX_SEQ_LENGTH)
    assert DEFAULT_MAX_SEQ_LENGTH <= 512
    assert argv[argv.index("--iters") + 1] == "50"
    # resume only if file exists — Path may not exist in unit test
    assert int(argv[argv.index("--steps-per-eval") + 1]) >= 10**6


def test_parse_iter_from_line():
    assert parse_iter_from_line("Iter 12: Train loss 1.234, Peak mem 9.1 GB") == 12
    assert parse_iter_from_line("Loading pretrained model") is None
    assert parse_iter_from_line("Saved final weights to /x") is None


def test_run_train_mlx_uses_chunks_and_progress(tmp_path: Path):
    paths, _, _ = init_profile("mlxsafe", home=tmp_path)
    index = [
        {
            "id": f"p{i}",
            "text": ("authentic voice sentence " * 20).strip(),
            "word_count": 60,
            "source": "note",
            "year": 2024,
            "path": str(tmp_path / f"{i}.txt"),
        }
        for i in range(25)
    ]
    paths.index_path.write_text(
        "\n".join(json.dumps(p) for p in index) + "\n", encoding="utf-8"
    )
    paths.selection_path.write_text(
        json.dumps(
            {
                "piece_ids": [p["id"] for p in index],
                "min_words": 50,
                "through_year": 2026,
                "include_undated": True,
                "summary": {"count": 25},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    progress_events: list[dict] = []

    def on_progress(info: dict) -> None:
        progress_events.append(info)

    with patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk:

        def _side_effect(**kwargs):
            (kwargs["adapter_dir"] / "adapters.safetensors").write_bytes(b"fake")
            (kwargs["adapter_dir"] / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            return MagicMock(
                returncode=0,
                adapters_ok=True,
                last_iter=kwargs["iters"],
                peak_mem_gb=8.0,
                stdout=f"Iter {kwargs['iters']}: Train loss 1.0\n",
                stderr="",
            )

        mock_chunk.side_effect = _side_effect

        result = run_train(
            paths,
            backend="mlx",
            max_steps=120,
            force=True,
            chunk_steps=50,
            memory_gb=16.0,
            progress_callback=on_progress,
            force_rebuild_sft=True,
        )

    assert result.status == "ok"
    assert result.backend == "mlx"
    assert result.steps == 120
    assert mock_chunk.call_count == 3  # 50+50+20
    assert any(e.get("kind") == "chunk_start" for e in progress_events)
    assert any(e.get("kind") == "done" for e in progress_events)
