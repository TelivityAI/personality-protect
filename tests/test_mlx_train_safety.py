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


def test_run_chunked_mlx_train_resume_keeps_adapter(tmp_path: Path):
    from personality_protect.mlx_train import run_chunked_mlx_train

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    existing = adapter_dir / "adapters.safetensors"
    existing.write_bytes(b"prior-weights")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    with patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk:

        def _side_effect(**kwargs):
            assert kwargs["resume_adapter"] is not None
            assert kwargs["resume_adapter"].is_file()
            (kwargs["adapter_dir"] / "adapters.safetensors").write_bytes(b"continued")
            return MagicMock(
                returncode=0,
                adapters_ok=True,
                last_iter=kwargs["iters"],
                peak_mem_gb=7.0,
                stdout="Iter 10: Train loss 1.0\n",
                stderr="",
            )

        mock_chunk.side_effect = _side_effect
        meta = run_chunked_mlx_train(
            model="mlx-community/Qwen3.5-9B-4bit",
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            total_steps=50,
            chunk_steps=50,
            memory_gb=16.0,
            resume=True,
        )

    assert meta["resume"] is True
    assert existing.read_bytes() == b"continued"
    assert mock_chunk.call_count == 1


def test_run_mlx_chunk_subprocess_forces_unbuffered(tmp_path: Path):
    """Detached/nohup trains must not hang on block-buffered child stdout."""
    from personality_protect.mlx_train import run_mlx_chunk_subprocess

    adapter_dir = tmp_path / "adapters"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    with patch("personality_protect.mlx_train.subprocess.Popen") as mock_popen:
        proc = MagicMock()
        proc.stdout = iter(["PP MLX chunk: iters=1\n", "Iter 1: Train loss 1.0\n"])
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        (adapter_dir).mkdir(parents=True, exist_ok=True)
        # Pretend adapters appear after run
        def _wait(*_a, **_k):
            (adapter_dir / "adapters.safetensors").write_bytes(b"x")
            return 0

        proc.wait.side_effect = _wait
        run_mlx_chunk_subprocess(
            model="m",
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            iters=1,
            wired_limit_bytes=16 * 10**9,
        )
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert "-u" in cmd
        assert kwargs["env"].get("PYTHONUNBUFFERED") == "1"


def test_run_chunked_mlx_train_fresh_deletes_adapter(tmp_path: Path):

    from personality_protect.mlx_train import run_chunked_mlx_train

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    existing = adapter_dir / "adapters.safetensors"
    existing.write_bytes(b"prior-weights")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    seen_resume: list = []

    with patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk:

        def _side_effect(**kwargs):
            seen_resume.append(kwargs.get("resume_adapter"))
            (kwargs["adapter_dir"] / "adapters.safetensors").write_bytes(b"fresh")
            return MagicMock(
                returncode=0,
                adapters_ok=True,
                last_iter=kwargs["iters"],
                peak_mem_gb=7.0,
                stdout="Iter 10: Train loss 1.0\n",
                stderr="",
            )

        mock_chunk.side_effect = _side_effect
        run_chunked_mlx_train(
            model="mlx-community/Qwen3.5-9B-4bit",
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            total_steps=50,
            chunk_steps=50,
            memory_gb=16.0,
            resume=False,
        )

    # First chunk of a fresh run must not resume prior weights
    assert seen_resume[0] is None
