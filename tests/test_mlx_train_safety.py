"""Tests for memory-safe chunked MLX training (no silent machine melt)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
    assert DEFAULT_MAX_SEQ_LENGTH == 1024
    assert argv[argv.index("--iters") + 1] == "50"
    # Train loss on assistant rewrite only — otherwise LoRA memorizes the draft.
    assert "--mask-prompt" in argv
    # resume only if file exists — Path may not exist in unit test
    assert int(argv[argv.index("--steps-per-eval") + 1]) >= 10**6


def test_default_wired_cap_is_at_most_16gb():
    """Studio stays usable; never default to mlx-lm's ~40GB recommendation."""
    from personality_protect.mlx_train import DEFAULT_WIRED_CAP_BYTES

    assert DEFAULT_WIRED_CAP_BYTES <= 16 * 10**9


def test_mlx_import_blocked_when_disabled(monkeypatch):
    from personality_protect import mlx_runtime as rt

    monkeypatch.setenv("PP_MLX_DISABLE", "1")
    try:
        rt.assert_mlx_import_allowed()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "PP_MLX_DISABLE" in str(exc)


def test_parse_iter_from_line():
    assert parse_iter_from_line("Iter 12: Train loss 1.234, Peak mem 9.1 GB") == 12
    assert parse_iter_from_line("Loading pretrained model") is None
    assert parse_iter_from_line("Saved final weights to /x") is None


def test_stdout_reports_nan_loss():
    from personality_protect.mlx_train import stdout_reports_nan_loss

    assert stdout_reports_nan_loss(
        "Iter 5: Train loss nan, Learning Rate 1.000e-05, Peak mem 13.7 GB"
    )
    assert not stdout_reports_nan_loss("Iter 5: Train loss 0.42, Peak mem 13.7 GB")


def test_nan_chunk_restores_last_good_adapter(tmp_path: Path):
    """Failed/nan chunks must not leave poisoned weights; restore pre-chunk snapshot."""
    from personality_protect.mlx_train import (
        LAST_GOOD_ADAPTER_NAME,
        run_chunked_mlx_train,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"GOOD_750")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    with patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk:

        def _side_effect(**kwargs):
            # Simulate mlx-lm overwriting weights then reporting nan.
            (kwargs["adapter_dir"] / "adapters.safetensors").write_bytes(b"POISON")
            return MagicMock(
                returncode=0,
                adapters_ok=True,
                last_iter=10,
                peak_mem_gb=13.0,
                stdout="Iter 10: Train loss nan, Learning Rate 1.000e-05, Peak mem 13.0 GB\n",
                stderr="",
            )

        mock_chunk.side_effect = _side_effect
        with pytest.raises(RuntimeError, match="Train loss nan"):
            run_chunked_mlx_train(
                model="m",
                data_dir=data_dir,
                adapter_dir=adapter_dir,
                total_steps=800,
                chunk_steps=50,
                memory_gb=16.0,
                resume=True,
            )

    assert (adapter_dir / "adapters.safetensors").read_bytes() == b"GOOD_750"
    assert (adapter_dir / LAST_GOOD_ADAPTER_NAME).read_bytes() == b"GOOD_750"
    meta = json.loads((adapter_dir / "train_chunks.json").read_text(encoding="utf-8"))
    assert meta["nan_loss"] is True
    assert meta["restored_last_good"] is True
    assert meta["completed_steps"] == 0  # no prior checkpoint meta → already=0


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

    import importlib.util

    real_find_spec = importlib.util.find_spec

    def _find_spec(name: str, package: str | None = None):
        if name == "mlx_lm":
            return MagicMock()  # present
        return real_find_spec(name, package)

    with (
        patch("personality_protect.train._has_mlx", return_value=True),
        patch("importlib.util.find_spec", side_effect=_find_spec),
        patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk,
    ):

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
            max_seq_length=2048,
            progress_callback=on_progress,
            force_rebuild_sft=True,
        )

    assert result.status == "ok"
    assert result.backend == "mlx"
    assert result.steps == 120
    assert mock_chunk.call_count == 3  # 50+50+20
    assert all(call.kwargs["max_seq_length"] == 2048 for call in mock_chunk.call_args_list)
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


def test_checkpoint_meta_roundtrip(tmp_path: Path):
    from personality_protect.mlx_train import (
        load_train_checkpoint_meta,
        write_train_checkpoint_meta,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    meta = {
        "status": "in_progress",
        "completed_steps": 100,
        "total_steps": 750,
        "chunk_plan": [50, 50, 50],
        "last_chunk": 2,
        "chunk_steps": 50,
    }
    write_train_checkpoint_meta(adapter_dir, meta)
    loaded = load_train_checkpoint_meta(adapter_dir)
    assert loaded is not None
    assert loaded["completed_steps"] == 100
    assert loaded["total_steps"] == 750
    assert loaded["last_chunk"] == 2
    assert loaded["status"] == "in_progress"


def test_resolve_resume_plan_remaining_only(tmp_path: Path):
    from personality_protect.mlx_train import (
        resolve_train_plan,
        write_train_checkpoint_meta,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"ckpt")
    write_train_checkpoint_meta(
        adapter_dir,
        {
            "status": "in_progress",
            "completed_steps": 100,
            "total_steps": 250,
            "chunk_plan": [50, 50, 50, 50, 50],
            "last_chunk": 2,
            "chunk_steps": 50,
        },
    )
    plan = resolve_train_plan(
        adapter_dir=adapter_dir,
        total_steps=250,
        chunk_steps=50,
        resume=True,
        force_retrain=False,
    )
    assert plan.already_completed == 100
    assert plan.chunks_to_run == [50, 50, 50]
    assert plan.total_steps == 250
    assert (adapter_dir / "adapters.safetensors").is_file()


def test_resolve_resume_plan_force_retrain_wipes(tmp_path: Path):
    from personality_protect.mlx_train import resolve_train_plan

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"old")
    (adapter_dir / "0000050_adapters.safetensors").write_bytes(b"old50")
    (adapter_dir / "train_chunks.json").write_text("{}", encoding="utf-8")
    plan = resolve_train_plan(
        adapter_dir=adapter_dir,
        total_steps=100,
        chunk_steps=50,
        resume=False,
        force_retrain=True,
    )
    assert plan.already_completed == 0
    assert plan.chunks_to_run == [50, 50]
    assert not (adapter_dir / "adapters.safetensors").exists()
    assert not (adapter_dir / "0000050_adapters.safetensors").exists()


def test_incomplete_checkpoint_auto_resumes_without_flag(tmp_path: Path):
    """Crash restart must NOT wipe — incomplete train_chunks.json auto-resumes."""
    from personality_protect.mlx_train import (
        is_incomplete_checkpoint,
        resolve_train_plan,
        write_train_checkpoint_meta,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"GOOD")
    write_train_checkpoint_meta(
        adapter_dir,
        {
            "status": "in_progress",
            "completed_steps": 50,
            "total_steps": 200,
            "chunk_plan": [50, 50, 50, 50],
            "last_chunk": 1,
            "chunk_steps": 50,
        },
    )
    assert is_incomplete_checkpoint(adapter_dir) is True
    plan = resolve_train_plan(
        adapter_dir=adapter_dir,
        total_steps=200,
        chunk_steps=50,
        resume=False,  # user forgot --resume
        force_retrain=False,
    )
    assert plan.resume is True
    assert plan.auto_resumed is True
    assert plan.already_completed == 50
    assert plan.chunks_to_run == [50, 50, 50]
    assert (adapter_dir / "adapters.safetensors").read_bytes() == b"GOOD"


def test_completed_checkpoint_still_wipes_without_resume(tmp_path: Path):
    """Finished runs without --resume still start clean (use --force-retrain to be explicit)."""
    from personality_protect.mlx_train import (
        is_incomplete_checkpoint,
        resolve_train_plan,
        write_train_checkpoint_meta,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"DONE")
    write_train_checkpoint_meta(
        adapter_dir,
        {
            "status": "complete",
            "completed_steps": 750,
            "total_steps": 750,
            "steps_this_run": 750,
        },
    )
    assert is_incomplete_checkpoint(adapter_dir) is False
    plan = resolve_train_plan(
        adapter_dir=adapter_dir,
        total_steps=200,
        chunk_steps=50,
        resume=False,
        force_retrain=False,
    )
    assert plan.resume is False
    assert plan.already_completed == 0
    assert not (adapter_dir / "adapters.safetensors").exists()


def test_legacy_steps_this_run_counts_as_completed(tmp_path: Path):
    from personality_protect.mlx_train import completed_steps_from_meta

    assert completed_steps_from_meta({"steps_this_run": 750}) == 750
    assert completed_steps_from_meta({"completed_steps": 100, "steps_this_run": 50}) == 100
    assert completed_steps_from_meta(None) == 0


def test_chunked_train_writes_meta_after_each_chunk(tmp_path: Path):
    from personality_protect.mlx_train import (
        load_train_checkpoint_meta,
        run_chunked_mlx_train,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    metas_during: list[dict] = []

    with patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk:

        def _side_effect(**kwargs):
            (kwargs["adapter_dir"] / "adapters.safetensors").write_bytes(b"w")
            # Numbered checkpoint (mlx-lm style)
            n = kwargs["iters"]
            (
                kwargs["adapter_dir"] / f"{kwargs.get('_done', 0) + n:07d}_adapters.safetensors"
            )
            meta = load_train_checkpoint_meta(adapter_dir)
            # Before this chunk finishes, prior meta should exist from previous chunk
            metas_during.append(meta)
            return MagicMock(
                returncode=0,
                adapters_ok=True,
                last_iter=kwargs["iters"],
                peak_mem_gb=6.0,
                stdout=f"Iter {kwargs['iters']}: Train loss 1.0\n",
                stderr="",
            )

        mock_chunk.side_effect = _side_effect
        final = run_chunked_mlx_train(
            model="m",
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            total_steps=120,
            chunk_steps=50,
            memory_gb=16.0,
            resume=False,
        )

    assert mock_chunk.call_count == 3
    mid = load_train_checkpoint_meta(adapter_dir)
    assert mid is not None
    assert mid["completed_steps"] == 120
    assert mid["total_steps"] == 120
    assert mid["status"] == "complete"
    assert mid["last_chunk"] == 3
    assert mid["chunk_plan"] == [50, 50, 20]
    assert final["completed_steps"] == 120
    # After chunk 1 completed, meta must already show 50 before chunk 2 runs
    assert metas_during[1] is not None
    assert metas_during[1]["completed_steps"] == 50
    assert metas_during[1]["status"] == "in_progress"
    from personality_protect.mlx_train import list_step_checkpoints

    steps = [p.name for p in list_step_checkpoints(adapter_dir)]
    assert steps == ["step_000050", "step_000100", "step_000120"]
    assert (adapter_dir / "checkpoints" / "step_000050" / "adapters.safetensors").is_file()


def test_persist_step_checkpoint_roundtrip(tmp_path: Path):
    from personality_protect.mlx_train import (
        clear_step_checkpoints,
        list_step_checkpoints,
        persist_step_checkpoint,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"w50")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    dest = persist_step_checkpoint(adapter_dir, 50)
    assert dest is not None
    assert dest.name == "step_000050"
    assert (dest / "adapters.safetensors").read_bytes() == b"w50"
    assert (dest / "adapter_config.json").read_text(encoding="utf-8") == "{}"
    (adapter_dir / "adapters.safetensors").write_bytes(b"w100")
    persist_step_checkpoint(adapter_dir, 100)
    assert [p.name for p in list_step_checkpoints(adapter_dir)] == [
        "step_000050",
        "step_000100",
    ]
    clear_step_checkpoints(adapter_dir)
    assert list_step_checkpoints(adapter_dir) == []


def test_chunked_train_resume_skips_completed_steps(tmp_path: Path):
    from personality_protect.mlx_train import (
        run_chunked_mlx_train,
        write_train_checkpoint_meta,
    )

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"prior")
    write_train_checkpoint_meta(
        adapter_dir,
        {
            "status": "in_progress",
            "completed_steps": 100,
            "total_steps": 150,
            "chunk_plan": [50, 50, 50],
            "last_chunk": 2,
            "chunk_steps": 50,
        },
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    seen_iters: list[int] = []

    with patch("personality_protect.mlx_train.run_mlx_chunk_subprocess") as mock_chunk:

        def _side_effect(**kwargs):
            seen_iters.append(kwargs["iters"])
            assert kwargs["resume_adapter"] is not None
            (kwargs["adapter_dir"] / "adapters.safetensors").write_bytes(b"more")
            return MagicMock(
                returncode=0,
                adapters_ok=True,
                last_iter=kwargs["iters"],
                peak_mem_gb=6.0,
                stdout="Iter 50: Train loss 0.5\n",
                stderr="",
            )

        mock_chunk.side_effect = _side_effect
        events: list[dict] = []
        meta = run_chunked_mlx_train(
            model="m",
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            total_steps=150,
            chunk_steps=50,
            memory_gb=16.0,
            resume=True,
            progress_callback=events.append,
        )

    assert seen_iters == [50]  # only remaining chunk
    assert meta["completed_steps"] == 150
    assert meta["already_completed"] == 100
    start = next(e for e in events if e.get("kind") == "start")
    assert start["completed_steps"] == 100
    assert start["total_steps"] == 150
    assert start["resume"] is True
