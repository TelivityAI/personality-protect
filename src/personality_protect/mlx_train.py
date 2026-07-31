"""Memory-safe chunked MLX LoRA training.

mlx-lm's trainer calls ``mx.set_wired_limit(max_recommended_working_set_size)``,
which on a 48 GB Mac Studio is ~40 GB — enough to make macOS unstable and get
Python jetsam-killed ("Python quit unexpectedly").

We:
1. Cap the wired limit so the OS / Cursor keep headroom.
2. Train in digestible subprocess chunks so Metal memory is released between pieces.
3. Use grad checkpointing + shorter seq length by default.
4. Surface progress events for a Rich progress bar in the CLI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Digests that finish without melting a 48 GB unified-memory Mac.
DEFAULT_CHUNK_STEPS = 50
# Translation sequences contain prompt + flattened input + full author target.
# 1024 preserves typical post closers; use 2048 explicitly for article sections
# with a higher --memory-gb cap on 48 GB machines.
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_NUM_LAYERS = 8
# mlx-lm's own defaults, named here so a caller can raise them per recipe
# instead of silently inheriting whatever the upstream config ships.
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_LORA_RANK = 8
# Writer recipe. The failed run used the translator recipe unchanged: 8 layers
# and rank 8 at 1e-5 for 300 steps over 100 rows. With pairs that now demand a
# real transformation rather than a copy, the adapter needs both more capacity
# and more passes over a smaller, cleaner set — and a slightly higher rate,
# because 1e-5 on rank 8 barely moves a 9B model in 300 steps.
WRITER_NUM_LAYERS = 16
WRITER_LORA_RANK = 16
WRITER_LEARNING_RATE = 3e-5
# 10 epochs on ~60 de-voiced pairs drove train loss ~0.08 and raised invention /
# parroting on the n=20 gate. Three passes is enough to move the adapter without
# memorizing the tiny set; mid-train step snapshots let a later gate pick earlier.
WRITER_EPOCHS = 3
# Cap wired Metal memory: leave OS/apps breathing room.
DEFAULT_WIRED_FRACTION = 0.40
DEFAULT_WIRED_CAP_BYTES = 16 * 10**9  # 16 GB hard cap (leave Studio headroom)
PROOF_MAX_STEPS = 150  # enough for real receipts without a marathon

ProgressCallback = Callable[[dict[str, Any]], None]

CHECKPOINT_META_NAME = "train_chunks.json"
# Durable per-chunk copies under adapter_dir/checkpoints/step_NNNNNN/.
# Distinct from mlx-lm's ephemeral ``0000050_adapters.safetensors`` which each
# chunk overwrites with the same name.
STEP_CHECKPOINTS_DIRNAME = "checkpoints"
# Snapshot before each chunk so nan / crash never leaves a wiped or poisoned adapter.
LAST_GOOD_ADAPTER_NAME = "adapters.safetensors.last_good"

_ITER_RE = re.compile(r"^Iter\s+(\d+)\s*:", re.MULTILINE)
_PEAK_RE = re.compile(r"Peak mem\s+([0-9.]+)\s*GB", re.IGNORECASE)
_NAN_LOSS_RE = re.compile(r"Train loss\s+nan\b", re.IGNORECASE)


def snapshot_last_good_adapter(adapter_dir: Path) -> Path | None:
    """Copy current adapters.safetensors aside before a risky chunk."""
    src = adapter_dir / "adapters.safetensors"
    if not src.is_file():
        return None
    dest = adapter_dir / LAST_GOOD_ADAPTER_NAME
    shutil.copy2(src, dest)
    return dest


def restore_last_good_adapter(adapter_dir: Path) -> bool:
    """Restore adapters.safetensors from the pre-chunk snapshot if present."""
    src = adapter_dir / LAST_GOOD_ADAPTER_NAME
    if not src.is_file():
        return False
    dest = adapter_dir / "adapters.safetensors"
    shutil.copy2(src, dest)
    return True


def persist_step_checkpoint(adapter_dir: Path, completed_steps: int) -> Path | None:
    """Copy live weights into ``checkpoints/step_NNNNNN/`` after a good chunk.

    mlx-lm's own numbered files reuse the same basename every chunk, so earlier
    steps disappear. These directories keep every completed step count so a gate
    can evaluate under-trained adapters without a full retrain.
    """
    src = adapter_dir / "adapters.safetensors"
    if not src.is_file():
        return None
    steps = max(0, int(completed_steps))
    dest_dir = adapter_dir / STEP_CHECKPOINTS_DIRNAME / f"step_{steps:06d}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "adapters.safetensors")
    for name in ("adapter_config.json", "adapter_config.yaml"):
        cfg = adapter_dir / name
        if cfg.is_file():
            shutil.copy2(cfg, dest_dir / name)
    return dest_dir


def list_step_checkpoints(adapter_dir: Path) -> list[Path]:
    """Return step checkpoint dirs oldest-first (under-trained → final)."""
    root = adapter_dir / STEP_CHECKPOINTS_DIRNAME
    if not root.is_dir():
        return []
    dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "adapters.safetensors").is_file()
    ]
    return sorted(dirs, key=lambda path: path.name)


def clear_step_checkpoints(adapter_dir: Path) -> None:
    """Drop durable step snapshots (used on ``--force-retrain``)."""
    root = adapter_dir / STEP_CHECKPOINTS_DIRNAME
    if root.is_dir():
        shutil.rmtree(root)


def plan_train_chunks(total_steps: int, chunk_size: int) -> list[int]:
    """Split ``total_steps`` into positive chunk sizes (last chunk may be shorter)."""
    total = max(0, int(total_steps))
    if total == 0:
        return []
    size = int(chunk_size)
    if size <= 0:
        return [total]
    chunks: list[int] = []
    remaining = total
    while remaining > 0:
        n = min(size, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


def write_train_checkpoint_meta(adapter_dir: Path, meta: dict[str, Any]) -> Path:
    """Persist resumable train progress next to adapters (after every chunk)."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    path = adapter_dir / CHECKPOINT_META_NAME
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_train_checkpoint_meta(adapter_dir: Path) -> dict[str, Any] | None:
    """Load ``train_chunks.json`` if present and valid."""
    path = adapter_dir / CHECKPOINT_META_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def completed_steps_from_meta(meta: dict[str, Any] | None) -> int:
    """Resolve how many steps are already trained (legacy-safe)."""
    if not meta:
        return 0
    if "completed_steps" in meta and meta.get("completed_steps") is not None:
        return max(0, int(meta["completed_steps"]))
    # Older finished runs only recorded steps_this_run.
    return max(0, int(meta.get("steps_this_run") or 0))


def is_incomplete_checkpoint(adapter_dir: Path) -> bool:
    """True when adapters + train_chunks.json show an unfinished chunked train.

    Safe default for crash recovery: do not wipe these weights unless
    ``--force-retrain`` is explicit.
    """
    if not (adapter_dir / "adapters.safetensors").is_file():
        return False
    meta = load_train_checkpoint_meta(adapter_dir)
    if not meta:
        return False
    status = str(meta.get("status") or "").lower()
    if status == "in_progress":
        return True
    done = completed_steps_from_meta(meta)
    total = int(meta.get("total_steps") or 0)
    return total > 0 and done < total


@dataclass
class TrainPlan:
    """Resolved chunk plan for a fresh or resumed MLX train."""

    chunks_to_run: list[int]
    already_completed: int
    total_steps: int
    chunk_steps: int
    resume: bool
    full_chunk_plan: list[int]
    auto_resumed: bool = False


def _clear_adapter_weights(adapter_dir: Path) -> None:
    adapter_file = adapter_dir / "adapters.safetensors"
    if adapter_file.is_file():
        adapter_file.unlink()
    for stale in adapter_dir.glob("*_adapters.safetensors"):
        stale.unlink()
    last_good = adapter_dir / LAST_GOOD_ADAPTER_NAME
    if last_good.is_file():
        last_good.unlink()
    meta_path = adapter_dir / CHECKPOINT_META_NAME
    if meta_path.is_file():
        meta_path.unlink()
    clear_step_checkpoints(adapter_dir)


def resolve_train_plan(
    *,
    adapter_dir: Path,
    total_steps: int,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
    resume: bool = False,
    force_retrain: bool = False,
) -> TrainPlan:
    """Decide chunks to run; resume keeps weights and skips completed steps.

    Incomplete ``train_chunks.json`` (status=in_progress or completed < total)
    auto-resumes so a crash restart does not wipe adapters. Pass
    ``force_retrain=True`` for an explicit clean slate.
    """
    if force_retrain and resume:
        raise ValueError("Pass only one of --resume or --force-retrain")

    total = max(1, int(total_steps))
    size = int(chunk_steps)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"
    auto_resumed = False

    if force_retrain:
        _clear_adapter_weights(adapter_dir)
        full = plan_train_chunks(total, size)
        return TrainPlan(
            chunks_to_run=full,
            already_completed=0,
            total_steps=total,
            chunk_steps=size,
            resume=False,
            full_chunk_plan=full,
            auto_resumed=False,
        )

    if not resume and is_incomplete_checkpoint(adapter_dir):
        resume = True
        auto_resumed = True

    if not resume:
        _clear_adapter_weights(adapter_dir)
        full = plan_train_chunks(total, size)
        return TrainPlan(
            chunks_to_run=full,
            already_completed=0,
            total_steps=total,
            chunk_steps=size,
            resume=False,
            full_chunk_plan=full,
            auto_resumed=False,
        )

    if not adapter_file.is_file():
        raise FileNotFoundError(
            f"--resume requested but no adapter at {adapter_file}. "
            "Run a fresh train first, or pass --force-retrain."
        )

    prior = load_train_checkpoint_meta(adapter_dir) or {}
    already = completed_steps_from_meta(prior)
    # ``total_steps`` is the desired final completed count for this invocation.
    target = max(1, int(total_steps))
    remaining = max(0, target - already)
    chunks = plan_train_chunks(remaining, size)
    full = plan_train_chunks(target, size)
    return TrainPlan(
        chunks_to_run=chunks,
        already_completed=already,
        total_steps=target,
        chunk_steps=size,
        resume=True,
        full_chunk_plan=full,
        auto_resumed=auto_resumed,
    )

def resolve_wired_limit_bytes(
    *,
    memory_size: int,
    max_recommended: int,
    memory_gb: float | None = None,
    fraction: float = DEFAULT_WIRED_FRACTION,
    cap_bytes: int = DEFAULT_WIRED_CAP_BYTES,
) -> int:
    """Compute a safe Metal wired limit — never the full mlx-lm recommendation."""
    if memory_gb is not None and memory_gb > 0:
        return max(1_000_000_000, int(memory_gb * 10**9))
    frac = max(0.15, min(0.60, float(fraction)))
    from_frac = int(memory_size * frac)
    # Never accept mlx's near-full recommendation as-is.
    safe = min(from_frac, cap_bytes, max(1_000_000_000, int(max_recommended * 0.5)))
    return max(1_000_000_000, safe)


def build_mlx_lora_argv(
    *,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    iters: int,
    resume_adapter: Path | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    num_layers: int = DEFAULT_NUM_LAYERS,
    batch_size: int = 1,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> list[str]:
    """CLI argv for one mlx-lm LoRA chunk (memory-safe defaults).

    LoRA rank is absent on purpose: mlx-lm exposes it only through the
    ``lora_parameters`` config, not as a CLI flag, so the chunk worker sets it
    on the args namespace instead of here.
    """
    argv = [
        "lora",
        "--model",
        model,
        "--data",
        str(data_dir),
        "--train",
        "--fine-tune-type",
        "lora",
        "--batch-size",
        str(max(1, batch_size)),
        "--iters",
        str(max(1, iters)),
        "--adapter-path",
        str(adapter_dir),
        "--max-seq-length",
        str(max_seq_length),
        "--num-layers",
        str(num_layers),
        "--grad-checkpoint",
        "--steps-per-report",
        "5",
        # Avoid mid-train eval (empty valid + eval allocates another forward pass).
        "--steps-per-eval",
        str(10**9),
        "--val-batches",
        "0",
        "--save-every",
        str(max(1, iters)),
        "--learning-rate",
        f"{learning_rate:g}",
        # Loss on assistant tokens only — rewrite SFT, not draft echo.
        "--mask-prompt",
    ]
    if resume_adapter is not None and resume_adapter.is_file():
        argv.extend(["--resume-adapter-file", str(resume_adapter)])
    return argv


def parse_iter_from_line(line: str) -> int | None:
    m = _ITER_RE.search(line.strip())
    return int(m.group(1)) if m else None


def stdout_reports_nan_loss(text: str) -> bool:
    """True when mlx-lm printed Train loss nan (usually empty masked labels)."""
    return bool(_NAN_LOSS_RE.search(text or ""))


def parse_peak_mem_gb(text: str) -> float | None:
    matches = _PEAK_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


@dataclass
class ChunkResult:
    returncode: int
    adapters_ok: bool
    last_iter: int
    peak_mem_gb: float | None
    stdout: str
    stderr: str


def run_mlx_chunk_subprocess(
    *,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    iters: int,
    wired_limit_bytes: int,
    resume_adapter: Path | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_rank: int = DEFAULT_LORA_RANK,
    on_line: Callable[[str], None] | None = None,
    timeout: int | None = None,
) -> ChunkResult:
    """Run one MLX LoRA chunk in a fresh process (releases Metal memory on exit)."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PP_MLX_WIRED_BYTES"] = str(int(wired_limit_bytes))
    env["TOKENIZERS_PARALLELISM"] = "true"
    # Critical: child stdout is a PIPE (not a TTY). Without this, mlx-lm's
    # prints stay block-buffered and the parent appears frozen / gets killed
    # while waiting for the first "Iter" line in redirected nohup logs.
    env["PYTHONUNBUFFERED"] = "1"
    # Keep HF from spawning thread storms.
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    cmd = [
        sys.executable,
        "-u",  # unbuffered stdio even if env is stripped
        "-m",
        "personality_protect.mlx_chunk_worker",
        "--model",
        model,
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(max(1, iters)),
        "--max-seq-length",
        str(max_seq_length),
        "--num-layers",
        str(num_layers),
        "--learning-rate",
        f"{learning_rate:g}",
        "--lora-rank",
        str(max(1, lora_rank)),
        "--wired-bytes",
        str(int(wired_limit_bytes)),
    ]
    if resume_adapter is not None and resume_adapter.is_file():
        cmd.extend(["--resume-adapter-file", str(resume_adapter)])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    lines: list[str] = []
    last_iter = 0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            lines.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
            parsed = parse_iter_from_line(line)
            if parsed is not None:
                last_iter = parsed
        returncode = proc.wait(timeout=timeout)
    except Exception:
        proc.kill()
        raise

    out = "".join(lines)
    adapters_ok = (adapter_dir / "adapters.safetensors").is_file()
    return ChunkResult(
        returncode=returncode,
        adapters_ok=adapters_ok,
        last_iter=last_iter,
        peak_mem_gb=parse_peak_mem_gb(out),
        stdout=out,
        stderr="",
    )


def detect_device_memory() -> tuple[int, int]:
    """Return (memory_size, max_recommended_working_set_size) from MLX if possible.

    Never import ``mlx`` unless ``PP_MLX_ALLOW=1`` opts in — sandboxed Cursor
    shells SIGABRT on ``metal::load_device`` (~80ms) and take Python with them.
    """
    try:
        from personality_protect.mlx_runtime import assert_mlx_import_allowed

        assert_mlx_import_allowed()
    except Exception:
        return 48 * 10**9, 40 * 10**9
    try:
        import mlx.core as mx

        info = mx.device_info()
        return int(info["memory_size"]), int(info["max_recommended_working_set_size"])
    except Exception:
        # Sensible fallbacks for tests / non-Metal hosts
        return 48 * 10**9, 40 * 10**9


def run_chunked_mlx_train(
    *,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    total_steps: int,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
    memory_gb: float | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lora_rank: int = DEFAULT_LORA_RANK,
    resume: bool = False,
    force_retrain: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train LoRA in subprocess chunks with progress events.

    Each successful chunk persists ``adapters.safetensors`` (mlx-lm) and updates
    ``train_chunks.json`` so a crash can ``--resume`` from completed_steps.
    Incomplete checkpoints (status=in_progress or completed < total) auto-resume
    even without ``resume=True`` so a plain restart does not wipe weights.
    Pass ``resume=True`` to continue a finished adapter toward a higher
    ``total_steps``. Pass ``force_retrain=True`` for an explicit clean slate.
    """
    if force_retrain and resume:
        raise ValueError("Pass only one of --resume or --force-retrain")

    plan = resolve_train_plan(
        adapter_dir=adapter_dir,
        total_steps=total_steps,
        chunk_steps=chunk_steps,
        resume=resume,
        force_retrain=force_retrain,
    )
    chunks = plan.chunks_to_run
    if not chunks:
        # Already at or past target — treat as complete receipt.
        meta = {
            "status": "complete",
            "chunks": 0,
            "chunk_plan": plan.full_chunk_plan,
            "completed_steps": plan.already_completed,
            "total_steps": plan.total_steps,
            "last_chunk": int(
                (load_train_checkpoint_meta(adapter_dir) or {}).get("last_chunk") or 0
            ),
            "chunk_steps": plan.chunk_steps,
            "wired_limit_gb": None,
            "peak_mem_gb": None,
            "max_seq_length": max_seq_length,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "adapter_file": str(adapter_dir / "adapters.safetensors"),
            "resume": plan.resume,
            "already_completed": plan.already_completed,
            "steps_this_run": 0,
        }
        write_train_checkpoint_meta(adapter_dir, meta)
        if progress_callback:
            progress_callback({"kind": "done", **meta})
        return meta

    # Explicit memory_gb: skip mlx.device_info() entirely (sandbox-safe for unit tests).
    if memory_gb is not None and memory_gb > 0:
        mem_size, max_rec = 48 * 10**9, 40 * 10**9
    else:
        mem_size, max_rec = detect_device_memory()
    wired = resolve_wired_limit_bytes(
        memory_size=mem_size,
        max_recommended=max_rec,
        memory_gb=memory_gb,
    )

    adapter_file = adapter_dir / "adapters.safetensors"
    completed = plan.already_completed
    peaks: list[float] = []
    prior_chunks_done = 0
    if plan.resume:
        prior_chunks_done = int(
            (load_train_checkpoint_meta(adapter_dir) or {}).get("last_chunk") or 0
        )

    if progress_callback:
        progress_callback(
            {
                "kind": "start",
                "total_steps": plan.total_steps,
                "completed_steps": completed,
                "chunks": len(chunks),
                "chunk_steps": chunk_steps,
                "wired_limit_gb": round(wired / 1e9, 2),
                "max_seq_length": max_seq_length,
                "resume": plan.resume,
                "auto_resumed": plan.auto_resumed,
            }
        )

    for i, n_iters in enumerate(chunks, start=1):
        resume_adapter = adapter_file if adapter_file.is_file() else None
        chunk_index = prior_chunks_done + i
        total_chunk_count = prior_chunks_done + len(chunks)
        # Preserve any good weights before this chunk can overwrite them.
        had_good_snapshot = snapshot_last_good_adapter(adapter_dir) is not None
        if progress_callback:
            progress_callback(
                {
                    "kind": "chunk_start",
                    "chunk": chunk_index,
                    "chunks": total_chunk_count,
                    "chunk_iters": n_iters,
                    "completed_steps": completed,
                    "total_steps": plan.total_steps,
                }
            )

        def _on_line(line: str, *, _completed=completed, _n=n_iters) -> None:
            it = parse_iter_from_line(line)
            if it is None or progress_callback is None:
                return
            progress_callback(
                {
                    "kind": "step",
                    "global_step": _completed + it,
                    "total_steps": plan.total_steps,
                    "chunk_iter": it,
                    "chunk_iters": _n,
                    "line": line.strip(),
                }
            )

        result = run_mlx_chunk_subprocess(
            model=model,
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            iters=n_iters,
            wired_limit_bytes=wired,
            resume_adapter=resume_adapter,
            max_seq_length=max_seq_length,
            num_layers=num_layers,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            on_line=_on_line,
        )
        if result.peak_mem_gb is not None:
            peaks.append(result.peak_mem_gb)

        if (
            result.returncode != 0
            or not result.adapters_ok
            or stdout_reports_nan_loss(result.stdout)
        ):
            err_tail = result.stdout[-2000:] if result.stdout else ""
            nan_loss = stdout_reports_nan_loss(result.stdout)
            restored = False
            if had_good_snapshot:
                restored = restore_last_good_adapter(adapter_dir)
            # Persist partial progress so --resume can continue from last good steps.
            write_train_checkpoint_meta(
                adapter_dir,
                {
                    "status": "error",
                    "completed_steps": completed,
                    "total_steps": plan.total_steps,
                    "chunk_plan": plan.full_chunk_plan,
                    "last_chunk": prior_chunks_done + i - 1,
                    "chunk_steps": plan.chunk_steps,
                    "wired_limit_gb": round(wired / 1e9, 2),
                    "peak_mem_gb": max(peaks) if peaks else None,
                    "max_seq_length": max_seq_length,
                    "num_layers": num_layers,
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                    "adapter_file": str(adapter_file),
                    "resume": plan.resume,
                    "already_completed": plan.already_completed,
                    "error_chunk": chunk_index,
                    "returncode": result.returncode,
                    "nan_loss": nan_loss,
                    "restored_last_good": restored,
                },
            )
            if progress_callback:
                progress_callback(
                    {
                        "kind": "error",
                        "chunk": chunk_index,
                        "returncode": result.returncode,
                        "nan_loss": nan_loss,
                        "restored_last_good": restored,
                        "detail": err_tail,
                    }
                )
            sig_hint = ""
            if result.returncode < 0:
                import signal

                try:
                    signame = signal.Signals(-result.returncode).name
                except ValueError:
                    signame = f"signal_{-result.returncode}"
                sig_hint = (
                    f" Process killed by {signame} (exit={result.returncode}). "
                    "If SIGTERM: another agent/script likely pkill'd the worker — "
                    "do not restart-kill mid-train. If SIGKILL: check jetsam/OOM. "
                    "Re-run with --resume to continue from the last good chunk."
                )
            nan_hint = ""
            if nan_loss:
                nan_hint = (
                    f" Train loss nan under mask_prompt/max_seq_length={max_seq_length} "
                    "(usually oversized SFT examples → empty assistant labels)."
                )
            restore_hint = ""
            if restored:
                restore_hint = (
                    " Restored adapters.safetensors from pre-chunk snapshot "
                    f"({LAST_GOOD_ADAPTER_NAME}); good weights were not wiped."
                )
            elif had_good_snapshot:
                restore_hint = " Pre-chunk snapshot restore failed."
            raise RuntimeError(
                f"MLX train chunk {chunk_index}/{total_chunk_count} failed "
                f"(exit={result.returncode}, adapters_ok={result.adapters_ok})."
                f"{nan_hint}{restore_hint}{sig_hint}\n"
                f"Wired limit was {wired / 1e9:.1f} GB (capped; mlx-lm alone would "
                f"try ~{max_rec / 1e9:.0f} GB and can kill the Mac).\n"
                f"Completed steps so far: {completed}/{plan.total_steps}. "
                f"Re-run with --resume.\n"
                f"Last output:\n{err_tail}"
            )

        completed += n_iters
        status = "complete" if completed >= plan.total_steps else "in_progress"
        step_ckpt = persist_step_checkpoint(adapter_dir, completed)
        chunk_meta = {
            "status": status,
            "completed_steps": completed,
            "total_steps": plan.total_steps,
            "chunk_plan": plan.full_chunk_plan,
            "last_chunk": chunk_index,
            "chunk_steps": plan.chunk_steps,
            "wired_limit_gb": round(wired / 1e9, 2),
            "peak_mem_gb": max(peaks) if peaks else None,
            "max_seq_length": max_seq_length,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "adapter_file": str(adapter_file),
            "resume": plan.resume,
            "already_completed": plan.already_completed,
            "steps_this_run": completed - plan.already_completed,
            "chunks": total_chunk_count,
            "step_checkpoint": str(step_ckpt) if step_ckpt is not None else None,
        }
        write_train_checkpoint_meta(adapter_dir, chunk_meta)

        if progress_callback:
            progress_callback(
                {
                    "kind": "chunk_done",
                    "chunk": chunk_index,
                    "chunks": total_chunk_count,
                    "completed_steps": completed,
                    "total_steps": plan.total_steps,
                    "peak_mem_gb": result.peak_mem_gb,
                    "step_checkpoint": str(step_ckpt) if step_ckpt is not None else None,
                }
            )

    meta = load_train_checkpoint_meta(adapter_dir) or {}
    meta.update(
        {
            "status": "complete",
            "completed_steps": completed,
            "total_steps": plan.total_steps,
            "chunk_plan": plan.full_chunk_plan,
            "wired_limit_gb": round(wired / 1e9, 2),
            "peak_mem_gb": max(peaks) if peaks else None,
            "max_seq_length": max_seq_length,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "adapter_file": str(adapter_file),
            "resume": plan.resume,
            "already_completed": plan.already_completed,
            "steps_this_run": completed - plan.already_completed,
            "chunks": prior_chunks_done + len(chunks),
        }
    )
    write_train_checkpoint_meta(adapter_dir, meta)
    if progress_callback:
        progress_callback({"kind": "done", **meta, "total_steps": plan.total_steps})
    return meta
