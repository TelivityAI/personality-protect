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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# Digests that finish without melting a 48 GB unified-memory Mac.
DEFAULT_CHUNK_STEPS = 50
# 512 keeps peak ~14 GB on Qwen3.5-9B-4bit; 1024 can push past a 16–20 GB wired cap.
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_NUM_LAYERS = 8
# Cap wired Metal memory: leave OS/apps breathing room.
DEFAULT_WIRED_FRACTION = 0.40
DEFAULT_WIRED_CAP_BYTES = 20 * 10**9  # 20 GB hard cap
PROOF_MAX_STEPS = 150  # enough for real receipts without a marathon

ProgressCallback = Callable[[dict[str, Any]], None]

_ITER_RE = re.compile(r"^Iter\s+(\d+)\s*:", re.MULTILINE)
_PEAK_RE = re.compile(r"Peak mem\s+([0-9.]+)\s*GB", re.IGNORECASE)


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
) -> list[str]:
    """CLI argv for one mlx-lm LoRA chunk (memory-safe defaults)."""
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
        "1e-5",
    ]
    if resume_adapter is not None and resume_adapter.is_file():
        argv.extend(["--resume-adapter-file", str(resume_adapter)])
    return argv


def parse_iter_from_line(line: str) -> int | None:
    m = _ITER_RE.search(line.strip())
    return int(m.group(1)) if m else None


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
    """Return (memory_size, max_recommended_working_set_size) from MLX if possible."""
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
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train LoRA in subprocess chunks with progress events.

    By default starts fresh (deletes existing adapters). Pass ``resume=True`` to
    continue from ``adapter_dir/adapters.safetensors`` across a new CLI invocation
    (chunks within one run always resume between subprocesses).
    """
    chunks = plan_train_chunks(total_steps, chunk_steps)
    if not chunks:
        raise ValueError("total_steps must be >= 1")

    mem_size, max_rec = detect_device_memory()
    wired = resolve_wired_limit_bytes(
        memory_size=mem_size,
        max_recommended=max_rec,
        memory_gb=memory_gb,
    )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapters.safetensors"
    if resume:
        if not adapter_file.is_file():
            raise FileNotFoundError(
                f"--resume requested but no adapter at {adapter_file}. "
                "Run a fresh train first, or omit --resume."
            )
    else:
        # Fresh run: drop stale adapter so we don't silently resume old weights
        # unless a prior chunk in THIS run wrote them.
        if adapter_file.is_file():
            adapter_file.unlink()
        for stale in adapter_dir.glob("*_adapters.safetensors"):
            stale.unlink()

    completed = 0
    peaks: list[float] = []
    if progress_callback:
        progress_callback(
            {
                "kind": "start",
                "total_steps": total_steps,
                "chunks": len(chunks),
                "chunk_steps": chunk_steps,
                "wired_limit_gb": round(wired / 1e9, 2),
                "max_seq_length": max_seq_length,
                "resume": resume and adapter_file.is_file(),
            }
        )

    for i, n_iters in enumerate(chunks, start=1):
        resume = adapter_file if adapter_file.is_file() else None
        if progress_callback:
            progress_callback(
                {
                    "kind": "chunk_start",
                    "chunk": i,
                    "chunks": len(chunks),
                    "chunk_iters": n_iters,
                    "completed_steps": completed,
                    "total_steps": total_steps,
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
                    "total_steps": total_steps,
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
            resume_adapter=resume,
            max_seq_length=max_seq_length,
            num_layers=num_layers,
            on_line=_on_line,
        )
        if result.peak_mem_gb is not None:
            peaks.append(result.peak_mem_gb)

        if result.returncode != 0 or not result.adapters_ok:
            err_tail = result.stdout[-2000:] if result.stdout else ""
            if progress_callback:
                progress_callback(
                    {
                        "kind": "error",
                        "chunk": i,
                        "returncode": result.returncode,
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
                    "do not restart-kill mid-train. If SIGKILL: check jetsam/OOM."
                )
            raise RuntimeError(
                f"MLX train chunk {i}/{len(chunks)} failed "
                f"(exit={result.returncode}, adapters_ok={result.adapters_ok})."
                f"{sig_hint}\n"
                f"Wired limit was {wired / 1e9:.1f} GB (capped; mlx-lm alone would "
                f"try ~{max_rec / 1e9:.0f} GB and can kill the Mac).\n"
                f"Last output:\n{err_tail}"
            )

        completed += n_iters
        if progress_callback:
            progress_callback(
                {
                    "kind": "chunk_done",
                    "chunk": i,
                    "chunks": len(chunks),
                    "completed_steps": completed,
                    "total_steps": total_steps,
                    "peak_mem_gb": result.peak_mem_gb,
                }
            )

    meta = {
        "chunks": len(chunks),
        "chunk_plan": chunks,
        "wired_limit_gb": round(wired / 1e9, 2),
        "peak_mem_gb": max(peaks) if peaks else None,
        "max_seq_length": max_seq_length,
        "num_layers": num_layers,
        "adapter_file": str(adapter_file),
        "resume": bool(resume),
        "steps_this_run": total_steps,
    }
    if progress_callback:
        progress_callback({"kind": "done", **meta, "total_steps": total_steps})
    # Persist chunk meta next to adapters for receipts
    (adapter_dir / "train_chunks.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta
