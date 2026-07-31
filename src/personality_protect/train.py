"""Local LoRA / QLoRA training backends (MLX 4-bit, CUDA QLoRA, mock/smoke)."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from personality_protect.config import (
    CORPUS_BLOCK_BELOW,
    CORPUS_WARN_BELOW,
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_SIZE_HINT,
    DEFAULT_MLX_MODEL,
    DEFAULT_MLX_SIZE_HINT,
    FULL_PRECISION_BASE_MODEL,
    SMOKE_MAX_STEPS,
    ProfilePaths,
    load_config,
)
from personality_protect.mlx_train import (
    DEFAULT_CHUNK_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_NUM_LAYERS,
    PROOF_MAX_STEPS,
    WRITER_EPOCHS,
    WRITER_LEARNING_RATE,
    WRITER_LORA_RANK,
    WRITER_NUM_LAYERS,
    ProgressCallback,
    run_chunked_mlx_train,
)
from personality_protect.select import selected_pieces
from personality_protect.sft import build_sft_from_pairs, build_sft_from_profile

BackendName = Literal["auto", "mlx", "cuda", "cpu", "mock"]

# Full-train defaults sized for hundreds of SFT examples (batch size 1).
DEFAULT_EPOCHS = 3
MIN_AUTO_STEPS = 50
MAX_AUTO_STEPS = 2000


@dataclass
class TrainResult:
    backend: str
    status: str
    adapter_dir: str
    base_model: str
    examples: int
    notes: str = ""
    meta: dict[str, Any] | None = None
    steps: int = 0
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MockFallbackError(RuntimeError):
    """Raised when a real backend would silently degrade to mock."""


def auto_max_steps(
    n_examples: int,
    *,
    smoke: bool = False,
    max_steps: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> int:
    """Resolve train steps: explicit override, smoke low-step, or auto from corpus size."""
    if max_steps is not None and max_steps > 0:
        return max_steps
    if smoke:
        return SMOKE_MAX_STEPS
    # ~epochs passes over the JSONL at batch size 1, clamped for tiny/huge corpora
    n = max(1, int(n_examples))
    return max(MIN_AUTO_STEPS, min(MAX_AUTO_STEPS, n * max(1, int(epochs))))


def writer_train_settings() -> dict[str, Any]:
    """LoRA hyperparameters for the writer recipe.

    Separated from the translator defaults because the two tasks are not the
    same size of change. Translation edits a draft it is already given; writing
    a post from a note has to produce the whole text, which needs more adapted
    layers and more rank than the 8/8 the first run inherited.
    """
    return {
        "num_layers": WRITER_NUM_LAYERS,
        "lora_rank": WRITER_LORA_RANK,
        "learning_rate": WRITER_LEARNING_RATE,
        "epochs": WRITER_EPOCHS,
    }


def check_corpus_size(n_selected: int, *, force: bool = False, smoke: bool = False) -> str | None:
    """Warn/block on selected *pieces* (not SFT row count).

    Synthetic Contoso pairs are always appended to SFT JSONL and must not pad
    a tiny real corpus past the gate. Returns a warning string (or None).
    Raises RuntimeError when blocked.
    """
    if smoke or force:
        if n_selected < CORPUS_WARN_BELOW:
            return (
                f"Corpus has {n_selected} selected pieces "
                f"(warn threshold {CORPUS_WARN_BELOW}); proceeding due to "
                f"{'smoke' if smoke else 'force'}."
            )
        return None
    if n_selected < CORPUS_BLOCK_BELOW:
        raise RuntimeError(
            f"Corpus too small for full train: {n_selected} selected pieces "
            f"(need at least {CORPUS_BLOCK_BELOW}). "
            "Ingest more writing, or pass --force / --smoke for a deliberate override."
        )
    if n_selected < CORPUS_WARN_BELOW:
        return (
            f"Warning: only {n_selected} selected pieces "
            f"(recommend >={CORPUS_WARN_BELOW} for a credible voice adapter)."
        )
    return None


def detect_backend(
    preferred: BackendName = "auto",
    *,
    allow_mock: bool = False,
) -> str:
    """Pick a train backend. Never silently falls back to mock unless allow_mock."""
    if preferred == "mock":
        return "mock"

    if preferred == "mlx":
        if _has_mlx():
            return "mlx"
        if allow_mock:
            return "mock"
        raise MockFallbackError(
            "MLX backend requested but mlx/mlx-lm is not available. "
            'Install: pip install -e ".[mlx]" — or pass --allow-mock / --backend mock '
            "for CI/smoke only."
        )

    if preferred == "cuda":
        if _has_cuda():
            return "cuda"
        if allow_mock:
            return "mock"
        raise MockFallbackError(
            "CUDA backend requested but torch CUDA is not available. "
            'Install: pip install -e ".[cuda]" on an NVIDIA machine — '
            "or pass --allow-mock / --backend mock for CI/smoke only."
        )

    if preferred == "cpu":
        return "cpu"

    # auto
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"} and _has_mlx():
        return "mlx"
    if _has_cuda():
        return "cuda"
    # Honest: full 9B SFT on CPU is impractical — do not pretend mock is a real train
    return "cpu"


def _has_mlx() -> bool:
    """True when mlx + mlx_lm are importable (package present)."""
    import importlib.util

    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _train_model_id(config_base: str, backend: str) -> str:
    """Prefer quantized train ids; never silently default to full BF16."""
    if backend == "mlx":
        # Profile may still hold an old full-precision id — coerce to 4-bit happy path
        if config_base in {"", FULL_PRECISION_BASE_MODEL, "Qwen/Qwen3.5-9B"}:
            return DEFAULT_MLX_MODEL
        return config_base or DEFAULT_MLX_MODEL
    if backend == "cuda":
        # QLoRA loads 4-bit in VRAM; HF may still fetch shards. Prefer documented path.
        return config_base or FULL_PRECISION_BASE_MODEL
    return config_base or DEFAULT_BASE_MODEL


def backend_docs(backend: str) -> str:
    if backend == "mlx":
        return (
            "Apple Silicon MLX path (train-from-quantized).\n"
            f"Default model: {DEFAULT_MLX_MODEL} ({DEFAULT_MLX_SIZE_HINT} on disk)\n"
            "Install: pip install -e \".[mlx]\" then: "
            "personality-protect download --format mlx\n"
            "Train runs in memory-capped subprocess chunks (CLI progress bar) "
            "so a 48 GB Mac stays usable — stock mlx-lm can wire ~40 GB and "
            "jetsam-kill Python.\n"
            "Adapters stay under your profile only. "
            f"Inference default is GGUF Q4 ({DEFAULT_GGUF_SIZE_HINT}) via llama.cpp.\n"
            "Receipts without a marathon: personality-protect train --proof"
        )
    if backend == "cuda":
        return (
            "CUDA QLoRA path (4-bit in VRAM).\n"
            f"Train base: {FULL_PRECISION_BASE_MODEL} loaded with bitsandbytes 4-bit.\n"
            "Install: pip install -e \".[cuda]\"\n"
            "Disk: prefer the GGUF runtime download (~5–7 GB) for filter; "
            "HF may cache extra shards during first train — RAM/disk spike possible, "
            "but the product download target remains the quantized GGUF/MLX artifact.\n"
            "Requires NVIDIA GPU (24GB+ VRAM recommended for 9B QLoRA)."
        )
    if backend == "cpu":
        return (
            "CPU training of Qwen3.5-9B is not practical for v1 "
            "(multi-day / high RAM). Use --backend mock or --smoke --allow-mock for "
            "pipeline smoke tests, or run on Apple Silicon (MLX 4-bit) / CUDA QLoRA. "
            "You can still build SFT JSONL with: personality-protect train --sft-only\n"
            f"For local filter without training GPU: download GGUF ({DEFAULT_GGUF_SIZE_HINT})."
        )
    if backend == "mock":
        return (
            "Mock/smoke train: writes a tiny local adapter stub from your SFT JSONL "
            "without downloading any multi-GB model. Used by `demo` and CI. "
            "Not a real voice model — prove the pipeline, then train for real "
            f"(MLX 4-bit ~{DEFAULT_MLX_SIZE_HINT} or CUDA QLoRA)."
        )
    return ""


def run_train(
    paths: ProfilePaths,
    *,
    backend: BackendName = "auto",
    max_steps: int | None = None,
    mock: bool = False,
    smoke: bool = False,
    allow_mock: bool = False,
    force: bool = False,
    sft_only: bool = False,
    force_rebuild_sft: bool = True,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
    memory_gb: float | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    proof: bool = False,
    resume: bool = False,
    force_retrain: bool = False,
    progress_callback: ProgressCallback | None = None,
    pairs: Path | None = None,
    writer: bool = False,
    num_layers: int | None = None,
    lora_rank: int | None = None,
    learning_rate: float | None = None,
) -> TrainResult:
    config = load_config(paths)
    if writer and pairs is not None:
        raise ValueError("Pass only one of --writer or --pairs")
    recipe = writer_train_settings() if writer else {}
    resolved_layers = (
        num_layers if num_layers is not None else recipe.get("num_layers", DEFAULT_NUM_LAYERS)
    )
    resolved_rank = (
        lora_rank if lora_rank is not None else recipe.get("lora_rank", DEFAULT_LORA_RANK)
    )
    resolved_lr = (
        learning_rate
        if learning_rate is not None
        else recipe.get("learning_rate", DEFAULT_LEARNING_RATE)
    )
    voice_pair_mode = pairs is not None
    if voice_pair_mode:
        # Gated flatten→author pairs are the data floor; skip selected-piece gate.
        corpus_note = None
    elif writer:
        corpus_note = None
    else:
        n_selected = len(selected_pieces(paths))
        corpus_note = check_corpus_size(
            n_selected, force=force or sft_only, smoke=smoke or mock
        )

    if writer:
        from personality_protect.writer_sft import run_build_writer_sft, writer_sft_path

        receipt = run_build_writer_sft(paths)
        sft_path = writer_sft_path(paths)
        # Train pipeline expects train.jsonl under sft/; copy writer rows there.
        paths.sft_jsonl.write_text(sft_path.read_text(encoding="utf-8"), encoding="utf-8")
        sft_path = paths.sft_jsonl
        n = int(receipt["examples"])
        if n < 1:
            raise FileNotFoundError(
                "Writer SFT produced 0 examples. Need longer linkedin_post pieces."
            )
    elif voice_pair_mode:
        assert pairs is not None  # for type checkers
        sft_path, n = build_sft_from_pairs(
            pairs,
            paths.sft_jsonl,
            max_seq_length=max_seq_length,
        )
    elif force_rebuild_sft or not paths.sft_jsonl.is_file():
        sft_path, n = build_sft_from_profile(paths)
    else:
        sft_path = paths.sft_jsonl
        n = sum(1 for _ in sft_path.open(encoding="utf-8") if _.strip())
    # --proof: real weights, bounded steps for receipts (not a silent mock).
    if proof and max_steps is None and not smoke and not mock:
        max_steps = PROOF_MAX_STEPS
    # Crash recovery: keep the original target when an incomplete checkpoint exists
    # and the user did not pass a new --max-steps.
    if max_steps is None and not force_retrain and not smoke and not mock:
        from personality_protect.mlx_train import (
            completed_steps_from_meta,
            is_incomplete_checkpoint,
            load_train_checkpoint_meta,
        )

        latest = paths.adapters_dir / "latest"
        if is_incomplete_checkpoint(latest) or resume:
            prior = load_train_checkpoint_meta(latest) or {}
            prior_total = int(prior.get("total_steps") or 0)
            done = completed_steps_from_meta(prior)
            if prior_total > done:
                max_steps = prior_total
    steps = auto_max_steps(
        n,
        smoke=smoke or mock,
        max_steps=max_steps,
        epochs=int(recipe.get("epochs", DEFAULT_EPOCHS)),
    )

    if sft_only:
        mode_note = (
            f" Voice-pair translator SFT from {pairs}."
            if voice_pair_mode
            else ""
        )
        result = TrainResult(
            backend="sft_only",
            status="sft_ready",
            adapter_dir=str(paths.adapters_dir),
            base_model=config.base_model,
            examples=n,
            notes=f"SFT JSONL written to {sft_path} ({n} examples). No weights trained."
            + mode_note
            + (f"\n{corpus_note}" if corpus_note else ""),
            steps=0,
            smoke=smoke,
        )
        _persist_train_meta(paths, result, sft_path, steps=0, corpus_note=corpus_note)
        return result

    # Explicit mock / --mock → mock path. Smoke alone is low-step, not mock.
    want_mock = mock or backend == "mock"
    if want_mock:
        chosen = "mock"
    else:
        try:
            chosen = detect_backend(
                backend,
                allow_mock=allow_mock,
            )
        except MockFallbackError:
            raise

    # If auto landed on something that isn't mock, but mlx/cuda was unavailable
    # and detect returned cpu — that's honest, not a silent mock fallthrough.
    if chosen == "mock" and not (mock or backend == "mock" or allow_mock):
        raise MockFallbackError(
            "Refusing silent mock fallback. Pass --backend mock, --allow-mock, "
            "or use --smoke only for low-step real backends."
        )

    adapter_dir = paths.adapters_dir / "latest"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model_id = _train_model_id(config.base_model, chosen)

    if chosen == "mock":
        result = _train_mock(paths, sft_path, adapter_dir, model_id, n, steps=steps, smoke=smoke)
    elif chosen == "mlx":
        result = _train_mlx(
            paths,
            sft_path,
            adapter_dir,
            model_id,
            n,
            steps,
            allow_mock=allow_mock,
            chunk_steps=chunk_steps,
            memory_gb=memory_gb,
            max_seq_length=max_seq_length,
            num_layers=resolved_layers,
            lora_rank=resolved_rank,
            learning_rate=resolved_lr,
            progress_callback=progress_callback,
            proof=proof,
            resume=resume,
            force_retrain=force_retrain,
        )
    elif chosen == "cuda":
        result = _train_cuda(paths, sft_path, adapter_dir, model_id, n, steps)
    elif chosen == "cpu":
        result = TrainResult(
            backend="cpu",
            status="skipped",
            adapter_dir=str(adapter_dir),
            base_model=model_id,
            examples=n,
            notes=backend_docs("cpu")
            + f"\nSFT ready at {sft_path}. Re-run with --backend mock|mlx|cuda.",
            steps=0,
            smoke=smoke,
        )
    else:
        raise RuntimeError(f"Unknown backend: {chosen}")

    if corpus_note:
        result.notes = (result.notes + "\n" + corpus_note).strip()
    result.steps = result.steps or steps
    result.smoke = smoke or result.smoke

    _persist_train_meta(paths, result, sft_path, steps=result.steps, corpus_note=corpus_note)
    return result


def _persist_train_meta(
    paths: ProfilePaths,
    result: TrainResult,
    sft_path: Path,
    *,
    steps: int,
    corpus_note: str | None,
) -> None:
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "backend": result.backend,
        "status": result.status,
        "base_model": result.base_model,
        "examples": result.examples,
        "steps": steps,
        "smoke": result.smoke,
        "adapter_dir": result.adapter_dir,
        "sft_path": str(sft_path),
        "corpus_note": corpus_note,
        "result": result.to_dict(),
        "loss": (result.meta or {}).get("loss"),
    }
    paths.adapters_dir.mkdir(parents=True, exist_ok=True)
    paths.adapter_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    adapter_dir = Path(result.adapter_dir)
    if adapter_dir.is_dir() or result.backend == "sft_only":
        target = adapter_dir if adapter_dir.is_dir() else paths.adapters_dir
        (target / "train_result.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


def _train_mock(
    paths: ProfilePaths,
    sft_path: Path,
    adapter_dir: Path,
    base_model: str,
    n: int,
    *,
    steps: int = SMOKE_MAX_STEPS,
    smoke: bool = False,
) -> TrainResult:
    """Write a deterministic adapter stub + voice fingerprint for local filter smoke."""
    anchors: list[str] = []
    with sft_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages") or []
            for m in msgs:
                if m.get("role") == "assistant" and m.get("content"):
                    anchors.append(m["content"])
                    break
            if len(anchors) >= 8:
                break

    fingerprint = {
        "type": "mock_voice_adapter",
        "base_model": base_model,
        "examples": n,
        "steps": steps,
        "smoke": smoke,
        "anchors": anchors,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Synthetic/local stub — not real LoRA weights.",
    }
    (adapter_dir / "mock_adapter.json").write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name in ("adapter_config.json", "adapters.safetensors"):
        p = adapter_dir / name
        if p.exists():
            p.unlink()

    return TrainResult(
        backend="mock",
        status="ok",
        adapter_dir=str(adapter_dir),
        base_model=base_model,
        examples=n,
        notes="Mock adapter written. Filter will use style anchors (no model download).",
        meta={"anchors": len(anchors), "loss": None},
        steps=steps,
        smoke=smoke,
    )


def _train_mlx(
    paths: ProfilePaths,
    sft_path: Path,
    adapter_dir: Path,
    base_model: str,
    n: int,
    max_steps: int,
    *,
    allow_mock: bool = False,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
    memory_gb: float | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    num_layers: int = DEFAULT_NUM_LAYERS,
    lora_rank: int = DEFAULT_LORA_RANK,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    progress_callback: ProgressCallback | None = None,
    proof: bool = False,
    resume: bool = False,
    force_retrain: bool = False,
) -> TrainResult:
    """LoRA fine-tune via chunked, memory-capped MLX subprocesses (not full BF16)."""
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:
        if allow_mock:
            mock_result = _train_mock(
                paths, sft_path, adapter_dir, base_model, n, steps=max_steps, smoke=True
            )
            mock_result.notes = (
                "mlx-lm not installed; wrote mock adapter because --allow-mock/--smoke. "
                'Install for real train: pip install -e ".[mlx]"'
            )
            return mock_result
        return TrainResult(
            backend="mlx",
            status="missing_deps",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes=(
                "mlx-lm not installed. Run: pip install -e \".[mlx]\"\n"
                "Or use --backend mock / --allow-mock for a pipeline smoke train."
            ),
            steps=0,
        )

    data_dir = paths.sft_dir / "mlx_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sft_path, data_dir / "train.jsonl")

    try:
        meta = run_chunked_mlx_train(
            model=base_model,
            data_dir=data_dir,
            adapter_dir=adapter_dir,
            total_steps=max(1, max_steps),
            chunk_steps=chunk_steps,
            num_layers=num_layers,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            memory_gb=memory_gb,
            max_seq_length=max_seq_length,
            resume=resume,
            force_retrain=force_retrain,
            progress_callback=progress_callback,
        )
    except Exception as exc:  # noqa: BLE001 — surface train errors honestly
        return TrainResult(
            backend="mlx",
            status="error",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes=(
                f"MLX train failed: {exc}\n"
                f"Expected quantized base {DEFAULT_MLX_MODEL} ({DEFAULT_MLX_SIZE_HINT}). "
                "Prefetch with: personality-protect download --format mlx\n"
                "Train uses memory-capped subprocess chunks so a 48 GB Mac stays usable. "
                "Try: --proof (bounded steps) or --memory-gb 16 --chunk-steps 25\n"
                "Crash mid-train? Re-run with --resume (keeps adapters + completed_steps).\n"
                "For a no-download smoke test: --backend mock"
            ),
            steps=0,
            meta={"error": str(exc)},
        )

    proof_note = " [proof mode]" if proof else ""
    resume_note = " [resumed]" if resume else ""
    retrain_note = " [force-retrain]" if force_retrain else ""
    return TrainResult(
        backend="mlx",
        status="ok",
        adapter_dir=str(adapter_dir),
        base_model=base_model,
        examples=n,
        notes=(
            f"MLX LoRA adapter saved under {adapter_dir} "
            f"(base={base_model}, steps={meta.get('completed_steps', max_steps)}, "
            f"chunks={meta.get('chunks')}, "
            f"wired_cap={meta.get('wired_limit_gb')} GB)"
            f"{proof_note}{resume_note}{retrain_note}"
        ),
        meta={"loss": None, **meta},
        steps=int(meta.get("completed_steps") or max_steps),
    )


def _train_cuda(
    paths: ProfilePaths,
    sft_path: Path,
    adapter_dir: Path,
    base_model: str,
    n: int,
    max_steps: int,
) -> TrainResult:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        return TrainResult(
            backend="cuda",
            status="missing_deps",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes=f"CUDA stack missing ({exc}). Install: pip install -e \".[cuda]\"",
            steps=0,
        )

    if not torch.cuda.is_available():
        return TrainResult(
            backend="cuda",
            status="no_gpu",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes="torch installed but CUDA not available.",
            steps=0,
        )

    train_loss: float | None = None
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)

        def to_text(example: dict) -> dict:
            msgs = example.get("messages") or []
            parts = []
            for m in msgs:
                parts.append(f"<|{m.get('role')}|>\n{m.get('content', '')}")
            return {"text": "\n".join(parts)}

        ds = load_dataset("json", data_files=str(sft_path), split="train")
        ds = ds.map(to_text)

        def tokenize(batch: dict) -> dict:
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=1024,
                padding="max_length",
            )

        tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        args = TrainingArguments(
            output_dir=str(adapter_dir / "runs"),
            max_steps=max(1, max_steps),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=max(1, max_steps),
            fp16=True,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        train_out = trainer.train()
        try:
            train_loss = float(train_out.training_loss)  # type: ignore[attr-defined]
        except Exception:
            train_loss = None
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    except Exception as exc:  # noqa: BLE001
        return TrainResult(
            backend="cuda",
            status="error",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes=(
                f"CUDA QLoRA train failed: {exc}\n"
                "First run may need extra disk while HF caches shards; "
                f"day-to-day runtime download target remains GGUF ({DEFAULT_GGUF_SIZE_HINT})."
            ),
            steps=0,
        )

    return TrainResult(
        backend="cuda",
        status="ok",
        adapter_dir=str(adapter_dir),
        base_model=base_model,
        examples=n,
        notes=f"QLoRA adapter saved under {adapter_dir} (steps={max_steps})",
        meta={"loss": train_loss},
        steps=max_steps,
    )
