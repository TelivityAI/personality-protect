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
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_SIZE_HINT,
    DEFAULT_MLX_MODEL,
    DEFAULT_MLX_SIZE_HINT,
    FULL_PRECISION_BASE_MODEL,
    ProfilePaths,
    load_config,
)
from personality_protect.sft import build_sft_from_profile

BackendName = Literal["auto", "mlx", "cuda", "cpu", "mock"]


@dataclass
class TrainResult:
    backend: str
    status: str
    adapter_dir: str
    base_model: str
    examples: int
    notes: str = ""
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_backend(preferred: BackendName = "auto") -> str:
    if preferred == "mock":
        return "mock"
    if preferred == "mlx":
        return "mlx" if _has_mlx() else "mock"
    if preferred == "cuda":
        return "cuda" if _has_cuda() else "mock"
    if preferred == "cpu":
        return "cpu"

    # auto
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"} and _has_mlx():
        return "mlx"
    if _has_cuda():
        return "cuda"
    # Honest: full 9B SFT on CPU is impractical
    if preferred == "auto":
        return "cpu"
    return preferred


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


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
            "Adapters stay under your profile only. "
            f"Inference default is GGUF Q4 ({DEFAULT_GGUF_SIZE_HINT}) via llama.cpp."
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
            "(multi-day / high RAM). Use --backend mock for pipeline smoke "
            "tests, or run on Apple Silicon (MLX 4-bit) / CUDA QLoRA. "
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
    max_steps: int = 100,
    mock: bool = False,
    sft_only: bool = False,
    force_rebuild_sft: bool = True,
) -> TrainResult:
    config = load_config(paths)
    if force_rebuild_sft or not paths.sft_jsonl.is_file():
        sft_path, n = build_sft_from_profile(paths)
    else:
        sft_path = paths.sft_jsonl
        n = sum(1 for _ in sft_path.open(encoding="utf-8") if _.strip())

    if sft_only:
        return TrainResult(
            backend="sft_only",
            status="sft_ready",
            adapter_dir=str(paths.adapters_dir),
            base_model=config.base_model,
            examples=n,
            notes=f"SFT JSONL written to {sft_path} ({n} examples). No weights trained.",
        )

    chosen = "mock" if mock else detect_backend(backend)
    adapter_dir = paths.adapters_dir / "latest"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model_id = _train_model_id(config.base_model, chosen)

    if chosen == "mock":
        result = _train_mock(paths, sft_path, adapter_dir, model_id, n)
    elif chosen == "mlx":
        result = _train_mlx(paths, sft_path, adapter_dir, model_id, n, max_steps)
    elif chosen == "cuda":
        result = _train_cuda(paths, sft_path, adapter_dir, model_id, n, max_steps)
    elif chosen == "cpu":
        result = TrainResult(
            backend="cpu",
            status="skipped",
            adapter_dir=str(adapter_dir),
            base_model=model_id,
            examples=n,
            notes=backend_docs("cpu")
            + f"\nSFT ready at {sft_path}. Re-run with --backend mock|mlx|cuda.",
        )
    else:
        raise RuntimeError(f"Unknown backend: {chosen}")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "result": result.to_dict(),
        "sft_path": str(sft_path),
    }
    paths.adapter_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (adapter_dir / "train_result.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return result


def _train_mock(
    paths: ProfilePaths,
    sft_path: Path,
    adapter_dir: Path,
    base_model: str,
    n: int,
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
        meta={"anchors": len(anchors)},
    )


def _train_mlx(
    paths: ProfilePaths,
    sft_path: Path,
    adapter_dir: Path,
    base_model: str,
    n: int,
    max_steps: int,
) -> TrainResult:
    """LoRA fine-tune via mlx-lm on a 4-bit MLX base (~6 GB), not full BF16."""
    try:
        from mlx_lm import lora as mlx_lora  # type: ignore
    except ImportError:
        return TrainResult(
            backend="mlx",
            status="missing_deps",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes=(
                "mlx-lm not installed. Run: pip install -e \".[mlx]\"\n"
                "Or use --backend mock for a pipeline smoke train."
            ),
        )

    data_dir = paths.sft_dir / "mlx_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sft_path, data_dir / "train.jsonl")

    try:
        import sys

        argv = [
            "lora",
            "--model",
            base_model,
            "--data",
            str(data_dir),
            "--train",
            "--batch-size",
            "1",
            "--iters",
            str(max(1, max_steps)),
            "--adapter-path",
            str(adapter_dir),
        ]
        old = sys.argv
        try:
            sys.argv = argv
            if hasattr(mlx_lora, "main"):
                mlx_lora.main()
            else:
                mock_result = _train_mock(paths, sft_path, adapter_dir, base_model, n)
                mock_result.notes = (
                    "mlx_lm.lora API mismatch; wrote mock adapter. "
                    "Check mlx-lm docs for your version and re-run train."
                )
                return mock_result
        finally:
            sys.argv = old
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
                "Training may briefly need more unified memory than the on-disk size. "
                "For a no-download smoke test: --backend mock"
            ),
        )

    return TrainResult(
        backend="mlx",
        status="ok",
        adapter_dir=str(adapter_dir),
        base_model=base_model,
        examples=n,
        notes=f"MLX LoRA adapter saved under {adapter_dir} (base={base_model})",
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
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
    except ImportError as exc:
        return TrainResult(
            backend="cuda",
            status="missing_deps",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes=f"CUDA stack missing ({exc}). Install: pip install -e \".[cuda]\"",
        )

    if not torch.cuda.is_available():
        return TrainResult(
            backend="cuda",
            status="no_gpu",
            adapter_dir=str(adapter_dir),
            base_model=base_model,
            examples=n,
            notes="torch installed but CUDA not available.",
        )

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
        trainer.train()
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
        )

    return TrainResult(
        backend="cuda",
        status="ok",
        adapter_dir=str(adapter_dir),
        base_model=base_model,
        examples=n,
        notes=f"QLoRA adapter saved under {adapter_dir}",
    )
