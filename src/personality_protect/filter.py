"""Rewrite drafts with a local voice adapter (llama.cpp GGUF / MLX / PEFT / mock)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from personality_protect.config import (
    DEFAULT_MLX_MODEL,
    ProfilePaths,
    load_config,
)
from personality_protect.download import resolve_gguf_path

FilterBackend = Literal["auto", "llama", "gguf", "mlx", "transformers", "mock"]


def _latest_adapter(paths: ProfilePaths) -> Path:
    adapter = paths.adapters_dir / "latest"
    if not adapter.is_dir():
        raise FileNotFoundError(
            f"No adapter at {adapter}. Run: personality-protect train"
        )
    return adapter


def _load_mock_adapter(adapter_dir: Path) -> dict | None:
    path = adapter_dir / "mock_adapter.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _has_peft_adapter(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").is_file()


def _has_mlx_adapter(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapters.safetensors").is_file()


def filter_draft(
    draft: str,
    paths: ProfilePaths,
    *,
    backend: FilterBackend = "auto",
    max_tokens: int = 512,
    gguf: Path | None = None,
) -> tuple[str, str]:
    """Return (rewritten_text, backend_used).

    Auto preference: mock-only adapter → mock; else local GGUF/llama.cpp →
    MLX 4-bit → transformers → mock.
    """
    config = load_config(paths)
    adapter_dir = _latest_adapter(paths)
    draft = draft.strip()
    if not draft:
        return "", "none"

    chosen: str = backend
    if backend == "auto":
        mock_only = (
            _load_mock_adapter(adapter_dir) is not None
            and not _has_peft_adapter(adapter_dir)
            and not _has_mlx_adapter(adapter_dir)
        )
        # Demo/smoke: mock adapter alone → no multi-GB runtime required
        if mock_only and resolve_gguf_path(paths, explicit=gguf) is None:
            chosen = "mock"
        else:
            gguf_path = resolve_gguf_path(paths, filename=config.gguf_file, explicit=gguf)
            if gguf_path is not None and _has_llama_cpp():
                chosen = "llama"
            else:
                try:
                    import mlx_lm  # noqa: F401

                    chosen = "mlx"
                except ImportError:
                    try:
                        import peft  # noqa: F401

                        chosen = "transformers"
                    except ImportError:
                        chosen = "mock"
    elif backend == "gguf":
        chosen = "llama"

    if chosen == "mock":
        return _filter_mock(draft, adapter_dir), "mock"
    if chosen == "llama":
        gguf_path = resolve_gguf_path(paths, filename=config.gguf_file, explicit=gguf)
        if gguf_path is None:
            raise RuntimeError(
                "No local GGUF found. Run: personality-protect download --format gguf\n"
                f"Expected under {paths.models_dir} (~5–7 GB quantized Qwen3.5-9B)."
            )
        return _filter_llama(draft, gguf_path, adapter_dir, max_tokens), "llama"
    if chosen == "mlx":
        model_id = config.base_model or DEFAULT_MLX_MODEL
        return _filter_mlx(draft, adapter_dir, model_id, max_tokens), "mlx"
    if chosen == "transformers":
        return _filter_transformers(draft, adapter_dir, config.base_model, max_tokens), "transformers"
    raise RuntimeError(f"Unknown filter backend: {chosen}")


def _has_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa: F401

        return True
    except ImportError:
        return False


def _filter_mock(draft: str, adapter_dir: Path) -> str:
    """Deterministic local rewrite using style anchors — no model download."""
    meta = _load_mock_adapter(adapter_dir) or {}
    anchors: list[str] = list(meta.get("anchors") or [])

    # Prefer short sentences / cadence cues from anchors
    openers = []
    for a in anchors:
        first = a.strip().split("\n", 1)[0].strip()
        if 20 < len(first) < 160:
            openers.append(first)

    body = draft
    # Soften generic AI tells
    body = re.sub(
        r"\bIn today's (?:fast-paced\s+)?(?:digital\s+)?(?:fast-paced\s+)?world,?\s*",
        "",
        body,
        flags=re.I,
    )
    body = re.sub(r"\bIt is important to note that\s*", "", body, flags=re.I)
    body = re.sub(r"\bMoreover,\s*", "Also, ", body, flags=re.I)
    body = re.sub(r"\bFurthermore,\s*", "And ", body, flags=re.I)
    body = re.sub(r"\butilize\b", "use", body, flags=re.I)
    body = re.sub(r"\bleverage\b", "use", body, flags=re.I)
    body = re.sub(r"\brobust\b", "solid", body, flags=re.I)
    body = re.sub(r"\bsynergies\b", "strengths", body, flags=re.I)
    body = re.sub(r"\bdelve into\b", "look at", body, flags=re.I)
    body = re.sub(r"\s{2,}", " ", body).strip()
    # Fix leftover capitalization after stripping openers
    if body:
        body = body[0].upper() + body[1:]

    if openers:
        # Blend: keep meaning of draft, tip cadence toward user's first lines
        cue = openers[0]
        # If draft already starts strongly, just return cleaned body with a voice tag line
        if len(body) < 40:
            return f"{cue}\n\n{body}".strip()
        return (
            f"{body}\n\n"
            f"— rewritten locally with your voice anchors "
            f"({len(anchors)} samples; mock adapter)"
        ).strip()

    return (
        f"{body}\n\n"
        "— rewritten locally (mock adapter; train a real LoRA for full voice match)"
    ).strip()


def _filter_llama(
    draft: str, gguf_path: Path, adapter_dir: Path, max_tokens: int
) -> str:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python required for GGUF filter. "
            'Install: pip install -e ".[gguf]" '
            "or use --backend mlx|mock"
        ) from exc

    prompt = (
        "Rewrite the draft in the user's authentic writing voice. "
        "Keep meaning; do not invent facts.\n\n"
        f"### Draft\n{draft}\n\n### Rewritten\n"
    )
    # Optional GGUF LoRA if present next to adapter
    lora_path = None
    for name in ("adapter.gguf", "lora.gguf"):
        candidate = adapter_dir / name
        if candidate.is_file():
            lora_path = str(candidate)
            break

    kwargs: dict = {
        "model_path": str(gguf_path),
        "n_ctx": 4096,
        "verbose": False,
    }
    if lora_path:
        kwargs["lora_path"] = lora_path

    llm = Llama(**kwargs)
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stop=["###", "</s>", "<|end|>"],
    )
    text = out["choices"][0]["text"] if out.get("choices") else str(out)
    return str(text).strip()


def _filter_mlx(draft: str, adapter_dir: Path, base_model: str, max_tokens: int) -> str:
    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm required for MLX filter. Install: pip install -e \".[mlx]\" "
            "or use --backend llama|mock"
        ) from exc

    adapter = str(adapter_dir) if _has_mlx_adapter(adapter_dir) else None
    # Prefer quantized MLX id from config (default ~6 GB 4-bit)
    model, tokenizer = load(base_model, adapter_path=adapter)
    prompt = (
        "Rewrite the draft in the user's authentic writing voice. "
        "Keep meaning; do not invent facts.\n\n"
        f"### Draft\n{draft}\n\n### Rewritten\n"
    )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return str(out).strip()


def _filter_transformers(
    draft: str, adapter_dir: Path, base_model: str, max_tokens: int
) -> str:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers+peft required. Install: pip install -e \".[train]\" "
            "or use --backend mock"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else base_model
    )
    # Prefer 4-bit load when bitsandbytes is available (avoids holding full BF16 in RAM)
    model_kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
    try:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, "bfloat16", torch.float16),
            bnb_4bit_quant_type="nf4",
        )
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if _has_peft_adapter(adapter_dir):
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    prompt = (
        "Rewrite the draft in the user's authentic writing voice. "
        "Keep meaning; do not invent facts.\n\n"
        f"### Draft\n{draft}\n\n### Rewritten\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "### Rewritten" in text:
        text = text.split("### Rewritten", 1)[1].strip()
    return text.strip()


def read_draft_input(text: str | None, file: Path | None) -> str:
    if text and file:
        raise ValueError("Pass either --text or --file, not both.")
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is not None:
        return text
    raise ValueError("Provide --text or --file.")
