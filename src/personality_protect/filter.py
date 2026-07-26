"""Rewrite drafts with a local voice adapter (llama.cpp GGUF / MLX / PEFT / mock)."""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path
from typing import Literal

from personality_protect.config import (
    DEFAULT_MLX_MODEL,
    ProfilePaths,
    load_config,
)
from personality_protect.download import resolve_gguf_path
from personality_protect.sft import SYSTEM_PROMPT, USER_TEMPLATE, USER_TEMPLATE_INFER

FilterBackend = Literal["auto", "llama", "gguf", "mlx", "transformers", "mock"]

# Stable decoding for voice rewrite (deterministic-leaning, not creative chat)
FILTER_TEMPERATURE = 0.4
FILTER_TOP_P = 0.9
FILTER_REPEAT_PENALTY = 1.1
FILTER_STOP = ("###", "</s>", "<|end|>", "<|im_end|>", "<|im_start|>")
_TEMPLATE_SECTION_RE = re.compile(
    r"\n\s*###\s*(?:Draft|Rewritten|My voice(?:\s*\(reference\))?)\b",
    re.IGNORECASE,
)


def filter_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_filter_user_content(draft: str, *, reference: str | None = None) -> str:
    """User turn matching SFT shape so LoRA completes a rewrite, not a template loop."""
    draft = draft.strip()
    if reference and reference.strip():
        return USER_TEMPLATE.format(draft=draft, reference=reference.strip())
    return USER_TEMPLATE_INFER.format(draft=draft)


def build_filter_messages(
    draft: str, *, reference: str | None = None, few_shot: str | None = None
) -> list[dict[str, str]]:
    """Chat messages aligned with SFT training (system + user)."""
    user = build_filter_user_content(draft, reference=reference)
    if few_shot and few_shot.strip() and not (reference and reference.strip()):
        # Receipts / baseline: inject few-shot above the draft block.
        user = few_shot.strip() + "\n\n" + user
    return [
        {"role": "system", "content": filter_system_prompt()},
        {"role": "user", "content": user},
    ]


def build_filter_prompt(
    draft: str, *, few_shot: str | None = None, reference: str | None = None
) -> str:
    """Flat prompt for backends without chat templates (and compare receipts)."""
    messages = build_filter_messages(draft, reference=reference, few_shot=few_shot)
    parts = [messages[0]["content"], "", messages[1]["content"]]
    # Ensure trailing newline after ### Rewritten so generation continues cleanly.
    text = "\n".join(parts)
    if not text.endswith("\n"):
        text += "\n"
    return text


def strip_ai_tells(text: str) -> str:
    """Remove common AI-tell phrases the system prompt asks the model to strip."""
    body = text
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
    body = re.sub(r"\bsynergias\b", "strengths", body, flags=re.I)
    body = re.sub(r"\bdelve into\b", "look at", body, flags=re.I)
    body = re.sub(r"\bunlock(?:ing)?\b", "open", body, flags=re.I)
    body = re.sub(r"\bnestled\b", "in", body, flags=re.I)
    body = re.sub(r"\ba testament to\b", "proof of", body, flags=re.I)
    body = re.sub(r"\bvibrant\b", "lively", body, flags=re.I)
    body = re.sub(r"\s{2,}", " ", body).strip()
    if body:
        body = body[0].upper() + body[1:]
    return body


def finalize_rewrite(text: str) -> str:
    """Template-echo cut + AI-tell cleanup for model backends."""
    return strip_ai_tells(extract_rewrite(text))


def extract_rewrite(text: str) -> str:
    """Keep only the rewrite; drop thinking + SFT template echo loops."""
    text = (text or "").strip()
    if not text:
        return ""

    # Qwen / mlx-lm thinking blocks (tagged or plain "Thinking Process:")
    for end_tag in ("</think>", "</longcat_think>", "<channel|>"):
        if end_tag in text:
            text = text.split(end_tag, 1)[-1].strip()
    if re.match(r"(?is)^\s*(?:thinking\s+process|analysis)\s*:", text):
        # Drop meta-reasoning preamble; keep text after a blank line if present.
        parts = re.split(r"\n\s*\n", text, maxsplit=1)
        if len(parts) == 2 and not re.match(
            r"(?is)^\s*(?:thinking\s+process|analysis|\d+\.)", parts[1]
        ):
            text = parts[1].strip()
        else:
            # Entire output was thinking — treat as empty so callers see failure clearly.
            # Prefer content after the last numbered step block if a rewrite follows.
            m = re.search(
                r"(?is)(?:^|\n)(?!thinking\s+process|analysis|\d+\.\s)([A-Z].{20,})$",
                text,
            )
            text = m.group(1).strip() if m else ""

    # If the model echoed a leading header, drop it.
    if text.lower().startswith("### rewritten"):
        text = text.split("\n", 1)[1].strip() if "\n" in text else ""
    # Cut at the first subsequent template section.
    m = _TEMPLATE_SECTION_RE.search("\n" + text)
    if m:
        text = text[: m.start()].rstrip()
    for marker in ("\n### Draft", "\n### Rewritten", "\n### My voice"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].rstrip()
    return text.strip()


def _load_voice_reference(paths: ProfilePaths, *, max_chars: int = 400) -> str | None:
    """Short corpus snippet so inference matches the SFT 'My voice (reference)' block."""
    adapter = paths.adapters_dir / "latest" / "mock_adapter.json"
    if adapter.is_file():
        try:
            data = json.loads(adapter.read_text(encoding="utf-8"))
            for a in data.get("anchors") or []:
                t = str(a).strip()
                if len(t) > 40:
                    return t[:max_chars]
        except (OSError, json.JSONDecodeError):
            pass
    if paths.sft_jsonl.is_file():
        try:
            with paths.sft_jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for m in row.get("messages") or []:
                        if m.get("role") == "assistant" and m.get("content"):
                            t = m["content"].strip()
                            if len(t) > 40:
                                return t[:max_chars]
        except (OSError, json.JSONDecodeError):
            pass
    return None


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


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def _has_mlx() -> bool:
    """True when mlx_lm is importable (package present)."""
    import importlib.util

    return importlib.util.find_spec("mlx_lm") is not None


def filter_draft(
    draft: str,
    paths: ProfilePaths,
    *,
    backend: FilterBackend = "auto",
    max_tokens: int = 512,
    gguf: Path | None = None,
) -> tuple[str, str]:
    """Return (rewritten_text, backend_used).

    Auto preference on Apple Silicon with an MLX adapter: MLX first.
    Otherwise: mock-only adapter → mock; else local GGUF/llama.cpp →
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
        elif (
            _is_apple_silicon()
            and _has_mlx()
            and (_has_mlx_adapter(adapter_dir) or not mock_only)
        ):
            # Prefer MLX train+filter on Apple Silicon when adapter/runtime available
            chosen = "mlx"
        else:
            gguf_path = resolve_gguf_path(paths, filename=config.gguf_file, explicit=gguf)
            if gguf_path is not None and _has_llama_cpp():
                chosen = "llama"
            elif _has_mlx():
                chosen = "mlx"
            else:
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
        return _filter_mlx(draft, adapter_dir, model_id, max_tokens, paths=paths), "mlx"
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
    body = strip_ai_tells(body)
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

    # Prefer chat-aligned user content; paths not available here — no reference.
    prompt = build_filter_prompt(draft)
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
        temperature=FILTER_TEMPERATURE,
        top_p=FILTER_TOP_P,
        repeat_penalty=FILTER_REPEAT_PENALTY,
        stop=list(FILTER_STOP),
    )
    text = out["choices"][0]["text"] if out.get("choices") else str(out)
    return finalize_rewrite(str(text))


def _filter_mlx(
    draft: str,
    adapter_dir: Path,
    base_model: str,
    max_tokens: int,
    *,
    paths: ProfilePaths | None = None,
    use_adapter: bool = True,
    few_shot: str | None = None,
) -> str:
    # CRITICAL: mlx_lm.generate installs wired_limit(~40GB) which jetsam-kills
    # Python on a 48GB Mac. Cap BEFORE importing/loading. Default ≤16 GB.
    from personality_protect.mlx_runtime import ensure_mlx_wired_cap, release_mlx_memory

    ensure_mlx_wired_cap(memory_gb=16.0)

    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm required for MLX filter. Install: pip install -e \".[mlx]\" "
            "or use --backend llama|mock"
        ) from exc

    adapter = None
    if use_adapter and _has_mlx_adapter(adapter_dir):
        adapter = str(adapter_dir)
    # Do NOT paste a long SFT assistant snippet as "### My voice (reference)":
    # weak adapters regurgitate that corpus block instead of rewriting the draft.
    # LoRA weights carry voice; few-shot (prompt baseline) may inject anchors above.
    messages = build_filter_messages(draft, reference=None, few_shot=few_shot)

    try:
        model, tokenizer = load(base_model, adapter_path=adapter)
        # Match training: chat template + assistant generation prompt.
        if getattr(tokenizer, "has_chat_template", False) or getattr(
            tokenizer, "chat_template", None
        ):
            # Qwen3 defaults enable_thinking=True; that burns tokens on meta
            # "Thinking Process" and never returns a clean rewrite.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = build_filter_prompt(draft, reference=None, few_shot=few_shot)

        # Stop on template markers so we never loop Draft/Rewritten sections.
        for stop in FILTER_STOP:
            try:
                tokenizer.add_eos_token(stop)
            except Exception:
                pass

        # Keep generations short enough that cadence shows; avoid long echo loops.
        gen_max = min(max_tokens, 320)
        gen_kwargs: dict = {"max_tokens": gen_max, "verbose": False}
        try:
            from mlx_lm.sample_utils import make_sampler

            gen_kwargs["sampler"] = make_sampler(
                temp=FILTER_TEMPERATURE, top_p=FILTER_TOP_P
            )
        except Exception:
            pass
        out = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        return finalize_rewrite(str(out))
    finally:
        release_mlx_memory()


def mlx_prompt_baseline(
    draft: str,
    paths: ProfilePaths,
    *,
    max_tokens: int = 320,
    few_shot: str | None = None,
) -> str:
    """Rewrite with base MLX weights only (no LoRA) — honest prompt baseline."""
    config = load_config(paths)
    model_id = config.base_model or DEFAULT_MLX_MODEL
    adapter_dir = paths.adapters_dir / "latest"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    return _filter_mlx(
        draft,
        adapter_dir,
        model_id,
        max_tokens,
        paths=paths,
        use_adapter=False,
        few_shot=few_shot,
    )

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

    prompt = build_filter_prompt(draft)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=FILTER_TEMPERATURE,
            top_p=FILTER_TOP_P,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "### Rewritten" in text:
        text = text.split("### Rewritten", 1)[1].strip()
    return finalize_rewrite(text)


def read_draft_input(text: str | None, file: Path | None) -> str:
    if text and file:
        raise ValueError("Pass either --text or --file, not both.")
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is not None:
        return text
    raise ValueError("Provide --text or --file.")
