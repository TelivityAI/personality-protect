"""Eval / compare receipts: raw vs prompt baseline vs LoRA filter.

Outputs land under the local profile `evals/` directory (gitignored).
Synthetic public drafts live in package data under `data/evals/`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from personality_protect.config import ProfilePaths, load_config
from personality_protect.filter import (
    FILTER_TEMPERATURE,
    build_filter_prompt,
    filter_draft,
    mlx_prompt_baseline,
    read_draft_input,
    strip_ai_tells,
)
from personality_protect.sft import SYSTEM_PROMPT

# Soft AI-tell patterns used for a tiny local "slop score" on receipts
_SLOP_PATTERNS = (
    r"\bleverage\b",
    r"\bsynerg",
    r"\bdelve\b",
    r"\brobust\b",
    r"\bin today's (?:fast-paced\s+)?(?:digital\s+)?world\b",
    r"\bit is important to note\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\bunlock\b",
    r"\btestament to\b",
)


def evals_data_dir() -> Path:
    """Return path to packaged synthetic eval drafts."""
    try:
        root = resources.files("personality_protect").joinpath("data/evals")
        if hasattr(root, "iterdir"):
            return Path(str(root))
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    return here / "data" / "evals"


def list_synthetic_drafts() -> list[Path]:
    root = evals_data_dir()
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    )


def slop_score(text: str) -> int:
    """Count AI-tell pattern hits (higher = more generic slop)."""
    lower = text.lower()
    return sum(1 for pat in _SLOP_PATTERNS if re.search(pat, lower, flags=re.I))


def _load_voice_anchors(paths: ProfilePaths, limit: int = 3) -> list[str]:
    adapter = paths.adapters_dir / "latest" / "mock_adapter.json"
    if adapter.is_file():
        data = json.loads(adapter.read_text(encoding="utf-8"))
        anchors = [a.strip() for a in (data.get("anchors") or []) if a and a.strip()]
        if anchors:
            return anchors[:limit]
    # Fall back to short snippets from SFT assistant turns
    if paths.sft_jsonl.is_file():
        out: list[str] = []
        with paths.sft_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                for m in row.get("messages") or []:
                    if m.get("role") == "assistant" and m.get("content"):
                        text = m["content"].strip()
                        if len(text) > 40:
                            out.append(text[:400])
                        break
                if len(out) >= limit:
                    break
        return out
    return []


def _few_shot_block(anchors: list[str]) -> str:
    if not anchors:
        return ""
    examples = []
    for i, a in enumerate(anchors, 1):
        examples.append(f"Example {i} (voice):\n{a}")
    return "\n\n".join(examples)


def heuristic_prompt_baseline(draft: str, paths: ProfilePaths) -> str:
    """Deterministic strip+tag baseline for CI / mock (no multi-GB model)."""
    anchors = _load_voice_anchors(paths)
    few_shot = _few_shot_block(anchors)
    _ = build_filter_prompt(draft, few_shot=few_shot or None)
    body = strip_ai_tells(draft)
    if anchors:
        return (
            f"{body}\n\n"
            f"— prompt-baseline (heuristic strip; anchors={len(anchors)}; "
            f"temp={FILTER_TEMPERATURE}; no LoRA)"
        ).strip()
    return (
        f"{body}\n\n"
        f"— prompt-baseline (heuristic strip; temp={FILTER_TEMPERATURE}; no LoRA)"
    ).strip()


def prompt_baseline_rewrite(
    draft: str,
    paths: ProfilePaths,
    *,
    backend: str = "auto",
) -> tuple[str, str]:
    """Return (rewritten, baseline_kind).

    For MLX backends: real base-model generation (no adapter) with few-shot anchors.
    Otherwise: local heuristic strip (CI / mock).
    """
    anchors = _load_voice_anchors(paths)
    few_shot = _few_shot_block(anchors) or None
    want_mlx = backend in {"mlx", "auto"}
    if want_mlx:
        try:
            import importlib.util

            if importlib.util.find_spec("mlx_lm") is not None:
                text = mlx_prompt_baseline(draft, paths, few_shot=few_shot)
                kind = "mlx_base_fewshot" if few_shot else "mlx_base"
                return text, kind
        except Exception as exc:  # noqa: BLE001 — fall back honestly
            heuristic = heuristic_prompt_baseline(draft, paths)
            return (
                f"{heuristic}\n\n— mlx baseline failed ({type(exc).__name__}); "
                "fell back to heuristic"
            ).strip(), "heuristic_fallback"
    return heuristic_prompt_baseline(draft, paths), "heuristic"


# Back-compat alias used by older tests / callers
def prompt_baseline_rewrite_text(draft: str, paths: ProfilePaths) -> str:
    text, _ = prompt_baseline_rewrite(draft, paths, backend="mock")
    return text


def run_eval(
    paths: ProfilePaths,
    draft: str,
    *,
    backend: str = "auto",
    label: str | None = None,
) -> dict[str, Any]:
    """Filter one draft and write before/after under profile evals/."""
    load_config(paths)
    paths.ensure()
    rewritten, used = filter_draft(draft, paths, backend=backend)  # type: ignore[arg-type]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", (label or "draft"))[:40] or "draft"
    out_dir = paths.evals_dir / f"eval_{stamp}_{safe_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "before.txt").write_text(draft.strip() + "\n", encoding="utf-8")
    (out_dir / "after.txt").write_text(rewritten.strip() + "\n", encoding="utf-8")
    receipt = {
        "kind": "eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": used,
        "label": label,
        "slop_before": slop_score(draft),
        "slop_after": slop_score(rewritten),
        "system_prompt": SYSTEM_PROMPT,
        "dir": str(out_dir),
    }
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "dir": str(out_dir),
        "backend": used,
        "draft": draft.strip(),
        "rewritten": rewritten.strip(),
        "slop_before": receipt["slop_before"],
        "slop_after": receipt["slop_after"],
        "receipt": receipt,
    }


def run_compare(
    paths: ProfilePaths,
    draft: str,
    *,
    backend: str = "auto",
    label: str | None = None,
) -> dict[str, Any]:
    """Three-way compare: raw vs prompt baseline vs LoRA/mock filter."""
    load_config(paths)
    paths.ensure()
    baseline, baseline_kind = prompt_baseline_rewrite(draft, paths, backend=backend)
    lora_text, used = filter_draft(draft, paths, backend=backend)  # type: ignore[arg-type]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", (label or "draft"))[:40] or "draft"
    out_dir = paths.evals_dir / f"compare_{stamp}_{safe_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw.txt").write_text(draft.strip() + "\n", encoding="utf-8")
    (out_dir / "prompt_baseline.txt").write_text(baseline + "\n", encoding="utf-8")
    (out_dir / "lora.txt").write_text(lora_text.strip() + "\n", encoding="utf-8")

    receipt = {
        "kind": "compare",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter_backend": used,
        "baseline_kind": baseline_kind,
        "label": label,
        "slop": {
            "raw": slop_score(draft),
            "prompt_baseline": slop_score(baseline),
            "lora": slop_score(lora_text),
        },
        "system_prompt": SYSTEM_PROMPT,
        "dir": str(out_dir),
    }
    (out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "dir": str(out_dir),
        "filter_backend": used,
        "baseline_kind": baseline_kind,
        "raw": draft.strip(),
        "prompt_baseline": baseline,
        "lora": lora_text.strip(),
        "slop": receipt["slop"],
        "receipt": receipt,
    }


def resolve_eval_draft(
    *,
    text: str | None = None,
    file: Path | None = None,
    synthetic: str | None = None,
) -> tuple[str, str]:
    """Return (draft_text, label). Prefer --text/--file; else named synthetic draft."""
    if text or file:
        return read_draft_input(text, file), (file.stem if file else "text")
    drafts = {p.stem: p for p in list_synthetic_drafts()}
    if synthetic:
        if synthetic not in drafts:
            known = ", ".join(sorted(drafts)) or "(none packaged)"
            raise FileNotFoundError(
                f"Unknown synthetic draft {synthetic!r}. Known: {known}"
            )
        path = drafts[synthetic]
        return path.read_text(encoding="utf-8"), path.stem
    if not drafts:
        raise FileNotFoundError("No synthetic eval drafts packaged under data/evals/.")
    # Default to first packaged draft
    path = sorted(drafts.values())[0]
    return path.read_text(encoding="utf-8"), path.stem
