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
    filter_system_prompt,
    mlx_prompt_baseline,
    read_draft_input,
    strip_ai_tells,
)

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

# Deterministic scaffolding leftovers (no LLM judgment).
_SCAFFOLDING_PATTERNS = (
    r"(?i)here's the problem:",
    r"(?i)here's the part that should genuinely bother",
    r"(?i)here's what\b[^\n]{0,160}\blooked like:",
    r"(?i)that's the part worth\b",
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


def scaffolding_count(text: str) -> int:
    """Count known throat-clearing scaffolding leftovers."""
    body = text or ""
    return sum(1 for pat in _SCAFFOLDING_PATTERNS if re.search(pat, body))


def _para_count(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()])


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


# Draft specificity gates (pre-voice). A filter cannot insert named entities or
# benchmarks — fail cheap at draft time.
#
# Proper-noun floors are channel p10s on the author's real corpus under
# ``count_proper_nouns`` (excludes sentence-initial capitals). A single 48/1k
# gate rejected ~34% of posts and ~17% of articles.
#
# Numbers: ~48% of real posts have zero figures (p10/p25 = 0) — no hard gate
# on LinkedIn. Articles keep the corpus floor (~6.6/1k). Demanding numbers
# when ``numbers_available`` is empty pushes the drafter to invent them.
# Median sentence / short-line / you>I are advisory only.
SCORECARD_MIN_PROPER_PER_1K_POST = 20.0
SCORECARD_MIN_PROPER_PER_1K_ARTICLE = 42.0
# Back-compat alias → article floor (stricter default when channel omitted).
SCORECARD_MIN_PROPER_PER_1K = SCORECARD_MIN_PROPER_PER_1K_ARTICLE
SCORECARD_MIN_NUMBERS_PER_1K_POST = 0.0
SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE = 6.6
# Back-compat alias → article numbers floor.
SCORECARD_MIN_NUMBERS_PER_1K = SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE


def resolve_proper_floor(
    channel: str | None = None,
    *,
    words: int | None = None,
) -> float:
    """Pick proper-noun /1k floor from channel (or word-count heuristic)."""
    ch = (channel or "").strip().lower()
    if ch in {"linkedin", "post", "posts", "li", "short"}:
        return SCORECARD_MIN_PROPER_PER_1K_POST
    if ch in {"article", "articles", "longform", "long"}:
        return SCORECARD_MIN_PROPER_PER_1K_ARTICLE
    if words is not None and int(words) < 500:
        return SCORECARD_MIN_PROPER_PER_1K_POST
    return SCORECARD_MIN_PROPER_PER_1K_ARTICLE


def resolve_numbers_floor(
    channel: str | None = None,
    *,
    words: int | None = None,
) -> float:
    """Pick numbers /1k floor. Posts: 0 (advisory only). Articles: corpus floor."""
    ch = (channel or "").strip().lower()
    if ch in {"linkedin", "post", "posts", "li", "short"}:
        return SCORECARD_MIN_NUMBERS_PER_1K_POST
    if ch in {"article", "articles", "longform", "long"}:
        return SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE
    if words is not None and int(words) < 500:
        return SCORECARD_MIN_NUMBERS_PER_1K_POST
    return SCORECARD_MIN_NUMBERS_PER_1K_ARTICLE

_COMMON_CAPS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "when",
        "where",
        "what",
        "why",
        "how",
        "who",
        "this",
        "that",
        "these",
        "those",
        "then",
        "than",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "after",
        "before",
        "because",
        "while",
        "although",
        "however",
        "therefore",
        "meanwhile",
        "also",
        "just",
        "only",
        "even",
        "still",
        "already",
        "here",
        "there",
        "was",
        "were",
        "are",
        "is",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "add",
        "was",
        "where",
        "most",
        "many",
        "some",
        "every",
        "each",
        "all",
        "none",
        "no",
        "not",
        "yes",
        "you",
        "your",
        "yours",
        "i",
        "we",
        "they",
        "he",
        "she",
        "it",
        "its",
        "our",
        "their",
    }
)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")


def _sentence_word_counts(text: str) -> list[int]:
    """Sentence lengths; bare short lines without terminals count as sentences."""
    body = (text or "").strip()
    if not body:
        return []
    counts: list[int] = []
    for block in re.split(r"\n+", body):
        block = block.strip()
        if not block:
            continue
        parts = re.split(r"(?<=[.!?…])\s+", block)
        for part in parts:
            n = len(_word_tokens(part))
            if n:
                counts.append(n)
    return counts


def count_proper_nouns(text: str) -> int:
    """Named-entity proxy: acronyms, multi-word Title Case, mid-sentence Capitals.

    Avoids counting sentence-initial ``Add`` / ``Was`` / ``Where`` as entities.
    Contoso-safe heuristic — not a NER model.
    """
    body = text or ""
    hits = 0
    hits += len(re.findall(r"\b[A-Z]{2,}\b", body))
    hits += len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", body))
    # Mid-sentence capital after lowercase/digit punctuation (not line start).
    for m in re.finditer(r"(?<=[a-z0-9,;:\"'”’)\]])\s+([A-Z][a-zA-Z0-9'-]{1,})\b", body):
        tok = m.group(1)
        if tok.isupper() and len(tok) >= 2:
            continue  # already counted as acronym
        if tok.lower() in _COMMON_CAPS:
            continue
        hits += 1
    return hits


# Evidence figures only — not labels or narrative clocks.
# - ATPCO Category 15/35 / version ids
# - T+N IRROPS timeline markers (concrete prose, not falsifiable claims)
_NON_EVIDENCE_NUMBER_CONTEXT = re.compile(
    r"(?i)\b(?:categor(?:y|ies)|cat\.?)\s*\d+(?:\s*/\s*\d+)?"
    r"|\b(?:categor(?:y|ies)|cat\.?)\s*\d+\s*(?:and|&|/)\s*\d+"
    r"|\b(?:rule|version|schema|v\.?)\s*[-/]?\s*\d+\b"
    r"|\bT\+\d+(?:\s*(?:to|/|-)\s*T\+\d+)?\b"
    r"|\bposition\s+\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
)
_EVIDENCE_NUMBER = re.compile(
    r"(?i)"
    r"(?:€|\$|£)\s?\d+(?:[.,]\d+)?(?:\s*(?:m|bn|k|million|billion))?"
    r"|\b\d+(?:[.,]\d+)?%"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\+?\s*-?\s*"
    r"(?:years?|yrs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?"
    r"|passengers?|pax|flights?|seconds?|tickets?|bookings?)\b"
    r"|\b\d+(?:[.,]\d+)?\+?\s*-?\s*"
    r"(?:years?|yrs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?"
    r"|passengers?|pax|flights?|seconds?|tickets?|bookings?)\b"
    r"|\b(?:19|20)\d{2}\b"
)


def count_numbers(text: str) -> int:
    """Evidence figures: %, money, N+unit (incl. thousands with a unit), years.

    Do **not** count: ``Category 15/35`` labels, ``T+15`` timeline markers, or
    bare/queue ``position 1,100`` style digits without a unit noun. Those are
    structure or unverifiable color — not ``75% of participants``.
    """
    masked = _NON_EVIDENCE_NUMBER_CONTEXT.sub(" ", text or "")
    return len(_EVIDENCE_NUMBER.findall(masked))


def specificity_scorecard(
    text: str,
    *,
    channel: str | None = None,
    min_proper_per_1k: float | None = None,
    min_numbers_per_1k: float | None = None,
) -> dict[str, Any]:
    """Deterministic draft gate: specificity before any voice filter.

    Hard FAIL: proper nouns /1k (channel p10). Numbers /1k are hard only on
    articles (~6.6); on LinkedIn posts they are advisory (floor 0) — half of
    real posts have zero figures.
    Advisory only (never FAIL): median sentence, short-line ratio, you vs I,
    and post-channel numbers.
    A voice filter cannot insert named entities or a benchmark table.
    """
    body = (text or "").strip()
    words = _word_tokens(body)
    n_words = max(1, len(words))
    if min_proper_per_1k is None:
        min_proper_per_1k = resolve_proper_floor(channel, words=len(words))
    if min_numbers_per_1k is None:
        min_numbers_per_1k = resolve_numbers_floor(channel, words=len(words))
    proper = count_proper_nouns(body)
    numbers = count_numbers(body)
    sent_counts = _sentence_word_counts(body)
    if sent_counts:
        ordered = sorted(sent_counts)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median_sent = float(ordered[mid])
        else:
            median_sent = (ordered[mid - 1] + ordered[mid]) / 2.0
    else:
        median_sent = 0.0
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    short_lines = sum(1 for ln in lines if len(_word_tokens(ln)) <= 8)
    short_line_ratio = (short_lines / len(lines)) if lines else 0.0
    you_n = len(re.findall(r"\byou\b", body, flags=re.I))
    i_n = len(re.findall(r"\bI\b", body))
    proper_1k = proper * 1000.0 / n_words
    numbers_1k = numbers * 1000.0 / n_words
    you_1k = you_n * 1000.0 / n_words
    i_1k = i_n * 1000.0 / n_words

    checks: dict[str, bool] = {
        "proper_nouns_per_1k": proper_1k >= float(min_proper_per_1k),
    }
    # Floor 0 → advisory only (do not hard-fail LinkedIn posts with no figures).
    if float(min_numbers_per_1k) > 0:
        checks["numbers_per_1k"] = numbers_1k >= float(min_numbers_per_1k)
    failed = [k for k, ok in checks.items() if not ok]
    advisory = {
        "median_sentence_words": round(median_sent, 2),
        "short_line_ratio": round(short_line_ratio, 4),
        "you_count": you_n,
        "i_count": i_n,
        "you_per_1k": round(you_1k, 2),
        "i_per_1k": round(i_1k, 2),
        "you_gt_i": you_n > i_n,
        "numbers_per_1k": round(numbers_1k, 2),
        "numbers_hard_gate": float(min_numbers_per_1k) > 0,
    }
    ch_norm = (channel or "").strip().lower() or (
        "linkedin" if len(words) < 500 else "article"
    )
    return {
        "words": len(words),
        "proper_nouns": proper,
        "numbers": numbers,
        "proper_nouns_per_1k": round(proper_1k, 2),
        "numbers_per_1k": round(numbers_1k, 2),
        "median_sentence_words": advisory["median_sentence_words"],
        "short_line_ratio": advisory["short_line_ratio"],
        "you_count": you_n,
        "i_count": i_n,
        "you_per_1k": advisory["you_per_1k"],
        "i_per_1k": advisory["i_per_1k"],
        "channel": ch_norm,
        "thresholds": {
            "min_proper_per_1k": min_proper_per_1k,
            "min_numbers_per_1k": min_numbers_per_1k,
        },
        "checks": checks,
        "advisory": advisory,
        "failed": failed,
        "pass": len(failed) == 0,
    }


def longform_metrics(draft: str, rewritten: str) -> dict[str, Any]:
    """Deterministic longform gate metrics (no LLM grader).

    Pipeline pass signal: scaffolding_after==0 and not near_copy and not
    blank_line_only_noop. Voice quality remains a human grade.
    """
    draft = (draft or "").strip()
    rewritten = (rewritten or "").strip()
    d_len = max(1, len(draft))
    r_len = len(rewritten)
    ratio = r_len / d_len
    dw, rw = _word_set(draft), _word_set(rewritten)
    overlap = len(dw & rw) / max(1, len(dw))
    # Near-copy: almost same length and high lexical overlap.
    near_copy = 0.92 <= ratio <= 1.08 and overlap >= 0.92
    # Blank-line-only: same words, more blank lines, no real rewrite.
    draft_flat = re.sub(r"\s+", " ", draft)
    re_flat = re.sub(r"\s+", " ", rewritten)
    blank_line_only = (
        draft_flat == re_flat
        or (overlap >= 0.97 and _para_count(rewritten) > _para_count(draft) and abs(ratio - 1.0) < 0.05)
    )
    scaff_after = scaffolding_count(rewritten)
    return {
        "scaffolding_before": scaffolding_count(draft),
        "scaffolding_after": scaff_after,
        "length_ratio": round(ratio, 4),
        "near_copy": near_copy,
        "blank_line_only_noop": blank_line_only,
        "paras_before": _para_count(draft),
        "paras_after": _para_count(rewritten),
        "slop_before": slop_score(draft),
        "slop_after": slop_score(rewritten),
        "lexical_overlap": round(overlap, 4),
        "pipeline_pass": bool(
            scaff_after == 0 and not near_copy and not blank_line_only and rewritten
        ),
    }


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
        "longform": longform_metrics(draft, rewritten),
        "system_prompt": filter_system_prompt(),
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
        "longform": {
            "prompt_baseline": longform_metrics(draft, baseline),
            "lora": longform_metrics(draft, lora_text),
        },
        "system_prompt": filter_system_prompt(),
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
