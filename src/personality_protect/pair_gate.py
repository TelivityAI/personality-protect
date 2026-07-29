"""Voice-rewrite training pair gate.

Pairs are ``(input=not-you, output=author voice)`` of the **same** meaning.
A poisoned flattener leaks author cadence into the input — those pairs must
fail closed before they hit the train set.

Also: ``sterile_flattener_check`` — one-shot contamination preflight before
bulk pair generation. Flatten an author post and a foreign press release with
the **same** prompt; if the author-derived flatten scores higher on proper
nouns, fragments, or you/I, the flattener is leaking and the dataset is
compromised at the source.

Contoso-safe heuristics only — not a NER model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from personality_protect.eval_compare import (
    _sentence_word_counts,
    _word_tokens,
    count_proper_nouns,
)

# Input still reads "voiced" if proper density is this high (Contoso-calibrated
# band for flat professional prose; re-derive on private corpus if needed).
MAX_INPUT_PROPER_PER_1K = 30.0
# Output should carry more short-line punch than the flattened input.
MIN_FRAG_GAP_RATIO = 0.08
# Flattened input should run longer sentences than the voiced output.
MIN_MEDIAN_SENTENCE_GAP = 2.0

# Sterile-flattener preflight: author_flat vs foreign_flat deltas.
MAX_STERILE_PROPER_DELTA = 8.0
MAX_STERILE_FRAG_DELTA = 0.12
MAX_STERILE_YOU_DELTA = 2

_PAIR_KEY_CANDIDATES = (
    ("input", "output"),
    ("not_you", "author"),
    ("source", "target"),
    ("prompt", "completion"),
)


def text_axes(text: str) -> dict[str, Any]:
    """Deterministic axes used by the pair gate and sterile-flattener check."""
    body = (text or "").strip()
    words = _word_tokens(body)
    n_words = max(1, len(words))
    proper = count_proper_nouns(body)
    sent = _sentence_word_counts(body)
    if sent:
        ordered = sorted(sent)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            median = float(ordered[mid])
        else:
            median = (ordered[mid - 1] + ordered[mid]) / 2.0
    else:
        median = 0.0
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    short = sum(1 for ln in lines if len(_word_tokens(ln)) <= 8)
    short_ratio = (short / len(lines)) if lines else 0.0
    you_n = len(re.findall(r"\byou\b", body, flags=re.I))
    i_n = len(re.findall(r"\bI\b", body))
    return {
        "words": len(words),
        "proper_nouns": proper,
        "proper_per_1k": round(proper * 1000.0 / n_words, 2),
        "median_sentence_words": round(median, 2),
        "short_line_ratio": round(short_ratio, 4),
        "you_count": you_n,
        "i_count": i_n,
        "you_gt_i": you_n > i_n,
    }


def gate_pair(
    input_text: str,
    output_text: str,
    *,
    max_input_proper_1k: float = MAX_INPUT_PROPER_PER_1K,
    min_frag_gap_ratio: float = MIN_FRAG_GAP_RATIO,
    min_median_sentence_gap: float = MIN_MEDIAN_SENTENCE_GAP,
) -> dict[str, Any]:
    """Return pass/fail plus reasons. Fail closed on contamination signals."""
    inp = text_axes(input_text)
    out = text_axes(output_text)
    failed: list[str] = []
    reasons: dict[str, str] = {}

    if not (input_text or "").strip():
        failed.append("empty_input")
        reasons["empty_input"] = "input text empty"
    if not (output_text or "").strip():
        failed.append("empty_output")
        reasons["empty_output"] = "output text empty"

    if inp["proper_per_1k"] > float(max_input_proper_1k):
        failed.append("max_input_proper_1k")
        reasons["max_input_proper_1k"] = (
            f"input proper/1k={inp['proper_per_1k']} > {max_input_proper_1k}"
        )

    frag_gap = float(out["short_line_ratio"]) - float(inp["short_line_ratio"])
    if frag_gap < float(min_frag_gap_ratio):
        failed.append("min_frag_gap_ratio")
        reasons["min_frag_gap_ratio"] = (
            f"frag_gap={round(frag_gap, 4)} < {min_frag_gap_ratio} "
            f"(out short_line={out['short_line_ratio']} vs in "
            f"{inp['short_line_ratio']})"
        )

    if inp["you_gt_i"]:
        failed.append("input_you_gt_i")
        reasons["input_you_gt_i"] = (
            f"input you({inp['you_count']}) > i({inp['i_count']}) — "
            "second-person voice leak"
        )

    median_gap = float(inp["median_sentence_words"]) - float(
        out["median_sentence_words"]
    )
    if median_gap < float(min_median_sentence_gap):
        failed.append("min_median_sentence_gap")
        reasons["min_median_sentence_gap"] = (
            f"median_gap={round(median_gap, 2)} < {min_median_sentence_gap} "
            f"(in {inp['median_sentence_words']} vs out "
            f"{out['median_sentence_words']})"
        )

    return {
        "pass": not failed,
        "failed": failed,
        "reasons": reasons,
        "input": inp,
        "output": out,
        "frag_gap_ratio": round(frag_gap, 4),
        "median_sentence_gap": round(median_gap, 2),
        "thresholds": {
            "max_input_proper_1k": max_input_proper_1k,
            "min_frag_gap_ratio": min_frag_gap_ratio,
            "min_median_sentence_gap": min_median_sentence_gap,
        },
    }


def sterile_flattener_check(
    flattened_author: str,
    flattened_foreign: str,
    *,
    max_proper_delta: float = MAX_STERILE_PROPER_DELTA,
    max_frag_delta: float = MAX_STERILE_FRAG_DELTA,
    max_you_delta: int = MAX_STERILE_YOU_DELTA,
) -> dict[str, Any]:
    """Contamination preflight: same flatten prompt on author post vs press release.

    If the flattener is sterile, both outputs land in the same band. If the
    author-derived flatten scores higher on proper / fragments / you, voice is
    leaking back in — stop before bulk pair generation.
    """
    a = text_axes(flattened_author)
    f = text_axes(flattened_foreign)
    failed: list[str] = []
    reasons: dict[str, str] = {}

    proper_delta = float(a["proper_per_1k"]) - float(f["proper_per_1k"])
    if proper_delta > float(max_proper_delta):
        failed.append("proper_delta")
        reasons["proper_delta"] = (
            f"author_flat proper/1k exceeds foreign_flat by {round(proper_delta, 2)}"
        )

    frag_delta = float(a["short_line_ratio"]) - float(f["short_line_ratio"])
    if frag_delta > float(max_frag_delta):
        failed.append("frag_delta")
        reasons["frag_delta"] = (
            f"author_flat short_line exceeds foreign_flat by {round(frag_delta, 4)}"
        )

    you_delta = int(a["you_count"]) - int(f["you_count"])
    if you_delta > int(max_you_delta):
        failed.append("you_delta")
        reasons["you_delta"] = (
            f"author_flat you-count exceeds foreign_flat by {you_delta}"
        )

    if a["you_gt_i"]:
        failed.append("author_flat_you_gt_i")
        reasons["author_flat_you_gt_i"] = "author flatten still has you > i"

    return {
        "pass": not failed,
        "failed": failed,
        "reasons": reasons,
        "author_flat": a,
        "foreign_flat": f,
        "deltas": {
            "proper_per_1k": round(proper_delta, 2),
            "short_line_ratio": round(frag_delta, 4),
            "you_count": you_delta,
            "median_sentence_words": round(
                float(a["median_sentence_words"]) - float(f["median_sentence_words"]),
                2,
            ),
        },
        "thresholds": {
            "max_proper_delta": max_proper_delta,
            "max_frag_delta": max_frag_delta,
            "max_you_delta": max_you_delta,
        },
    }


def extract_pair_texts(row: dict[str, Any]) -> tuple[str, str] | None:
    """Pull (input, output) from a JSONL row under several common key names."""
    for ik, ok in _PAIR_KEY_CANDIDATES:
        if ik in row and ok in row:
            return str(row[ik] or ""), str(row[ok] or "")
    return None


def iter_jsonl_pairs(path: Path) -> Iterator[tuple[int, dict[str, Any], str, str]]:
    """Yield (1-based line, raw row, input, output) from a JSONL file."""
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"line {i}: expected JSON object")
        pair = extract_pair_texts(row)
        if pair is None:
            raise ValueError(
                f"line {i}: need one of {_PAIR_KEY_CANDIDATES} key pairs"
            )
        yield i, row, pair[0], pair[1]


def gate_jsonl(
    path: Path,
    *,
    max_input_proper_1k: float = MAX_INPUT_PROPER_PER_1K,
    min_frag_gap_ratio: float = MIN_FRAG_GAP_RATIO,
    min_median_sentence_gap: float = MIN_MEDIAN_SENTENCE_GAP,
) -> dict[str, Any]:
    """Gate every pair in a JSONL file; return kept/dropped with reasons."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for line_no, row, inp, out in iter_jsonl_pairs(path):
        result = gate_pair(
            inp,
            out,
            max_input_proper_1k=max_input_proper_1k,
            min_frag_gap_ratio=min_frag_gap_ratio,
            min_median_sentence_gap=min_median_sentence_gap,
        )
        entry = {
            "line": line_no,
            "pass": result["pass"],
            "failed": result["failed"],
            "reasons": result["reasons"],
            "row": row,
            "axes": {
                "input": result["input"],
                "output": result["output"],
                "frag_gap_ratio": result["frag_gap_ratio"],
                "median_sentence_gap": result["median_sentence_gap"],
            },
        }
        if result["pass"]:
            kept.append(entry)
        else:
            dropped.append(entry)
    return {
        "path": str(path),
        "total": len(kept) + len(dropped),
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_rows": kept,
        "dropped_rows": dropped,
        "thresholds": {
            "max_input_proper_1k": max_input_proper_1k,
            "min_frag_gap_ratio": min_frag_gap_ratio,
            "min_median_sentence_gap": min_median_sentence_gap,
        },
    }


def write_kept_jsonl(entries: Iterable[dict[str, Any]], path: Path) -> int:
    """Write kept pair rows (original JSON objects) to ``path``."""
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry["row"], ensure_ascii=False) + "\n")
            n += 1
    return n
