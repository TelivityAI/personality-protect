"""Translator holdout eval — product-shaped success/fail without personal data.

Hold out foreign (not-you) text and a Contoso author-band reference. Score a
translator rewrite:

* **Fail** — byte-identical echo, or output still in press-release band
* **Pass** — proper / fragments / you·I axes move toward the author band
  vs the sterile input

Ear-test (“sounds like me”) stays operator judgment; this module automates
the axis checks only. Contoso-safe public fixtures — no personal corpus.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from personality_protect.pair_gate import MIN_FRAG_GAP_RATIO, text_axes

FOREIGN_FIXTURE = "translator_foreign.txt"
AUTHOR_HOLDOUT_FIXTURE = "translator_author_holdout.txt"

# Author proper density must exceed sterile by this much before we require
# upward movement on the rewrite (Contoso-calibrated).
MIN_AUTHOR_PROPER_GAP = 5.0


def _evals_root():
    return resources.files("personality_protect").joinpath("data/evals")


def load_packaged_foreign_holdout() -> str:
    """Foreign press-release holdout (text the Contoso author did not write)."""
    return (_evals_root() / FOREIGN_FIXTURE).read_text(encoding="utf-8")


def load_packaged_author_holdout() -> str:
    """Held-out Contoso author post used as the voice-band reference."""
    return (_evals_root() / AUTHOR_HOLDOUT_FIXTURE).read_text(encoding="utf-8")


def score_translator_holdout(
    sterile_input: str,
    model_output: str,
    author_band: str,
    *,
    min_frag_gap_ratio: float = MIN_FRAG_GAP_RATIO,
    min_author_proper_gap: float = MIN_AUTHOR_PROPER_GAP,
) -> dict[str, Any]:
    """Score one translator rewrite against sterile input + author band.

    Returns pass/fail, failed codes, per-axis movement flags, and raw axes.
    """
    inp_text = sterile_input or ""
    out_text = model_output or ""
    auth_text = author_band or ""

    inp = text_axes(inp_text)
    out = text_axes(out_text)
    author = text_axes(auth_text)

    failed: list[str] = []
    reasons: dict[str, str] = {}
    axes_moved = {"frag": False, "proper": False, "you_i": False}

    if not out_text.strip():
        failed.append("empty_output")
        reasons["empty_output"] = "translator output empty"

    if out_text.strip() == inp_text.strip() and inp_text.strip():
        failed.append("byte_identical_echo")
        reasons["byte_identical_echo"] = (
            "output is byte-identical to sterile input — no translation"
        )

    frag_gap = float(out["short_line_ratio"]) - float(inp["short_line_ratio"])
    author_wants_frag = float(author["short_line_ratio"]) > float(
        inp["short_line_ratio"]
    )
    if author_wants_frag:
        if frag_gap >= float(min_frag_gap_ratio):
            axes_moved["frag"] = True
        else:
            failed.append("frag_not_toward_author")
            reasons["frag_not_toward_author"] = (
                f"frag_gap={round(frag_gap, 4)} < {min_frag_gap_ratio} "
                f"(out short_line={out['short_line_ratio']} vs in "
                f"{inp['short_line_ratio']}; author={author['short_line_ratio']})"
            )
    else:
        axes_moved["frag"] = True

    proper_gap_author = float(author["proper_per_1k"]) - float(inp["proper_per_1k"])
    if proper_gap_author >= float(min_author_proper_gap):
        if float(out["proper_per_1k"]) > float(inp["proper_per_1k"]):
            axes_moved["proper"] = True
        else:
            failed.append("proper_not_toward_author")
            reasons["proper_not_toward_author"] = (
                f"output proper/1k={out['proper_per_1k']} did not rise toward "
                f"author={author['proper_per_1k']} from input={inp['proper_per_1k']}"
            )
    else:
        axes_moved["proper"] = True

    if author["you_gt_i"] and not inp["you_gt_i"]:
        if out["you_gt_i"] or int(out["you_count"]) > int(inp["you_count"]):
            axes_moved["you_i"] = True
        else:
            failed.append("you_i_not_toward_author")
            reasons["you_i_not_toward_author"] = (
                f"author you>i but output you={out['you_count']} i={out['i_count']} "
                f"(input you={inp['you_count']} i={inp['i_count']})"
            )
    elif int(author["you_count"]) > int(inp["you_count"]):
        if int(out["you_count"]) > int(inp["you_count"]):
            axes_moved["you_i"] = True
        else:
            failed.append("you_i_not_toward_author")
            reasons["you_i_not_toward_author"] = (
                f"you-count did not rise toward author "
                f"(out={out['you_count']} in={inp['you_count']} "
                f"author={author['you_count']})"
            )
    else:
        axes_moved["you_i"] = True

    # Paraphrased sterile: not a literal echo, but still press-release band.
    if "byte_identical_echo" not in failed and out_text.strip():
        still_press = (
            frag_gap < float(min_frag_gap_ratio)
            and int(out["you_count"]) <= int(inp["you_count"])
            and abs(float(out["proper_per_1k"]) - float(inp["proper_per_1k"])) < 1.0
        )
        if still_press:
            failed.append("still_press_release")
            reasons["still_press_release"] = (
                "output remains in press-release band "
                "(no fragment / you·I / proper move vs sterile input)"
            )

    return {
        "pass": not failed,
        "failed": failed,
        "reasons": reasons,
        "echo": "byte_identical_echo" in failed,
        "axes_moved": axes_moved,
        "input": inp,
        "output": out,
        "author_band": author,
        "frag_gap_ratio": round(frag_gap, 4),
        "thresholds": {
            "min_frag_gap_ratio": min_frag_gap_ratio,
            "min_author_proper_gap": min_author_proper_gap,
        },
    }


def score_from_files(
    input_path: Path,
    output_path: Path,
    author_band_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score translator rewrite from three text files."""
    return score_translator_holdout(
        input_path.read_text(encoding="utf-8"),
        output_path.read_text(encoding="utf-8"),
        author_band_path.read_text(encoding="utf-8"),
        **kwargs,
    )
