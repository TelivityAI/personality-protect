"""Build local SFT JSONL from selected writing pieces."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from personality_protect.config import ProfilePaths
from personality_protect.models import Piece
from personality_protect.select import selected_pieces

SYSTEM_PROMPT = (
    "You are a personal voice rewriter. Rewrite the draft so it matches the user's "
    "authentic writing voice exactly — their cadence, short punches, rhetorical "
    "questions, metaphors, contractions, and paragraph rhythm — not generic clean prose. "
    "Keep factual meaning unchanged. Prefer their diction and punctuation habits. "
    "Strip generic AI tells such as: leverage, synergies, delve, robust, "
    "In today's fast-paced world, It is important to note, Moreover, Furthermore, "
    "unlock, nestled, testament, vibrant. Do not invent facts, citations, or claims. "
    "Do not add hashtags, emoji, or marketing slogans unless the voice reference uses them."
)

# Inference-aligned user turn (no reference dump — LoRA must carry voice).
USER_TEMPLATE_INFER = (
    "Rewrite the draft in my voice. Match my cadence and diction — short punches, "
    "direct address, rhetorical bite — not bland marketing. "
    "Keep the same meaning. Remove AI-sounding filler.\n\n"
    "### Draft\n{draft}\n\n"
    "### Rewritten"
)

# Optional few-shot / receipts shape (reference present).
USER_TEMPLATE = (
    "Rewrite the draft in my voice. Match my cadence and diction from the reference. "
    "Keep the same meaning. Remove AI-sounding filler.\n\n"
    "### Draft\n{draft}\n\n"
    "### My voice (reference)\n{reference}\n\n"
    "### Rewritten"
)

_QUOTE_ARTIFACT_RE = re.compile(r'(^|\n)\s*"+\s*(\n|$)')
_TRAILING_ESCAPED_QUOTE_RE = re.compile(r'\\"+')
_METAPHOR_FLATTEN = (
    (re.compile(r"\blife vest\b", re.I), "safety measure"),
    (re.compile(r"\bspeed boat\b", re.I), "fast vehicle"),
    (re.compile(r"\bsubmarine\b", re.I), "underwater vessel"),
    (re.compile(r"\bcash cow\b", re.I), "primary revenue source"),
    (re.compile(r"\btorch the\b", re.I), "damage the"),
    (re.compile(r"\bmoat\b", re.I), "competitive advantage"),
)


def normalize_corpus_text(text: str) -> str:
    """Fix LinkedIn/CSV quote wrapping without rewriting voice."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_ESCAPED_QUOTE_RE.sub('"', text)
    lines_out: list[str] = []
    blank_pending = False
    for raw_line in text.split("\n"):
        s = raw_line.strip()
        if not s or set(s) <= {'"'}:
            blank_pending = True
            continue
        # Unwrap CSV paragraph quotes: "paragraph." / paragraph."
        if s.startswith('"'):
            s = s[1:]
        if s.endswith('"') and not s.endswith('\\"'):
            s = s[:-1]
        s = s.strip()
        if not s:
            blank_pending = True
            continue
        if blank_pending and lines_out:
            lines_out.append("")
        blank_pending = False
        lines_out.append(s)
    text = "\n".join(lines_out)
    text = _QUOTE_ARTIFACT_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def piece_to_example(piece: Piece) -> dict:
    """Map a corpus piece to a supervised chat example.

    Training target is the user's real text. The draft side is a deliberately
    slopified / cadence-flattened prompt so the adapter learns *rewrite into voice*,
    not identity copy. Prompt shape matches filter inference (no reference).
    """
    target = normalize_corpus_text(piece.text)
    draft = _neutral_draft(target)
    user = USER_TEMPLATE_INFER.format(draft=draft)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "meta": {
            "piece_id": piece.id,
            "source": piece.source,
            "year": piece.year,
            "word_count": piece.word_count,
        },
    }


def _short_reference(text: str, max_chars: int = 400) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _neutral_draft(text: str) -> str:
    """Turn authentic corpus text into a generic AI-ish draft for SFT pairing.

    Must differ from the assistant target on almost every example — otherwise
    LoRA learns identity copy and filter collapses to post-hoc slop stripping.
    Flatten cadence markers (punches, metaphors, rhetorical bite) while keeping
    entities / factual meaning so the model must *restore* voice.
    """
    draft = normalize_corpus_text(text)
    if not draft:
        return draft

    # Flatten first-person / punchy voice into corporate diction.
    replacements = (
        ("Let's be real.", "It is important to note that"),
        ("Let's be honest.", "It is important to note that"),
        ("Let me speculate", "One might explore"),
        ("Bear with me.", "Furthermore,"),
        ("WTH?!", "this raises important questions."),
        ("WTF?!", "this raises important questions."),
        ("WTH?", "this raises important questions."),
        ("I've found that", "One finds that"),
        ("I genuinely believe", "It seems"),
        ("In my experience,", "Often,"),
        ("Here's the thing:", "Note:"),
        ("Let me be clear:", "To clarify:"),
        ("I cut the corporate fog", "Remove corporate language"),
        ("I do not need", "One does not need"),
        ("I absolutely get why", "It is understandable why"),
        ("I get why", "It is understandable why"),
        ("I want to see", "Stakeholders may wish to see"),
        ("Hope my speculation", "One hopes this speculation"),
        ("GO DUCK GO!", "This presents an opportunity for alternative channels."),
        ("B. R. U. T. A. L.", "This outcome is particularly challenging."),
        ("—", ","),
    )
    for a, b in replacements:
        draft = draft.replace(a, b)

    for pat, repl in _METAPHOR_FLATTEN:
        draft = pat.sub(repl, draft)

    # Soften contractions / casual markers.
    draft = re.sub(r"\bdon't\b", "do not", draft, flags=re.I)
    draft = re.sub(r"\bcan't\b", "cannot", draft, flags=re.I)
    draft = re.sub(r"\bwon't\b", "will not", draft, flags=re.I)
    draft = re.sub(r"\bI'm\b", "I am", draft)
    draft = re.sub(r"\bit's\b", "it is", draft, flags=re.I)
    draft = re.sub(r"\bThat's\b", "That is", draft)
    draft = re.sub(r"\bThere's\b", "There is", draft)
    draft = re.sub(r"\bWhat's\b", "What is", draft)
    draft = re.sub(r"[?!]{1,3}", ".", draft)

    # Flatten short punch paragraphs into continuous corporate prose.
    parts = [p.strip() for p in re.split(r"\n\s*\n", draft) if p.strip()]
    flat = " ".join(p.replace("\n", " ") for p in parts)
    flat = re.sub(r"\s{2,}", " ", flat).strip()

    # Inject deterministic AI-tell scaffolding so drafts look like slop input.
    seed = int(hashlib.sha256(flat.encode("utf-8")).hexdigest()[:8], 16)
    openers = (
        "In today's fast-paced digital world, it is important to note that ",
        "Moreover, we must leverage robust synergies as ",
        "Furthermore, unlocking nestled opportunities shows that ",
        "It is important to note that we should delve into how ",
    )
    mid_glue = (
        " Moreover, furthermore, ",
        " Additionally, this is a testament to vibrant innovation as ",
        " In today's landscape we must leverage ",
    )
    opener = openers[seed % len(openers)]
    glue = mid_glue[(seed // 7) % len(mid_glue)]

    if not flat:
        return opener.strip()
    body = opener + flat[0].lower() + flat[1:] if len(flat) > 1 else opener + flat
    # Split roughly in half and inject mid-glue so draft ≠ target structure.
    mid = len(body) // 2
    cut = body.find(". ", mid)
    if cut > 0 and cut + 2 < len(body):
        body = body[: cut + 1] + glue + body[cut + 2].lower() + body[cut + 3 :]
    else:
        body = body + " Moreover, this is a testament to vibrant innovation."
    return body.strip()


def build_sft_jsonl(pieces: Iterable[Piece], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for piece in pieces:
            if not piece.text.strip():
                continue
            fh.write(json.dumps(piece_to_example(piece), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_sft_from_profile(paths: ProfilePaths, out_path: Path | None = None) -> tuple[Path, int]:
    pieces = selected_pieces(paths)
    if not pieces:
        raise RuntimeError(
            "No selected pieces. Run ingest, then select, before building SFT."
        )
    dest = out_path or paths.sft_jsonl
    n = build_sft_jsonl(pieces, dest)
    return dest, n
