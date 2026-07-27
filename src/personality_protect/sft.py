"""Build local SFT JSONL from selected writing pieces."""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from personality_protect.config import ProfilePaths
from personality_protect.models import Piece
from personality_protect.select import selected_pieces

# Keep short: MLX mask_prompt @ 512 tokens — long system text starves assistant labels.
SYSTEM_PROMPT = (
    "Personal voice rewriter. Match cadence: short punches, rhetorical bite, "
    "contractions, paragraph rhythm — not generic clean prose. Keep meaning. "
    "Preserve paragraph breaks; never flatten multi-paragraph drafts. "
    "Keep a strong opener unless sloppy/AI scaffolding. "
    "Leave alone only for user's voice cadence (multi-para punches / rhetorical "
    "bite) — not bland clean professional prose. "
    "Flat, corporate, soulless, or AI-slop → multi-paragraph voice with blank "
    "lines — never a thesaurus one-liner or blank-line-only split. "
    "Strip AI tells (leverage, synergies, delve, robust, In today's fast-paced world, "
    "It is important to note, Moreover, Furthermore, unlock, nestled, testament, "
    "vibrant). No invented facts, hashtags, or emoji unless the voice uses them."
)

# Inference-aligned user turn (no reference dump — LoRA must carry voice).
USER_TEMPLATE_INFER = (
    "Rewrite in my voice when needed. Cadence — not bland marketing. "
    "Preserve paragraph breaks. Keep a strong opener unless sloppy. "
    "Flat/corporate/soulless/AI-slop → multi-para voice with blank lines "
    "(not thesaurus or blank-line-only split). "
    "Already my voice cadence → unchanged. Same meaning; drop AI filler.\n\n"
    "### Draft\n{draft}\n\n"
    "### Rewritten"
)

# Optional few-shot / receipts shape (reference present).
USER_TEMPLATE = (
    "Rewrite in my voice when needed. Match cadence from the reference. "
    "Preserve paragraph breaks. Keep a strong opener unless sloppy. "
    "Flat/corporate/soulless/AI-slop → multi-para voice with blank lines "
    "(not thesaurus or blank-line-only split). "
    "Already my voice cadence → unchanged. Same meaning; drop AI filler.\n\n"
    "### Draft\n{draft}\n\n"
    "### My voice (reference)\n{reference}\n\n"
    "### Rewritten"
)

_QUOTE_ARTIFACT_RE = re.compile(r'(^|\n)\s*"+\s*(\n|$)')
_TRAILING_ESCAPED_QUOTE_RE = re.compile(r'\\"+')
# LinkedIn article exports often paste Medium/Ghost CSS ahead of the body.
_CSS_RULE_RE = re.compile(
    r"(?ms)^[ \t]*[a-zA-Z_*#.][^{;\n]{0,160}\{[^{}]*\}[ \t]*\n?"
)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# MLX trains at max_seq_length=512 with mask_prompt. Prompt (system+user+draft)
# plus assistant target must fit; overflow leaves zero assistant tokens →
# Train loss nan and a dead / corrupt chunk. Keep drafts/targets short enough
# that chat-templated rows stay under 512 even with the longer voice prompt.
MAX_SFT_TARGET_CHARS = 280
MAX_SFT_DRAFT_CHARS = 280
# Hard char budget for system+user+assistant (chat template adds a little more).
# Slightly above 1600 so leave-alone/soulless prompt wording still fits
# max-length drafts without dropping the whole row (nan risk stays on seq=512).
MAX_SFT_EXAMPLE_CHARS = 1650
# Bland public business idioms only — never encode user-specific metaphors.
_METAPHOR_FLATTEN = (
    (re.compile(r"\bcash cow\b", re.I), "primary revenue source"),
    (re.compile(r"\btorch the\b", re.I), "damage the"),
    (re.compile(r"\bmoat\b", re.I), "competitive advantage"),
    (re.compile(r"\bsilver bullet\b", re.I), "simple solution"),
    (re.compile(r"\belephant in the room\b", re.I), "obvious issue"),
    (re.compile(r"\blow-hanging fruit\b", re.I), "easy opportunity"),
)


class _HTMLTextExtractor(HTMLParser):
    """Pull text from HTML while keeping block-level paragraph breaks."""

    _BLOCK = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag == "br":
            self._chunks.append("\n")
        elif tag in self._BLOCK:
            self._chunks.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK:
            self._chunks.append("\n\n")

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        raw = "".join(self._chunks)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _strip_html_css(text: str) -> str:
    """Remove embedded CSS rules and HTML tags from corpus/article paste-ups."""
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", "\n", text)
    prev = None
    while prev != text:
        prev = text
        text = _CSS_RULE_RE.sub("\n", text)
    if _HTML_TAG_RE.search(text):
        parser = _HTMLTextExtractor()
        try:
            parser.feed(text)
            extracted = parser.text()
            if extracted.strip():
                text = extracted
        except Exception:
            text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def normalize_corpus_text(text: str) -> str:
    """Fix LinkedIn/CSV quote wrapping and strip HTML/CSS without rewriting voice."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_html_css(text)
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


def _truncate_for_seq_budget(text: str, max_chars: int) -> str:
    """Word-boundary truncate so masked MLX examples stay inside max_seq_length.

    Preserves internal newlines/paragraph breaks in the kept prefix.
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    prefix = text[: max_chars - 1]
    # Prefer cutting at paragraph, then newline, then word.
    for sep in ("\n\n", "\n", " "):
        idx = prefix.rfind(sep)
        if idx >= max_chars // 3:
            cut = prefix[:idx].rstrip(",;:—- \t")
            break
    else:
        cut = prefix.rsplit(" ", 1)[0].rstrip(",;:—-") if " " in prefix else prefix
    return (cut or text[: max_chars - 1]).rstrip() + "…"


def _truncate_voice_target(text: str, max_chars: int = MAX_SFT_TARGET_CHARS) -> str:
    """Keep assistant targets inside the masked MLX sequence budget."""
    return _truncate_for_seq_budget(text, max_chars)


def _ensure_voice_paragraphs(text: str) -> str:
    """Turn sentence-level voice rhythm into blank-line paragraph punches.

    Rewrite pairs must teach flat/slop → multi-paragraph cadence. Corpus pieces
    that are already one block but have 2+ punchy sentences get paragraphized
    for slop/clean targets only (leave-alone keeps the original shape).
    """
    text = (text or "").strip()
    if not text or "\n\n" in text:
        return text
    # Soft line breaks inside a block → treat as paragraph candidates.
    if "\n" in text:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 2:
            return "\n\n".join(lines)
    parts = [
        p.strip()
        for p in re.split(r"(?<=[.!?])\s+", text)
        if p.strip()
    ]
    if len(parts) < 2:
        return text
    # Prefer 2–4 short punches; merge tiny trailing fragments into prior para.
    paras: list[str] = []
    for p in parts:
        if paras and len(p) < 28:
            paras[-1] = f"{paras[-1]} {p}"
        else:
            paras.append(p)
    if len(paras) < 2:
        return text
    return "\n\n".join(paras)


def _example_row(
    *,
    draft: str,
    target: str,
    piece: Piece,
    pair_kind: str,
) -> dict | None:
    """Build one SFT row, or None if it would overflow the MLX 512-token budget."""
    draft = (draft or "").strip()
    target = (target or "").strip()
    if not draft or not target:
        return None
    # Shrink draft then target until the row fits (prefer keeping the pair).
    for _ in range(6):
        user = USER_TEMPLATE_INFER.format(draft=draft)
        total = len(SYSTEM_PROMPT) + len(user) + len(target)
        if total <= MAX_SFT_EXAMPLE_CHARS:
            break
        overflow = total - MAX_SFT_EXAMPLE_CHARS
        if len(draft) > 96:
            draft = _truncate_for_seq_budget(
                draft, max(96, len(draft) - overflow - 8)
            )
            continue
        if len(target) > 96:
            target = _truncate_for_seq_budget(
                target, max(96, len(target) - overflow - 8)
            )
            continue
        return None
    else:
        return None
    user = USER_TEMPLATE_INFER.format(draft=draft)
    if len(SYSTEM_PROMPT) + len(user) + len(target) > MAX_SFT_EXAMPLE_CHARS:
        return None
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
            "pair_kind": pair_kind,
        },
    }


def piece_to_examples(piece: Piece) -> list[dict]:
    """Map one corpus piece to supervised rewrite pairs.

    Pair kinds:
    - slop→voice: AI-tell draft → authentic text (strip + restore cadence)
    - clean→voice: flat professional draft → authentic text with real paragraphs
    - leave_alone: already-voice draft → identical assistant (do not fidget)
    """
    raw_target = _truncate_voice_target(normalize_corpus_text(piece.text))
    if not raw_target.strip():
        return []
    # Rewrite pairs: multi-paragraph voice targets (cadence restore).
    voice_target = _truncate_voice_target(_ensure_voice_paragraphs(raw_target))
    out: list[dict] = []
    for kind, draft_fn in (("slop", _neutral_draft), ("clean", _clean_generic_draft)):
        # Draft from the paragraphized target so meaning aligns; flatten still
        # collapses to a single block for the user turn.
        draft = _truncate_for_seq_budget(draft_fn(voice_target), MAX_SFT_DRAFT_CHARS)
        if not draft.strip() or draft.strip() == voice_target.strip():
            continue
        row = _example_row(
            draft=draft, target=voice_target, piece=piece, pair_kind=kind
        )
        if row is not None:
            out.append(row)
        # Extra short multi-para pairs: reinforce branding-length cadence restore
        # (eval drafts are ~1–2 sentences of mush/flat).
        if (
            "\n\n" in voice_target
            and len(voice_target) <= 280
            and voice_target.count("\n\n") >= 1
        ):
            row = _example_row(
                draft=draft,
                target=voice_target,
                piece=piece,
                pair_kind=f"{kind}_cadence",
            )
            if row is not None:
                out.append(row)

    # Leave-alone only for *original* multi-paragraph voice pieces — teaches
    # "don't flatten / don't fidget". Do not mint leave-alone from artificially
    # paragraphized flats (that would dilute clean/slop → voice rewrite signal).
    leave_draft = _truncate_for_seq_budget(raw_target, MAX_SFT_DRAFT_CHARS)
    if leave_draft.strip() and "\n\n" in leave_draft:
        row = _example_row(
            draft=leave_draft,
            target=leave_draft,
            piece=piece,
            pair_kind="leave_alone",
        )
        if row is not None:
            out.append(row)
    return out


def piece_to_example(piece: Piece) -> dict:
    """Back-compat: return the slop→voice pair (or the only available pair)."""
    examples = piece_to_examples(piece)
    if not examples:
        target = _truncate_voice_target(normalize_corpus_text(piece.text))
        draft = _truncate_for_seq_budget(target, MAX_SFT_DRAFT_CHARS)
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE_INFER.format(draft=draft)},
                {"role": "assistant", "content": target},
            ],
            "meta": {
                "piece_id": piece.id,
                "source": piece.source,
                "year": piece.year,
                "word_count": piece.word_count,
                "pair_kind": "identity",
            },
        }
    for ex in examples:
        if ex["meta"].get("pair_kind") == "slop":
            return ex
    return examples[0]


def _short_reference(text: str, max_chars: int = 400) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _flatten_voice_markers(text: str) -> str:
    """Flatten cadence/metaphor/contractions into plain continuous prose."""
    draft = normalize_corpus_text(text)
    if not draft:
        return draft

    # Flatten first-person / punchy voice into corporate diction.
    replacements = (
        ("Let's be real.", "It is worth noting that"),
        ("Let's be honest.", "It is worth noting that"),
        ("Let me speculate", "One might explore"),
        ("Bear with me.", "Next,"),
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

    # Flatten short punch paragraphs into continuous prose.
    parts = [p.strip() for p in re.split(r"\n\s*\n", draft) if p.strip()]
    flat = " ".join(p.replace("\n", " ") for p in parts)
    return re.sub(r"\s{2,}", " ", flat).strip()


def _neutral_draft(text: str) -> str:
    """Turn authentic corpus text into a generic AI-ish draft for SFT pairing.

    Must differ from the assistant target on almost every example — otherwise
    LoRA learns identity copy and filter collapses to post-hoc slop stripping.
    Flatten cadence markers (punches, metaphors, rhetorical bite) while keeping
    entities / factual meaning so the model must *restore* voice.
    """
    flat = _flatten_voice_markers(text)
    if not flat:
        return flat

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

    body = opener + flat[0].lower() + flat[1:] if len(flat) > 1 else opener + flat
    # Split roughly in half and inject mid-glue so draft ≠ target structure.
    mid = len(body) // 2
    cut = body.find(". ", mid)
    if cut > 0 and cut + 2 < len(body):
        body = body[: cut + 1] + glue + body[cut + 2].lower() + body[cut + 3 :]
    else:
        body = body + " Moreover, this is a testament to vibrant innovation."
    return body.strip()


def _clean_generic_draft(text: str) -> str:
    """Cadence-flattened professional prose *without* AI-tell scaffolding.

    Teaches voice injection on already-clean drafts — the gap where slop-only
    SFT collapses to identity copy. Must stay lexically distant from the
    assistant target even when the corpus piece is already flat prose;
    near-identity clean pairs teach LoRA pass-through.
    """
    target = normalize_corpus_text(text)
    flat = _flatten_voice_markers(text)
    if not flat:
        return flat

    soft = (
        ("One finds that", "Research suggests that"),
        ("It is understandable why", "It makes sense that"),
        ("Stakeholders may wish to see", "It may be useful to see"),
        ("It is worth noting that", "Notably,"),
    )
    for a, b in soft:
        flat = flat.replace(a, b)

    # Impersonalize first-person / direct address.
    for pat, repl in (
        (re.compile(r"\bI've\b"), "one has"),
        (re.compile(r"\bI'm\b"), "one is"),
        (re.compile(r"\bI'd\b"), "one would"),
        (re.compile(r"\bI\b"), "one"),
        (re.compile(r"\bWe\b"), "Teams"),
        (re.compile(r"\bwe\b"), "teams"),
        (re.compile(r"\bmy\b"), "the"),
        (re.compile(r"\bour\b"), "organizational"),
        (re.compile(r"\byou\b"), "stakeholders"),
        (re.compile(r"\byour\b"), "their"),
    ):
        flat = pat.sub(repl, flat)

    # Broad professional synonym drift (no AI-tell openers).
    for pat, repl in (
        (re.compile(r"\bmore than ever\b", re.I), "increasingly"),
        (re.compile(r"\bCompanies\b"), "Organizations"),
        (re.compile(r"\bcompanies\b"), "organizations"),
        (re.compile(r"\bmatters\b", re.I), "is essential"),
        (re.compile(r"\bneed a clear\b", re.I), "require a distinct"),
        (re.compile(r"\bpoint of view\b", re.I), "perspective"),
        (re.compile(r"\banother template post\b", re.I), "a templated statement"),
        (re.compile(r"\btemplate post\b", re.I), "generic statement"),
        (re.compile(r"\babout authenticity\b", re.I), "regarding genuine positioning"),
        (re.compile(r"\bauthenticity\b", re.I), "genuine positioning"),
        (re.compile(r"\bflood every channel\b", re.I), "appear across every channel"),
        (re.compile(r"\bflood\b", re.I), "appear across"),
        (re.compile(r"\bAI tools\b"), "AI systems"),
        (re.compile(r"\bwent well\b", re.I), "proceeded successfully"),
        (re.compile(r"\bnoticed\b", re.I), "observed"),
        (re.compile(r"\bsimpler\b", re.I), "more streamlined"),
        (re.compile(r"\bkept the messaging honest\b", re.I), "maintained straightforward messaging"),
        (re.compile(r"\bCustomers\b"), "Clients"),
        (re.compile(r"\bcustomers\b"), "clients"),
        (re.compile(r"\bproduct launch\b", re.I), "release"),
        (re.compile(r"\bcheckout\b", re.I), "purchase flow"),
        (re.compile(r"\bknowledge\b", re.I), "familiarity"),
        (re.compile(r"\bvertical\b", re.I), "sector"),
        (re.compile(r"\bsuppliers\b", re.I), "vendors"),
        (re.compile(r"\bsignaling\b", re.I), "indicating"),
        (re.compile(r"\bdesperate\b", re.I), "eager"),
        (re.compile(r"\bdomain expertise\b", re.I), "specialist experience"),
        (re.compile(r"\bdecades of\b", re.I), "years of"),
        (re.compile(r"\bgive away\b", re.I), "relinquish"),
        (re.compile(r"\bget into this\b", re.I), "enter this space"),
        (re.compile(r"\bbillable hours\b", re.I), "utilization targets"),
        (re.compile(r"\breal\b", re.I), "practical"),
        (re.compile(r"\bclear\b", re.I), "well-defined"),
        (re.compile(r"\bneed\b", re.I), "require"),
        (re.compile(r"\bkeep\b", re.I), "maintain"),
        (re.compile(r"\bhonest\b", re.I), "straightforward"),
        (re.compile(r"\bimportant\b", re.I), "material"),
        (re.compile(r"\bthing\b", re.I), "matter"),
        (re.compile(r"\bpeople\b", re.I), "individuals"),
        (re.compile(r"\bwork\b", re.I), "effort"),
        (re.compile(r"\bmake\b", re.I), "create"),
        (re.compile(r"\bthink\b", re.I), "consider"),
        (re.compile(r"\bwant\b", re.I), "prefer"),
        (re.compile(r"\blook\b", re.I), "examine"),
        (re.compile(r"\bbig\b", re.I), "substantial"),
        (re.compile(r"\bsmall\b", re.I), "limited"),
        (re.compile(r"\bgood\b", re.I), "favorable"),
        (re.compile(r"\bbad\b", re.I), "unfavorable"),
        (re.compile(r"\bhelp\b", re.I), "assist"),
        (re.compile(r"\bshow\b", re.I), "demonstrate"),
        (re.compile(r"\btalk\b", re.I), "discuss"),
        (re.compile(r"\bsay\b", re.I), "state"),
        (re.compile(r"\bget\b", re.I), "obtain"),
        (re.compile(r"\bgo\b", re.I), "proceed"),
        (re.compile(r"\bsee\b", re.I), "observe"),
        (re.compile(r"\bknow\b", re.I), "recognize"),
        (re.compile(r"\buse\b", re.I), "apply"),
        (re.compile(r"\btry\b", re.I), "attempt"),
        (re.compile(r"\bstart\b", re.I), "initiate"),
        (re.compile(r"\bstop\b", re.I), "discontinue"),
        (re.compile(r"\bbuild\b", re.I), "construct"),
        (re.compile(r"\bchange\b", re.I), "adjust"),
        (re.compile(r"\bmove\b", re.I), "shift"),
        (re.compile(r"\bfind\b", re.I), "identify"),
        (re.compile(r"\bgive\b", re.I), "provide"),
        (re.compile(r"\btake\b", re.I), "adopt"),
        (re.compile(r"\bcome\b", re.I), "arrive"),
        (re.compile(r"\bask\b", re.I), "request"),
        (re.compile(r"\btell\b", re.I), "inform"),
        (re.compile(r"\bfeel\b", re.I), "perceive"),
        (re.compile(r"\bbleieve\b", re.I), "hold"),
        (re.compile(r"\bbelieve\b", re.I), "hold"),
        (re.compile(r"\bseems?\b", re.I), "appears"),
        (re.compile(r"\breally\b", re.I), "genuinely"),
        (re.compile(r"\bjust\b", re.I), "simply"),
        (re.compile(r"\bstill\b", re.I), "nevertheless"),
        (re.compile(r"\beven\b", re.I), "additionally"),
        (re.compile(r"\bmuch\b", re.I), "considerably"),
        (re.compile(r"\bmany\b", re.I), "numerous"),
        (re.compile(r"\bway\b", re.I), "approach"),
        (re.compile(r"\btime\b", re.I), "period"),
        (re.compile(r"\byear\b", re.I), "period"),
        (re.compile(r"\byears\b", re.I), "periods"),
    ):
        flat = pat.sub(repl, flat)

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", flat) if p.strip()]
    if len(parts) >= 2:
        rest = " ".join(
            (p[0].lower() + p[1:]) if len(p) > 1 else p.lower() for p in parts[1:]
        )
        flat = f"{parts[0]} Also, {rest}"
    elif flat and flat == target:
        flat = "Simply put, " + flat[0].lower() + flat[1:]

    flat = re.sub(r"\s{2,}", " ", flat).strip()
    flat = _force_clean_draft_distance(flat, target)
    return flat.strip()


_CLEAN_STOP = frozenset(
    """
    a an the and or but if in on at to for of as is are was were be been being
    it this that these those with from by not no so than then also their its
    they them he she his her him who what when where which while about into
    over after before between through during without within along against
    one teams stakeholders organizational notably also additionally simply
    brief practical operational standpoint terms useful note consideration
    """.split()
)

_CLEAN_BLAND = (
    "aspect",
    "factor",
    "element",
    "arrangement",
    "development",
    "situation",
    "process",
    "capability",
    "priority",
    "outcome",
    "practice",
    "position",
    "activity",
    "perspective",
    "approach",
    "constraint",
    "requirement",
    "implication",
    "assessment",
    "objective",
)


def _force_clean_draft_distance(draft: str, target: str, max_overlap: float = 0.72) -> str:
    """Ensure clean drafts are lexically distant enough to teach rewrite→voice.

    Keeps stopwords and mid-text Capitalized entities. Blandifies other content
    words only while overlap stays too high. Last resort keeps entities + a few
    topic anchors in a professional skeleton sentence.
    """
    draft = (draft or "").strip()
    target = (target or "").strip()
    if not draft:
        return draft

    def _overlap(a: str, b: str) -> float:
        tw = set(re.findall(r"[a-z0-9']+", b.lower()))
        dw = set(re.findall(r"[a-z0-9']+", a.lower()))
        return len(tw & dw) / max(1, len(tw))

    seed = int(hashlib.sha256((draft + "|" + target).encode("utf-8")).hexdigest()[:8], 16)
    frames = (
        "A practical consideration is that ",
        "From an operational standpoint, ",
        "In practical terms, ",
        "It may be useful to note that ",
    )

    if draft == target:
        draft = (
            "In brief: " + target[0].lower() + target[1:]
            if len(target) > 1
            else "In brief: " + target
        )

    if _overlap(draft, target) >= max_overlap:
        body = draft[0].lower() + draft[1:] if len(draft) > 1 else draft.lower()
        draft = (frames[seed % len(frames)] + body).strip()

    if _overlap(draft, target) < max_overlap:
        return draft

    tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]+|\s+", draft)
    out: list[str] = []
    alpha_i = 0
    for idx, tok in enumerate(tokens):
        if not tok.isalpha():
            out.append(tok)
            continue
        low = tok.lower()
        is_entity = tok[0].isupper() and alpha_i > 0
        remainder = "".join(tokens[idx + 1 :])
        live = "".join(out)
        if low in _CLEAN_STOP or is_entity:
            out.append(tok)
        elif _overlap(live + tok + remainder, target) < max_overlap:
            out.append(tok)
        else:
            # Prefer keeping longer topic words when a shorter neighbor can absorb the cut.
            if len(tok) >= 7 and _overlap(live + tok + remainder, target) < max_overlap + 0.08:
                out.append(tok)
            else:
                repl = _CLEAN_BLAND[(seed + alpha_i) % len(_CLEAN_BLAND)]
                out.append(repl if tok.islower() else repl.capitalize())
        alpha_i += 1

    draft = re.sub(r"\s{2,}", " ", "".join(out)).strip()

    if _overlap(draft, target) >= max_overlap:
        ents = re.findall(
            r"\b[A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z][A-Za-z0-9&'-]*)*\b", target
        )
        ents = [e for e in ents if e.lower() not in {"i", "a", "the"}][:6]
        anchors = [
            w
            for w in re.findall(r"[a-z]{5,}", target.lower())
            if w not in _CLEAN_STOP
        ]
        # Prefer distinctive anchors; keep order, unique.
        seen: set[str] = set()
        topic: list[str] = []
        for w in anchors:
            if w in seen:
                continue
            seen.add(w)
            topic.append(w)
            if len(topic) >= 4:
                break
        ent_bit = ", ".join(ents) if ents else "this domain"
        topic_bit = ", ".join(topic) if topic else "the core message"
        draft = (
            f"{frames[seed % len(frames)]}"
            f"organizations connected with {ent_bit} require a well-defined "
            f"perspective on {topic_bit}, rather than a templated statement."
        )
    return draft.strip()


def _synthetic_short_cadence_examples() -> list[dict]:
    """Branding-length synthetic pairs: flat/slop → punchy multi-para voice.

    Contoso/Northwind only — no personal corpus. Pattern-matched to short
    eval-length drafts but not verbatim copies of packaged eval stems.
    """
    pairs: list[tuple[str, str, str]] = [
        (
            "slop",
            (
                "In today's fast-paced world, it is important to note that Contoso "
                "must leverage robust synergies to delve into thought leadership. "
                "Moreover, furthermore, unlocking nestled opportunities is a "
                "testament to vibrant innovation."
            ),
            (
                "Let's be real.\n\n"
                "Contoso doesn't need a synonym parade. It needs to sound like a "
                "person when every channel is drowning in the same sludge.\n\n"
                "Say something with a spine — or don't post."
            ),
        ),
        (
            "clean",
            (
                "Thought leadership matters increasingly as generative systems "
                "appear across every channel. Organizations require a distinct "
                "perspective, not another templated statement regarding genuine "
                "positioning."
            ),
            (
                "Thought leadership matters more than ever.\n\n"
                "Generative sludge floods every channel with the same polished "
                "nothing.\n\n"
                "Organizations need a point of view — not another template about "
                "authenticity."
            ),
        ),
        (
            "slop",
            (
                "Furthermore, Contoso Labs must leverage robust synergies to unlock "
                "nestled opportunities across the vertical. It is important to note "
                "that this is a testament to vibrant innovation."
            ),
            (
                "Contoso Labs keeps talking about synergies.\n\n"
                "Meanwhile the vertical already knows who has a real point of view.\n\n"
                "Stop unlocking nestled nothing. Ship the take."
            ),
        ),
        (
            "clean",
            (
                "Organizations require a well-defined perspective on genuine "
                "positioning as AI systems appear across every channel."
            ),
            (
                "Everyone's posting about authenticity.\n\n"
                "Almost nobody sounds like a person.\n\n"
                "If AI floods the channel, your point of view is the only filter left."
            ),
        ),
        (
            "slop",
            (
                "Moreover, we must delve into how Northwind Analytics can leverage "
                "synergies. In today's landscape this is a testament to vibrant "
                "innovation."
            ),
            (
                "Northwind Analytics — really?\n\n"
                "You don't need synergies. You need a clear take that isn't another "
                "template.\n\n"
                "That's the whole game."
            ),
        ),
        (
            "clean",
            (
                "From an operational standpoint, teams require a distinct perspective "
                "rather than a templated statement regarding genuine positioning."
            ),
            (
                "Here's the thing:\n\n"
                "A templated statement about authenticity is still a template.\n\n"
                "Have a point of view, or stay quiet."
            ),
        ),
    ]
    out: list[dict] = []
    for kind, draft, target in pairs:
        piece = Piece(
            id=f"synthetic_cadence_{kind}_{len(out)}",
            source="demo",
            text=target,
            year=2026,
            word_count=len(target.split()),
        )
        row = _example_row(draft=draft, target=target, piece=piece, pair_kind=kind)
        if row is None:
            continue
        out.append(row)
        # Light oversample — multipara pairs carry the heavy lift now.
        for _ in range(1):
            dup = _example_row(
                draft=draft,
                target=target,
                piece=piece,
                pair_kind=f"{kind}_cadence",
            )
            if dup is not None:
                out.append(dup)
    return out


def _synthetic_clean_flat_voice_examples() -> list[dict]:
    """Bland clean-flat → multi-para voice (Contoso/Northwind only).

    Teaches restyle on soulless professional prose that has no AI-tells — the
    clean_neutral failure mode where LoRA only inserts a blank line. Heavy
    oversample; leave-alone stays on real multi-para corpus pieces elsewhere.
    """
    pairs: list[tuple[str, str]] = [
        (
            (
                "Personal branding is increasingly essential as AI systems appear "
                "across every channel. Organizations require a distinct perspective "
                "rather than a templated statement regarding genuine positioning."
            ),
            (
                "Personal branding matters more than ever.\n\n"
                "AI floods every channel with the same polished nothing.\n\n"
                "Companies need a point of view — not another authenticity template.\n\n"
                "Who's saying something real?"
            ),
        ),
        (
            (
                "From an operational standpoint, Contoso teams should maintain "
                "straightforward messaging as generative systems appear across "
                "channels. A well-defined perspective outperforms templated statements."
            ),
            (
                "Here's the thing:\n\n"
                "Contoso doesn't need another polished nothing-post.\n\n"
                "Have a point of view, or stay quiet.\n\n"
                "That's the filter now."
            ),
        ),
        (
            (
                "Northwind organizations require a well-defined perspective on genuine "
                "positioning. Templated statements fail to resonate with stakeholders "
                "when AI systems appear across every channel."
            ),
            (
                "Northwind — everyone's posting about authenticity.\n\n"
                "Almost nobody sounds like a person.\n\n"
                "If AI floods the channel, your point of view is the only filter left.\n\n"
                "Make it count."
            ),
        ),
        (
            (
                "Thought leadership is essential as generative systems appear across "
                "every channel. Contoso Labs require a distinct perspective rather "
                "than a templated statement regarding genuine positioning."
            ),
            (
                "Thought leadership matters more than ever.\n\n"
                "Contoso Labs keeps shipping the same template about authenticity.\n\n"
                "Generative sludge floods every channel.\n\n"
                "Say something with a spine — or don't post."
            ),
        ),
        (
            (
                "Organizations connected with Contoso require a well-defined "
                "perspective on personal branding, rather than a templated statement. "
                "AI systems appear across every channel with increasing frequency."
            ),
            (
                "Let's be real.\n\n"
                "Contoso's problem isn't missing a branding deck.\n\n"
                "It's sounding like every other post in the feed.\n\n"
                "Cut the fog. Keep the spine."
            ),
        ),
        (
            (
                "Notably, teams should examine how Northwind Analytics maintains "
                "straightforward messaging. Clients prefer a distinct perspective "
                "over a templated statement regarding genuine positioning."
            ),
            (
                "Northwind Analytics — really?\n\n"
                "Clients don't want another authenticity template.\n\n"
                "They want a clear take that isn't mush.\n\n"
                "Ship the take."
            ),
        ),
    ]
    out: list[dict] = []
    for draft, target in pairs:
        draft_t = _truncate_for_seq_budget(draft, MAX_SFT_DRAFT_CHARS)
        target_t = _truncate_for_seq_budget(target, MAX_SFT_TARGET_CHARS)
        # Keep blank-line punches under budget (prefer full paras over ellipsis mush).
        if "\n\n" in target and len(target) > MAX_SFT_TARGET_CHARS:
            paras = [p.strip() for p in target.split("\n\n") if p.strip()]
            kept: list[str] = []
            for p in paras:
                cand = "\n\n".join(kept + [p]) if kept else p
                if len(cand) <= MAX_SFT_TARGET_CHARS:
                    kept.append(p)
                elif not kept:
                    kept.append(_truncate_for_seq_budget(p, MAX_SFT_TARGET_CHARS))
                    break
                else:
                    break
            target_t = "\n\n".join(kept) if kept else target_t
        piece = Piece(
            id=f"synthetic_clean_flat_{len(out)}",
            source="demo",
            text=target_t,
            year=2026,
            word_count=len(target_t.split()),
        )
        row = _example_row(
            draft=draft_t, target=target_t, piece=piece, pair_kind="clean"
        )
        if row is None:
            continue
        out.append(row)
        # Light oversample — heavier 5× oversample collapsed multipara cadence
        # into repetition loops; 2× keeps clean→voice signal without drowning
        # leave-alone / multipara pairs.
        for _ in range(2):
            dup = _example_row(
                draft=draft_t,
                target=target_t,
                piece=piece,
                pair_kind="clean_flat",
            )
            if dup is not None:
                out.append(dup)
    return out


def _synthetic_multipara_cadence_examples() -> list[dict]:
    """Multi-paragraph Contoso/Northwind slop→voice pairs (not one-liner branding).

    Drafts are multi-para AI mush with blank lines. Targets keep blank lines,
    rhetorical questions, and short punches. Sized for the MLX 512-token budget.
    """
    pairs: list[tuple[str, str, str]] = [
        (
            "slop",
            (
                "In today's world, Contoso must leverage robust synergies across channels.\n\n"
                "Moreover, unlocking nestled opportunities is a testament to vibrant "
                "innovation.\n\n"
                "Furthermore, Contoso Labs should delve into storytelling with robust "
                "methods."
            ),
            (
                "Let's be real.\n\n"
                "Contoso doesn't need another synonym parade.\n\n"
                "Who's saying something with a spine?\n\n"
                "Ship the take — or don't post."
            ),
        ),
        (
            "slop",
            (
                "Moreover, Northwind must leverage synergies to delve into leadership.\n\n"
                "Unlocking nestled opportunities is a testament to vibrant innovation.\n\n"
                "Additionally, teams utilize robust frameworks that unlock synergistic "
                "outcomes."
            ),
            (
                "Northwind Analytics — really?\n\n"
                "You don't need synergies. You need a clear take.\n\n"
                "Stop unlocking nestled nothing.\n\n"
                "That's the whole game."
            ),
        ),
        (
            "slop",
            (
                "Contoso Labs must leverage cutting-edge paradigms for thought leadership.\n\n"
                "It is important to note that we should delve into customer delight.\n\n"
                "Additionally, robust synergies nestle innovation at the core."
            ),
            (
                "Contoso Labs keeps talking about paradigms.\n\n"
                "The vertical already knows who has a real point of view.\n\n"
                "Customer delight isn't a framework.\n\n"
                "Drop the sludge. Say the thing."
            ),
        ),
        (
            "clean",
            (
                "Thought leadership matters as generative systems appear across channels.\n\n"
                "Organizations require a distinct perspective, not another templated "
                "statement.\n\n"
                "Contoso teams should maintain straightforward messaging."
            ),
            (
                "Thought leadership matters more than ever.\n\n"
                "Generative sludge floods every channel.\n\n"
                "Contoso needs a point of view — not another authenticity template.\n\n"
                "Have a spine, or stay quiet."
            ),
        ),
        (
            "clean",
            (
                "Personal branding is essential as AI systems appear across channels.\n\n"
                "Northwind organizations require a well-defined perspective on genuine "
                "positioning.\n\n"
                "Templated statements fail to resonate with stakeholders."
            ),
            (
                "Personal branding? It's the filter now.\n\n"
                "Northwind — everyone's posting about authenticity.\n\n"
                "Almost nobody sounds like a person.\n\n"
                "Make it count."
            ),
        ),
        (
            "slop",
            (
                "Furthermore, Contoso must leverage robust synergies to unlock nestled "
                "opportunities.\n\n"
                "Moreover, vibrant innovation is a testament to leadership in today's "
                "world.\n\n"
                "Additionally, teams utilize robust methods with confidence."
            ),
            (
                "Here's the thing:\n\n"
                "Contoso's problem isn't missing synergies.\n\n"
                "It's sounding like every other post in the feed.\n\n"
                "Cut the fog. Keep the spine."
            ),
        ),
    ]
    def _fit_paras(text: str, limit: int) -> str:
        """Keep as many blank-line paragraphs as fit under the char budget."""
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        kept: list[str] = []
        for p in paras:
            cand = "\n\n".join(kept + [p]) if kept else p
            if len(cand) <= limit:
                kept.append(p)
            elif not kept:
                kept.append(_truncate_for_seq_budget(p, limit))
                break
            else:
                break
        return "\n\n".join(kept) if kept else _truncate_for_seq_budget(text, limit)

    out: list[dict] = []
    for kind, draft, target in pairs:
        draft_t = _fit_paras(draft, MAX_SFT_DRAFT_CHARS)
        target_t = _fit_paras(target, MAX_SFT_TARGET_CHARS)
        piece = Piece(
            id=f"synthetic_multipara_{kind}_{len(out)}",
            source="demo",
            text=target_t,
            year=2026,
            word_count=len(target_t.split()),
        )
        row = _example_row(
            draft=draft_t, target=target_t, piece=piece, pair_kind=kind
        )
        if row is None:
            continue
        out.append(row)
        # Heavy oversample — multipara cadence must stay ≥ A− (don't drown in clean_flat).
        for _ in range(7):
            dup = _example_row(
                draft=draft_t,
                target=target_t,
                piece=piece,
                pair_kind=f"{kind}_multipara",
            )
            if dup is not None:
                out.append(dup)
    return out


def build_sft_jsonl(pieces: Iterable[Piece], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for piece in pieces:
            if not piece.text.strip():
                continue
            for ex in piece_to_examples(piece):
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1
        for ex in _synthetic_short_cadence_examples():
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
        for ex in _synthetic_multipara_cadence_examples():
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
        for ex in _synthetic_clean_flat_voice_examples():
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
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
