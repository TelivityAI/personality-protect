"""Holdout eval for RAG write vs bare-base (Camp A Lane G).

Carve holdout pieces that were never indexed, mine briefs from them, draft with
RAG and with a bare-base (no-exemplar) prompt, then score Contoso-safe receipts.

MLX is never imported here. Callers inject ``generate_fn`` in tests; the CLI
defaults to :func:`personality_protect.write.mlx_generate_no_adapter` which
gates Metal behind ``PP_MLX_ALLOW``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personality_protect.chat_prompt import flatten_chat_messages
from personality_protect.config import ProfilePaths, load_config
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.draft_trim import trim_draft
from personality_protect.eval_compare import extract_evidence_number_keys
from personality_protect.models import Piece, load_index
from personality_protect.pair_gate import text_axes
from personality_protect.prompt_write import build_write_messages
from personality_protect.style_profile import DEFAULT_DRAFT_WORD_TARGET
from personality_protect.voice_index import VECTORS_FILENAME
from personality_protect.write import (
    DEFAULT_WRITE_K,
    DEFAULT_WRITE_MAX_TOKENS,
    GenerateFn,
    build_brief,
    mlx_generate_no_adapter,
    normalize_sentence_case,
    run_write,
)
from personality_protect.writer_guards import (
    brief_echo_reject,
    check_invention,
    extract_named_entity_keys,
    parrot_reject,
)

TIE_EPSILON = 0.05
BARE_BASE_EXAMPLES: tuple[str, ...] = ()
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOPIC_WORD_CAP = 10
_TOPIC_MIN_WORDS = 4
# A 7-word cap forced every bullet to be cut mid-sentence. Clause-boundary
# fitting needs room to land on a complete claim; the overlap cap below is what
# actually keeps the brief lossy.
_POINT_WORD_CAP = 12
_MIN_POINTS = 2
_MAX_POINTS = 3
# Below this a bullet is a fragment, not a claim worth handing to the model.
_MIN_POINT_WORDS = 3
# The visible brief is deliberately lossy. This absolute cap prevents long
# holdouts from receiving a long extract, while the overlap cap protects short
# holdouts. Both count normalized words, not formatting markers.
_MAX_BRIEF_WORDS = 40
_MAX_BRIEF_OVERLAP = 0.25
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+|\s+[—–]\s+")
# Words that cannot end a claim. Kept deliberately small: articles, copulas,
# conjunctions, prepositions, and bare temporal adverbs.
_DANGLING_TAIL_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "if", "then",
        "than", "that", "which", "who", "whom", "whose", "as", "because",
        "while", "when", "where", "is", "are", "was", "were", "be", "been",
        "being", "am", "do", "does", "did", "has", "have", "had", "will",
        "would", "can", "could", "should", "may", "might", "must", "of", "to",
        "in", "on", "at", "by", "for", "with", "from", "into", "about", "over",
        "under", "between", "through", "after", "before", "now", "just", "also",
        "very", "not", "no", "more", "most", "one", "its", "it", "their",
        "our", "your", "his", "her", "my",
    }
)
_NUMBER_TAIL_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
        "million", "billion",
    }
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
def min_briefable_words(max_overlap: float = _MAX_BRIEF_OVERLAP) -> int:
    """Shortest source that can yield a topic and two bullets inside ``max_overlap``."""
    return math.ceil((_TOPIC_MIN_WORDS + _MIN_POINTS * _MIN_POINT_WORDS) / max_overlap)


_MIN_HOLDOUT_WORDS = min_briefable_words()

# Raw prompts and drafts contain personal text. They live under the profile
# directory (already gitignored, and outside the repo) and are never surfaced
# in a receipt.
RAW_SUBDIR = ("dogfood", "raw")

# Contoso tokens that must never appear in public receipts (personal leak check).
_RECEIPT_BANNED = (
    "linkedin.com",
    "@gmail",
    "dusan",
    "telivity.ai/private",
)


def _voice_index_dir(paths: ProfilePaths) -> Path:
    return paths.root / "voice_index"


def load_holdout_pieces(
    paths: ProfilePaths,
    holdout_ids: Iterable[str],
) -> list[Piece]:
    """Load corpus pieces matching holdout ids (order preserved by id sort)."""
    wanted = {str(piece_id) for piece_id in holdout_ids}
    if not wanted:
        raise ValueError("holdout_ids must not be empty")
    pieces = load_index(paths.index_path)
    by_id = {piece.id: piece for piece in pieces}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise FileNotFoundError(f"Holdout ids not in corpus index: {', '.join(missing)}")
    return [by_id[piece_id] for piece_id in sorted(wanted)]


def indexed_piece_ids(paths: ProfilePaths) -> set[str]:
    """Return ids currently present in the voice retrieval index."""
    vectors_path = _voice_index_dir(paths) / VECTORS_FILENAME
    if not vectors_path.is_file():
        return set()
    ids: set[str] = set()
    with vectors_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            piece = row.get("piece") or {}
            piece_id = piece.get("id")
            if piece_id is not None:
                ids.add(str(piece_id))
    return ids


def verify_holdouts_never_indexed(
    paths: ProfilePaths,
    holdout_ids: Iterable[str],
) -> dict[str, Any]:
    """g1: confirm holdout ids are absent from voice_index vectors."""
    wanted = {str(piece_id) for piece_id in holdout_ids}
    indexed = indexed_piece_ids(paths)
    leaked = sorted(wanted & indexed)
    return {
        "ok": not leaked,
        "holdout_ids": sorted(wanted),
        "indexed_holdout_ids": leaked,
        # Profile-relative only — never absolute home paths (Contoso-safe).
        "profile": paths.name,
        "voice_index": "voice_index",
        "indexed_total": len(indexed),
    }


def _sentences(body: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(body) if part.strip()]


def _truncate_words(text: str, word_cap: int) -> str:
    """Keep at most ``word_cap`` lexical words without splitting the last one."""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= word_cap:
        return text.strip()
    return text[: matches[max(1, word_cap) - 1].end()].strip()


def _strip_trailing_punctuation(text: str) -> str:
    return text.strip().rstrip(".,;:—-").strip()


def _is_dangling_tail(token: str) -> bool:
    lowered = token.casefold()
    # A severed quantity ("...by twelve percent over seven") reads as a fragment
    # once its unit is gone, so a trailing bare number goes with the cut.
    return lowered in _DANGLING_TAIL_WORDS or lowered in _NUMBER_TAIL_WORDS or token.isdigit()


def _strip_dangling(text: str) -> str:
    """Drop trailing words that leave a cut phrase grammatically mid-air.

    Cutting a sentence short lands on words like "is" or "and", and a bullet
    reading "IATA is now" is worse than no bullet: the writer copies the fragment
    instead of making a claim. Only applied where a cut happened — a sentence
    that fit whole is left exactly as the author ended it.
    """
    cleaned = _strip_trailing_punctuation(text)
    while True:
        tokens = _WORD_RE.findall(cleaned)
        if not tokens or not _is_dangling_tail(tokens[-1]):
            return cleaned
        end = list(_WORD_RE.finditer(cleaned))[-1].start()
        cleaned = _strip_trailing_punctuation(cleaned[:end])


def _fit_phrase(sentence: str, word_cap: int) -> str:
    """Longest self-contained phrase from ``sentence`` fitting ``word_cap``.

    Whole sentence when it fits; otherwise the longest run of leading clauses
    that fits, so cuts land on punctuation the author wrote. Word-count
    truncation is the last resort and always has its dangling tail removed.
    """
    flat = re.sub(r"\s+", " ", _URL_RE.sub("", sentence)).strip()
    if not flat:
        return ""
    if len(_word_tokens(flat)) <= word_cap:
        return _strip_trailing_punctuation(flat)

    clauses = [clause for clause in _CLAUSE_SPLIT.split(flat) if clause.strip()]
    kept: list[str] = []
    for clause in clauses:
        candidate = " ".join([*kept, clause.strip()])
        if len(_word_tokens(candidate)) > word_cap:
            break
        kept.append(clause.strip())
    if kept:
        return _strip_dangling(" ".join(kept))
    return _strip_dangling(_truncate_words(flat, word_cap))


def _mine_topic_with_source(
    sentences: Sequence[str],
    *,
    word_cap: int = _TOPIC_WORD_CAP,
) -> tuple[str, int]:
    """Subject line plus the index of the sentence it came from.

    Openers are often interjections ("Sweet lord."). Concatenating them onto the
    next sentence produced topics that read as a stray exclamation followed by a
    severed clause, so short leads are skipped rather than glued. The index lets
    the caller keep the topic's own sentence out of the bullet pool.
    """
    for index, sentence in enumerate(sentences):
        cleaned = _URL_RE.sub("", sentence).strip()
        if not cleaned or len(_word_tokens(cleaned)) < _TOPIC_MIN_WORDS:
            continue
        fitted = _fit_phrase(cleaned, word_cap)
        if len(_word_tokens(fitted)) >= _TOPIC_MIN_WORDS:
            return fitted, index
    # Nothing stands alone: fall back to concatenation so a thin holdout still
    # yields a topic instead of raising.
    joined = " ".join(_URL_RE.sub("", s).strip() for s in sentences if s.strip())
    return _strip_dangling(_truncate_words(joined, word_cap)), 0


def _mine_topic(sentences: Sequence[str], *, word_cap: int = _TOPIC_WORD_CAP) -> str:
    """Subject line mined from the holdout's opening sentences."""
    return _mine_topic_with_source(sentences, word_cap=word_cap)[0]


def _fact_score(sentence: str) -> float:
    """Rank sentences by how much of the post's substance they carry."""
    entities = len(extract_named_entity_keys(sentence))
    numbers = len(extract_evidence_number_keys(sentence))
    words = len(_word_tokens(_URL_RE.sub("", sentence)))
    return 2.0 * entities + 3.0 * numbers + min(words, 20) / 20.0


def _to_bullet(sentence: str, *, word_cap: int = _POINT_WORD_CAP) -> str:
    """Collapse a sentence to one terse, single-line bullet."""
    return "- " + _fit_phrase(sentence, word_cap)


def _content_word_count(topic: str, points: str) -> int:
    """Words the brief actually hands over, ignoring bullet markers.

    The ``- `` prefixes are our formatting, not the author's text, so counting
    them would overstate leakage and shrink the real fact budget.
    """
    point_words = [word for word in points.split() if word not in {"-", "*", "•"}]
    return len(topic.split()) + len(point_words)


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD_RE.finditer(text or "")]


def brief_word_overlap_ratio(brief: dict[str, str], holdout_text: str) -> float:
    """Share of source-word occurrences copied into the model-visible brief."""
    source = Counter(_word_tokens(normalize_corpus_text(holdout_text)))
    brief_tokens = Counter(_word_tokens(f"{brief['topic']}\n{brief['points']}"))
    overlap = sum(min(count, source[token]) for token, count in brief_tokens.items())
    return round(overlap / max(1, sum(source.values())), 4)


def mine_brief_from_holdout(
    text: str,
    *,
    holdout_id: str = "",
    max_points: int = _MAX_POINTS,
    max_overlap: float = _MAX_BRIEF_OVERLAP,
) -> dict[str, str]:
    """g2: deterministically mine a lossy brief (short topic + 2–3 bullets).

    A brief is what the author would jot down *before* writing: the subject and
    a few key claims. It is emphatically **not** an extract of the finished post.
    The visible brief has both a hard word cap and a source-overlap cap. Bullets
    are picked by :func:`_fact_score` and re-emitted in document order.

    ``guard_facts`` is intentionally separate from what the model sees. It uses
    the source post as the internal allowed-facts set, so the invention guard
    can reject facts absent from the source without leaking those facts into the
    generation prompt. Receipts serialize neither field.
    """
    body = normalize_corpus_text(text)
    if not body:
        raise ValueError("holdout text must not be empty")

    holdout_words = len(_word_tokens(body))
    # Derive the floor from the caller's budget instead of the module default.
    # The two only diverge for a source that is already de-voiced, where a
    # looser overlap cap is correct and a floor pinned to 25% would reject
    # sources long enough to brief.
    min_words = min_briefable_words(max_overlap)
    if holdout_words < min_words:
        raise ValueError(
            f"holdout is {holdout_words} words — too short to brief without "
            f"handing back more than {max_overlap:.0%} of it; "
            "pick a longer holdout rather than relaxing the budget"
        )

    budget = min(_MAX_BRIEF_WORDS, int(holdout_words * max_overlap))
    sentences = _sentences(body)
    topic, topic_index = _mine_topic_with_source(
        sentences,
        word_cap=min(_TOPIC_WORD_CAP, budget - _MIN_POINTS * _MIN_POINT_WORDS),
    )

    candidates = [s for i, s in enumerate(sentences) if i != topic_index]
    if len(candidates) < _MIN_POINTS:
        raise ValueError("holdout needs at least three sentences to mine two lossy bullets")
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (-_fact_score(item[1]), item[0]),
    )

    remaining = budget - len(topic.split())
    chosen: list[tuple[int, str]] = []
    target_points = min(_MAX_POINTS, max(_MIN_POINTS, int(max_points)))
    for position, sentence in ranked:
        if len(chosen) >= target_points:
            break
        # Do not let the first high-ranked bullet consume the minimum budget
        # required for a second one.
        minimum_after = max(0, (_MIN_POINTS - len(chosen) - 1) * _MIN_POINT_WORDS)
        word_cap = min(_POINT_WORD_CAP, remaining - minimum_after)
        if word_cap < _MIN_POINT_WORDS:
            break
        bullet = _to_bullet(sentence, word_cap=word_cap)
        bullet_words = len(_word_tokens(bullet))
        if bullet_words < _MIN_POINT_WORDS:
            continue
        remaining -= bullet_words
        chosen.append((position, bullet))

    if len(chosen) < _MIN_POINTS:
        raise ValueError("brief budget cannot fit two short bullets")
    points = "\n".join(bullet for _, bullet in sorted(chosen, key=lambda item: item[0]))
    brief = {
        "holdout_id": holdout_id,
        "topic": topic,
        "points": points,
        "guard_facts": body,
    }
    if _content_word_count(topic, points) > _MAX_BRIEF_WORDS:
        raise ValueError("mined brief exceeds hard word cap")
    overlap = brief_word_overlap_ratio(brief, body)
    if overlap > max_overlap:
        raise ValueError(
            f"mined brief overlaps {overlap:.0%} of the holdout "
            f"(max {max_overlap:.0%}) — mining is not lossy"
        )
    return brief


def run_bare_base_write(
    topic: str,
    points: str,
    *,
    generate_fn: GenerateFn,
    base_model: str,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
    word_target: int = DEFAULT_DRAFT_WORD_TARGET,
    prompt_sink: list[str] | None = None,
) -> dict[str, Any]:
    """Generate from the locked prompt with no retrieved exemplars.

    The same tail trim as the RAG arm: a length edit applied to only one side
    would decide the comparison by itself.
    """
    topic = topic.strip()
    points = points.strip()
    if not topic or not points:
        raise ValueError("topic and points must not be empty")
    messages = build_write_messages(topic=topic, points=points, examples=BARE_BASE_EXAMPLES)
    draft = trim_draft(
        str(
            generate_fn(
                messages,
                base_model=base_model,
                max_tokens=max_tokens,
                prompt_sink=prompt_sink,
            )
        ).strip(),
        max_words=word_target,
    )
    brief = build_brief(topic, points)
    invention = check_invention(brief, normalize_sentence_case(draft))
    return {
        "text": draft,
        "mode": "bare_base",
        "adapter": "none",
        "model": base_model,
        "k": 0,
        "exemplar_ids": [],
        "parrot_reject": parrot_reject(draft, ()),
        "invent_reject": not invention.passed,
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
        "messages": messages,
        "prompt": flatten_chat_messages(messages),
    }


def raw_artifacts_dir(paths: ProfilePaths) -> Path:
    """Local, gitignored home for raw prompts and drafts (personal text)."""
    return paths.root.joinpath(*RAW_SUBDIR)


def write_raw_artifacts(
    paths: ProfilePaths,
    *,
    holdout_id: str,
    arm: str,
    prompt: str,
    draft: str,
    brief: dict[str, str] | None = None,
) -> dict[str, str]:
    """Persist the exact prompt and raw draft so a human can eyeball them.

    Nothing here is Contoso-safe by construction — it is verbatim personal
    text. It stays under the profile directory and must never be committed or
    folded into a receipt. Returns profile-relative paths only.
    """
    directory = raw_artifacts_dir(paths)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{holdout_id}.{arm}"
    written: dict[str, str] = {}
    payloads = {"prompt.txt": prompt, "draft.txt": draft}
    if brief is not None:
        payloads["brief.json"] = json.dumps(brief, indent=2, ensure_ascii=False)
    for suffix, body in payloads.items():
        target = directory / f"{stem}.{suffix}"
        target.write_text((body or "").rstrip() + "\n", encoding="utf-8")
        written[suffix] = str(Path(*RAW_SUBDIR) / target.name)
    return written


def _length_penalty(reference_words: int, candidate_words: int) -> float:
    """Penalize a draft that is not the length of a post at all.

    Symmetric in log space, so half-length and double-length cost the same. A
    stub is otherwise unpunished: with every other axis length-normalized, three
    sentences can sit at near-zero distance from a full post.
    """
    reference = max(1, int(reference_words))
    candidate = max(1, int(candidate_words))
    return abs(math.log(candidate / reference))


def _axes_distance(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    """Lower = closer to the holdout voice band.

    Every term is length-normalized. Raw pronoun counts were used here and they
    measured length rather than voice: a long draft was penalized per extra
    "you" while a three-sentence stub scored near zero, so the comparison
    rewarded not writing a post.
    """
    return round(
        abs(float(reference["short_line_ratio"]) - float(candidate["short_line_ratio"]))
        + abs(float(reference["median_sentence_words"]) - float(candidate["median_sentence_words"]))
        / 20.0
        + abs(float(reference["you_gt_i"]) - float(candidate["you_gt_i"]))
        + abs(float(reference["proper_per_1k"]) - float(candidate["proper_per_1k"])) / 50.0
        + abs(float(reference["you_per_1k"]) - float(candidate["you_per_1k"])) / 20.0
        + abs(float(reference["i_per_1k"]) - float(candidate["i_per_1k"])) / 20.0
        + _length_penalty(reference["words"], candidate["words"]),
        4,
    )


def score_draft_against_holdout(
    holdout_text: str,
    draft: str,
    brief: str,
    exemplars: Sequence[str] = (),
) -> dict[str, Any]:
    """g4: axis distance + guard flags (counts + keys; no draft body).

    ``disqualified`` marks a draft that failed a guard. Axis distance measures
    rhythm, and an exemplar dump has perfect rhythm because it *is* the author's
    text — so distance alone once crowned a draft that was a copy of its own
    prompt. A disqualified draft cannot win, whatever its distance.
    """
    ref_axes = text_axes(holdout_text)
    draft_axes = text_axes(draft)
    invention = check_invention(brief, normalize_sentence_case(draft))
    parroted = parrot_reject(draft, list(exemplars))
    # Handing the mined bullets back is the other way to score a flattering
    # distance without writing anything, and it is what the winning draft did on
    # one holdout.
    echoed = brief_echo_reject(draft, brief)
    return {
        "distance": _axes_distance(ref_axes, draft_axes),
        "axes": draft_axes,
        "parrot_reject": parroted,
        "brief_echo_reject": echoed,
        "invent_reject": not invention.passed,
        "disqualified": bool(parroted or echoed or not invention.passed),
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
        "invented_entities_count": len(invention.invented_entities),
        "invented_numbers_count": len(invention.invented_numbers),
    }


def score_rag_vs_base(
    holdout_text: str,
    rag_draft: str,
    base_draft: str,
    brief: str,
    *,
    rag_exemplars: Sequence[str] = (),
    tie_epsilon: float = TIE_EPSILON,
) -> dict[str, Any]:
    """Three-way score: holdout reference vs RAG draft vs bare-base draft."""
    holdout_axes = text_axes(holdout_text)
    rag = score_draft_against_holdout(holdout_text, rag_draft, brief, rag_exemplars)
    base = score_draft_against_holdout(holdout_text, base_draft, brief)
    delta = float(base["distance"]) - float(rag["distance"])
    if rag["disqualified"] and base["disqualified"]:
        winner = "tie"
    elif rag["disqualified"]:
        winner = "base"
    elif base["disqualified"]:
        winner = "rag"
    elif delta > float(tie_epsilon):
        winner = "rag"
    elif delta < -float(tie_epsilon):
        winner = "base"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "delta_base_minus_rag": round(delta, 4),
        "holdout_axes": holdout_axes,
        "rag": rag,
        "base": base,
        "tie_epsilon": float(tie_epsilon),
    }


def brief_leakage_ratio(brief: dict[str, str], holdout_text: str) -> float:
    """Visible brief word count divided by source word count (legacy receipt metric).

    The original harness scored 1.0 here (``points`` was the whole post) and
    every arm was really being tested on paraphrase speed. Tracked in the
    receipt so a regression is visible without reading personal text.
    """
    holdout_words = len((holdout_text or "").split()) or 1
    brief_words = _content_word_count(brief["topic"], brief["points"])
    return round(brief_words / holdout_words, 4)


def assert_brief_is_not_the_post(brief: dict[str, str], holdout_text: str) -> None:
    """Fail closed if brief mining regresses into handing over the target post."""
    if brief["points"].strip() == (holdout_text or "").strip():
        raise ValueError(
            "mined brief reproduces the holdout body verbatim — "
            "the eval would be scoring paraphrase, not voice"
        )
    ratio = brief_word_overlap_ratio(brief, holdout_text)
    if ratio > _MAX_BRIEF_OVERLAP:
        raise ValueError(
            f"mined brief returns {ratio:.0%} of the holdout "
            f"(max {_MAX_BRIEF_OVERLAP:.0%}) — tighten brief mining"
        )


def _item_receipt(
    *,
    holdout_id: str,
    brief: dict[str, str],
    holdout_text: str,
    rag_result: dict[str, Any],
    base_result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    """Contoso-safe per-holdout row: ids, scores, invent flags — never draft text."""
    return {
        "holdout_id": holdout_id,
        "topic_words": len(brief["topic"].split()),
        "points_words": len(brief["points"].split()),
        "points_bullets": len([ln for ln in brief["points"].splitlines() if ln.strip()]),
        "holdout_words": len((holdout_text or "").split()),
        "brief_leakage_ratio": brief_leakage_ratio(brief, holdout_text),
        "brief_word_overlap_ratio": brief_word_overlap_ratio(brief, holdout_text),
        "winner": score["winner"],
        "delta_base_minus_rag": score["delta_base_minus_rag"],
        "rag_distance": score["rag"]["distance"],
        "base_distance": score["base"]["distance"],
        "rag_invent_reject": score["rag"]["invent_reject"],
        "base_invent_reject": score["base"]["invent_reject"],
        "rag_parrot_reject": score["rag"]["parrot_reject"],
        "base_parrot_reject": score["base"]["parrot_reject"],
        "rag_brief_echo_reject": score["rag"]["brief_echo_reject"],
        "base_brief_echo_reject": score["base"]["brief_echo_reject"],
        "rag_disqualified": score["rag"]["disqualified"],
        "base_disqualified": score["base"]["disqualified"],
        "rag_invented_entities_count": score["rag"]["invented_entities_count"],
        "base_invented_entities_count": score["base"]["invented_entities_count"],
        "rag_invented_numbers_count": score["rag"]["invented_numbers_count"],
        "base_invented_numbers_count": score["base"]["invented_numbers_count"],
        "exemplar_ids": list(rag_result.get("exemplar_ids") or []),
        "rag_k": int(rag_result.get("k") or 0),
        "base_k": int(base_result.get("k") or 0),
        "rag_adapter": rag_result.get("adapter", "none"),
        "base_adapter": base_result.get("adapter", "none"),
        "rag_attempts": int(rag_result.get("attempts") or 1),
        "rag_draft_words": len(str(rag_result.get("text") or "").split()),
        "base_draft_words": len(str(base_result.get("text") or "").split()),
    }


def assert_receipt_contoso_safe(receipt: dict[str, Any]) -> None:
    """Fail closed if a receipt embeds personal or draft body markers."""
    blob = json.dumps(receipt, ensure_ascii=False).lower()
    for token in _RECEIPT_BANNED:
        if token in blob:
            raise ValueError(f"receipt leaks banned token: {token!r}")
    # Draft/holdout bodies must not be serialized under common text keys.
    stack: list[Any] = [receipt]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                if key in {"text", "draft", "holdout_text", "points", "topic", "prompt"}:
                    raise ValueError(f"receipt must omit body field {key!r}")
                stack.append(value)
        elif isinstance(cur, list):
            stack.extend(cur)


def run_eval_write_holdout(
    paths: ProfilePaths,
    holdout_ids: Sequence[str],
    *,
    k: int = DEFAULT_WRITE_K,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
    generate_fn: GenerateFn | None = None,
    generate_fn_base: GenerateFn | None = None,
    tie_epsilon: float = TIE_EPSILON,
    save_raw: bool = False,
) -> dict[str, Any]:
    """Run RAG + bare-base drafts per holdout and return a Contoso-safe receipt.

    ``save_raw`` dumps the exact prompt and raw draft per item and arm under the
    profile's gitignored ``dogfood/raw`` directory. Those files are personal
    text for human review only; the returned receipt never references them.
    """
    ids = [str(piece_id) for piece_id in holdout_ids]
    carve = verify_holdouts_never_indexed(paths, ids)
    if not carve["ok"]:
        raise ValueError(
            "Holdout ids are present in voice_index (data leak into retrieval): "
            + ", ".join(carve["indexed_holdout_ids"])
        )

    config = load_config(paths)
    model_id = config.base_model
    generator = generate_fn or mlx_generate_no_adapter
    base_generator = generate_fn_base or generator

    pieces = load_holdout_pieces(paths, ids)
    items: list[dict[str, Any]] = []
    wins = {"rag": 0, "base": 0, "tie": 0}

    for piece in pieces:
        holdout_text = normalize_corpus_text(piece.text)
        brief = mine_brief_from_holdout(holdout_text, holdout_id=piece.id)
        assert_brief_is_not_the_post(brief, holdout_text)

        rag_prompts: list[str] = []
        base_prompts: list[str] = []
        rag_result = run_write(
            brief["topic"],
            brief["points"],
            paths,
            k=k,
            max_tokens=max_tokens,
            generate_fn=generator,
            prompt_sink=rag_prompts,
        )
        base_result = run_bare_base_write(
            brief["topic"],
            brief["points"],
            generate_fn=base_generator,
            base_model=model_id,
            max_tokens=max_tokens,
            word_target=int(rag_result["word_target"]),
            prompt_sink=base_prompts,
        )
        # The guard scores drafts against the same brief the model saw.
        score = score_rag_vs_base(
            holdout_text,
            rag_result["text"],
            base_result["text"],
            brief["guard_facts"],
            rag_exemplars=list(rag_result.get("exemplar_texts") or []),
            tie_epsilon=tie_epsilon,
        )
        wins[score["winner"]] = wins.get(score["winner"], 0) + 1
        items.append(
            _item_receipt(
                holdout_id=piece.id,
                brief=brief,
                holdout_text=holdout_text,
                rag_result=rag_result,
                base_result=base_result,
                score=score,
            )
        )
        if save_raw:
            for arm, result, sink in (
                ("rag", rag_result, rag_prompts),
                ("bare_base", base_result, base_prompts),
            ):
                write_raw_artifacts(
                    paths,
                    holdout_id=piece.id,
                    arm=arm,
                    # Sink holds the chat-templated string the model actually
                    # saw; fall back to the flat rendering for mock backends.
                    prompt=sink[-1] if sink else str(result.get("prompt") or ""),
                    draft=str(result.get("text") or ""),
                    brief=brief,
                )

    receipt: dict[str, Any] = {
        "kind": "eval_write_holdout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "voice_mode": config.voice_mode,
        "adapter": "none",
        "model": model_id,
        "k": k,
        "n_holdouts": len(items),
        "holdout_ids": [item["holdout_id"] for item in items],
        "carve": carve,
        "brief_mining": {
            "topic_word_cap": _TOPIC_WORD_CAP,
            "point_word_cap": _POINT_WORD_CAP,
            "hard_brief_word_cap": _MAX_BRIEF_WORDS,
            "min_points": _MIN_POINTS,
            "max_points": _MAX_POINTS,
            "max_brief_overlap": _MAX_BRIEF_OVERLAP,
        },
        "raw_artifacts_saved": bool(save_raw),
        "wins": wins,
        "rag_beats_base": wins["rag"] > wins["base"],
        "items": items,
    }
    assert_receipt_contoso_safe(receipt)
    return receipt


def write_receipt(receipt: dict[str, Any], path: Path) -> Path:
    """Persist a Contoso-safe receipt JSON (gitignored evals/ recommended)."""
    assert_receipt_contoso_safe(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
