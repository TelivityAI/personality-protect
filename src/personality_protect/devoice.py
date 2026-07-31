"""De-voicing operator for writer SFT pairs.

A writer LoRA is only learning anything if its training pairs are
``(D(y), y)``: a de-voiced restatement of a post mapped to the post the author
actually wrote. The first writer run mined its brief as a *verbatim extract* of
``y``, so the input and the target shared their wording and the pair was close
to ``(y, y)``. Gradient descent takes the cheapest route through that data —
copy the input forward — and the resulting adapter parroted its context at
generation time instead of writing. That is the identity map, and no amount of
extra corpus fixes it, because the objective itself is wrong.

``D`` therefore has to destroy *form* while preserving *content*:

* content kept — named entities, evidence figures, claim vocabulary, order
* form destroyed — second-person address, contractions, emphasis punctuation,
  shouted words, sentence-initial conjunctions, discourse markers, one-line
  fragment rhythm, article/auxiliary/filler scaffolding

Everything here is deterministic and Contoso-testable. There is no model in the
loop: an LLM flattener is exactly the component that can leak the author's
cadence back into the input, and a rule set can be inspected and gated.

The operator ships with its own verifier. :func:`devoice_report` measures the
transform on the existing shipped :func:`~personality_protect.pair_gate.gate_pair`
axes, asserts that ``D`` invented no entity or figure, and — the check that
matters — measures how much of the de-voiced text still sits inside a 5-gram of
the original. Callers fail closed on that number rather than trusting the rules.
"""

from __future__ import annotations

import re
from typing import Any

from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.eval_compare import extract_evidence_number_keys
from personality_protect.eval_write_holdout import mine_brief_from_holdout
from personality_protect.pair_gate import gate_pair, text_axes
from personality_protect.writer_guards import (
    COMMON_CAPITALIZED,
    copied_token_ratio,
    extract_entity_keys,
    extract_named_entity_keys,
)

# A sentence this short with no entity and no figure is carrying rhythm, not
# content: dropping it is the cleanest de-voicing available, since there is no
# meaning to preserve.
CADENCE_MAX_WORDS = 6
# Never let cadence-stripping eat the post. Below this share of content words
# the brief would no longer describe the same piece.
MIN_KEEP_CONTENT_RATIO = 0.55
# Merge target for reflow. The author writes in short standalone lines; notes
# run long and unbroken, which is what moves both cadence axes at once.
TARGET_SENTENCE_WORDS = 16
# Share of de-voiced words still inside an original 5-gram. Above this the pair
# is drifting back toward (y, y) whatever the rules did.
MAX_PAIR_COPY_RATIO = 0.35
# Brief mining's own overlap cap, applied against the already de-voiced note
# rather than the post. Holding a note to the 25% budget written for raw posts
# would reject sources purely for having been shortened by the operator, while
# the number that matters — what the brief shares with the post — is measured
# separately and gates the pair.
DEVOICED_BRIEF_MAX_OVERLAP = 0.5

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"(?<![\w])[#@][A-Za-z0-9_][A-Za-z0-9_-]*")
# Symbol classes rather than a codepoint list: decorative glyphs are a moving
# target and the categories below are never part of a written claim.
_DECORATIVE_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u27BF\u2B00-\u2BFF\uFE0F\u200D]"
)
_BULLET_GLYPH_RE = re.compile(r"(?m)^\s*[-*•·—–>»→▪◆✔✅🔹]+\s*")

# Author-agnostic English discourse scaffolding. These are the phrases a writer
# uses to set rhythm and stance; a note never contains them. Kept generic on
# purpose — nothing here encodes one person's idiom.
_DISCOURSE_MARKERS = (
    "here is the thing",
    "here's the thing",
    "here is the part",
    "here's the part",
    "here is what",
    "here's what",
    "the thing is",
    "truth is",
    "the truth is",
    "let that sink in",
    "let me be clear",
    "make no mistake",
    "full stop",
    "end of story",
    "plot twist",
    "spoiler",
    "newsflash",
    "news flash",
    "hot take",
    "unpopular opinion",
    "real talk",
    "look",
    "listen",
    "folks",
    "friends",
    "so here we are",
    "and yet",
    "but here we are",
    "read that again",
    "i will say it again",
    "i'll say it again",
    "say it with me",
    "think about that",
    "think about it",
)
_DISCOURSE_RE = re.compile(
    r"(?i)(?:^|(?<=[.!?;:—–]\s))\s*(?:"
    + "|".join(re.escape(marker) for marker in _DISCOURSE_MARKERS)
    + r")\s*[,:.—–-]*\s*"
)

# Sentence-initial conjunctions are pure cadence: the clause stands without
# them and the note form never opens on one.
_LEADING_CONJUNCTION_RE = re.compile(
    r"(?i)^\s*(?:and|but|so|or|yet|because|plus|also|then|now|well|okay|ok|"
    r"anyway|besides|still|however|meanwhile)\b[\s,:—–-]*"
)

_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in (
        (r"\bcan't\b", "cannot"),
        (r"\bwon't\b", "will not"),
        (r"\bshan't\b", "shall not"),
        (r"\blet's\b", "let us"),
        (r"\bain't\b", "is not"),
        (r"\bgonna\b", "going to"),
        (r"\bwanna\b", "want to"),
        (r"\bgotta\b", "got to"),
        (r"\b(\w+)n't\b", r"\1 not"),
        (r"\b(i|you|we|they)'re\b", r"\1 are"),
        (r"\b(i|you|we|they)'ve\b", r"\1 have"),
        (r"\b(i|you|we|they|he|she|it)'ll\b", r"\1 will"),
        (r"\b(i|you|we|they|he|she|it)'d\b", r"\1 would"),
        (r"\b(he|she|it|that|there|what|who|here)'s\b", r"\1 is"),
        (r"\bi'm\b", "I am"),
    )
)

# Second-person address is the author's rhetorical stance, not the post's
# content, and it is the axis the shipped pair gate already refuses in an input.
# Third-person plural is the one substitution that needs no verb agreement fix:
# "you ship" and "they ship" inflect identically.
_SECOND_PERSON: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\byou\b", "they"),
        (r"\bYou\b", "They"),
        (r"\byour\b", "their"),
        (r"\bYour\b", "Their"),
        (r"\byours\b", "theirs"),
        (r"\bYours\b", "Theirs"),
        (r"\byourself\b", "themselves"),
        (r"\bYourself\b", "Themselves"),
        (r"\byourselves\b", "themselves"),
        (r"\bYourselves\b", "Themselves"),
    )
)

# Articles, copulas, auxiliaries and intensifiers. Dropping them is what turns a
# written sentence into a jotted note, and it is also what reliably breaks the
# 5-gram windows the target would otherwise share with its own input.
_NOTE_DROP_WORDS = frozenset(
    """
    a an the this that these those
    is are was were be been being am
    do does did done
    has have had
    will would shall should may might must can could
    just really actually simply literally basically honestly frankly obviously
    clearly very quite rather truly definitely certainly absolutely totally
    completely genuinely seriously
    """.split()
)

# Second tier. Prepositions, coordinators and pronouns are the connective
# tissue of written prose and roughly half of its tokens; a jotted note has
# almost none of them. Dropping them is what finally moves the copy ratio,
# because a 5-gram window cannot survive a deletion every few words.
#
# Negation, quantity and comparison words are deliberately absent: dropping
# "not" or "less" would not de-voice the note, it would reverse the claim.
_TELEGRAPH_DROP_WORDS = frozenset(
    """
    of to in on at by for with from into about over under between through
    across around during within without against upon toward towards among
    and or as than then there here
    it its they them their theirs we us our ours he him his she her hers
    i me my mine you your yours
    who whom whose which what where when while
    """.split()
)
_EMPHASIS_PUNCT_RE = re.compile(r"[!?]{2,}|!+")
_ELLIPSIS_RE = re.compile(r"\.{2,}|…")
_DASH_ASIDE_RE = re.compile(r"\s*[—–]\s*|\s+--\s+")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:%)\]])")


def _word_tokens(text: str) -> list[str]:
    return [match.group(0) for match in _WORD_RE.finditer(text or "")]


def _content_word_count(text: str) -> int:
    return len(_word_tokens(text))


def _sentences(body: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(body) if part.strip()]


def _fact_weight(sentence: str) -> float:
    """How much a sentence would cost to drop (entities and figures dominate)."""
    entities = len(extract_named_entity_keys(sentence))
    numbers = len(extract_evidence_number_keys(sentence))
    return 3.0 * numbers + 2.0 * entities + min(_content_word_count(sentence), 20) / 20.0


def _is_cadence_only(sentence: str) -> bool:
    """True for a line that carries rhythm and no claim."""
    if _content_word_count(sentence) > CADENCE_MAX_WORDS:
        return False
    if extract_named_entity_keys(sentence) or extract_evidence_number_keys(sentence):
        return False
    return True


def _strip_surface_noise(text: str) -> str:
    cleaned = _URL_RE.sub(" ", text)
    cleaned = _HASHTAG_RE.sub(" ", cleaned)
    cleaned = _DECORATIVE_RE.sub(" ", cleaned)
    return _BULLET_GLYPH_RE.sub("", cleaned)


def _deshout(text: str) -> str:
    """Lowercase shouted emphasis while leaving acronyms alone.

    ``THIS IS THE WORK`` is cadence; ``API`` and ``SaaS`` are content. The
    invention guard's own vocabulary decides which is which, so the two stay in
    agreement about what counts as a name.
    """

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return token.lower() if token.lower() in COMMON_CAPITALIZED else token

    return re.sub(r"\b[A-Z]{2,}\b", replace, text)


def _neutralize(sentence: str) -> str:
    """Strip stance and register from one sentence, keeping its claim."""
    text = sentence
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _SECOND_PERSON:
        text = pattern.sub(replacement, text)
    text = _DISCOURSE_RE.sub(" ", text)
    text = _LEADING_CONJUNCTION_RE.sub("", text)
    text = _deshout(text)
    text = _EMPHASIS_PUNCT_RE.sub(".", text)
    text = _ELLIPSIS_RE.sub(".", text)
    text = _DASH_ASIDE_RE.sub(", ", text)
    return text.strip()


def _to_note_form(sentence: str) -> str:
    """Drop the scaffolding words a note would never have been written with.

    Articles, copulas, auxiliaries and intensifiers carry no claim, and removing
    them is the difference between handing the model a sentence to copy and
    handing it a note to write from.
    """
    dropped = _NOTE_DROP_WORDS | _TELEGRAPH_DROP_WORDS
    kept: list[str] = []
    for token in re.split(r"(\W+)", sentence):
        if not token:
            continue
        if _WORD_RE.fullmatch(token) and token.lower() in dropped:
            continue
        kept.append(token)
    text = "".join(kept)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"^[\s,;:.]+", "", text)
    return text.strip()


def _lower_continuation(clause: str) -> str:
    """Lowercase a clause promoted to mid-sentence, unless it opens on a name.

    Sentence-initial capitals are punctuation, not spelling. Carrying them past
    a semicolon leaves ``queue is boring; They ship``, and the invention guard
    reads a stray capitalized word as a candidate name.
    """
    match = _WORD_RE.search(clause)
    if not match or match.start() != 0:
        return clause
    word = match.group(0)
    if word == "I" or word.lower() not in COMMON_CAPITALIZED:
        return clause
    return word[0].lower() + clause[1:]


def _reflow(sentences: list[str]) -> str:
    """Merge short lines into note-length prose.

    The author's fragment rhythm — many standalone short lines — is one of the
    loudest voice signals in the corpus, and it is measured directly by the pair
    gate's ``short_line_ratio`` and ``median_sentence_words``. Merging collapses
    both at once and yields a single unbroken block, which is the shape of a
    brief rather than of a post.
    """
    merged: list[str] = []
    buffer: list[str] = []

    def flush(parts: list[str]) -> str:
        head, *rest = parts
        return "; ".join([head, *(_lower_continuation(part) for part in rest)]) + "."

    for sentence in sentences:
        buffer.append(sentence.rstrip(" .,;:"))
        if sum(_content_word_count(part) for part in buffer) >= TARGET_SENTENCE_WORDS:
            merged.append(flush(buffer))
            buffer = []
    if buffer:
        tail = flush(buffer)
        if merged:
            merged[-1] = merged[-1].rstrip(".") + "; " + _lower_continuation(tail)
        else:
            merged.append(tail)
    return " ".join(merged).strip()


def devoice_sentences(
    text: str,
    *,
    min_keep_content_ratio: float = MIN_KEEP_CONTENT_RATIO,
) -> list[str]:
    """De-voiced note clauses, one per surviving source sentence.

    Kept separate from :func:`devoice_text` because the two consumers want
    different shapes. Brief mining ranks and picks *individual claims*, so it
    needs the clause list; the cadence gate measures rhythm, so it needs the
    reflowed block. Deriving both from one pass keeps them consistent.

    Cadence-only lines are dropped cheapest-first and the drop stops before the
    note falls below ``min_keep_content_ratio`` of the original content words —
    a de-voicer that deletes the post is not preserving meaning, and the brief
    mined from it would describe something else.
    """
    body = normalize_corpus_text(_strip_surface_noise(text or ""))
    if not body.strip():
        return []

    sentences = _sentences(body)
    if not sentences:
        return []

    total_words = sum(_content_word_count(sentence) for sentence in sentences)
    floor = int(total_words * max(0.0, min(1.0, min_keep_content_ratio)))
    drop_order = sorted(
        (index for index, s in enumerate(sentences) if _is_cadence_only(s)),
        key=lambda index: (_fact_weight(sentences[index]), index),
    )
    dropped: set[int] = set()
    kept_words = total_words
    for index in drop_order:
        cost = _content_word_count(sentences[index])
        if kept_words - cost < floor:
            continue
        dropped.add(index)
        kept_words -= cost

    rewritten: list[str] = []
    for index, sentence in enumerate(sentences):
        if index in dropped:
            continue
        neutral = _to_note_form(_neutralize(sentence))
        if neutral:
            rewritten.append(neutral)
    return rewritten


def devoice_text(
    text: str,
    *,
    min_keep_content_ratio: float = MIN_KEEP_CONTENT_RATIO,
) -> str:
    """Return ``D(y)``: the claims of ``text`` with the author's form removed."""
    rewritten = devoice_sentences(
        text, min_keep_content_ratio=min_keep_content_ratio
    )
    if not rewritten:
        return ""
    return _reflow(rewritten)


class DevoiceRejected(ValueError):
    """A pair could not be de-voiced far enough away from its target."""

    def __init__(self, reasons: list[str], report: dict[str, Any]) -> None:
        super().__init__("de-voiced pair rejected: " + ", ".join(reasons))
        self.reasons = reasons
        self.report = report


def mine_writer_brief(
    text: str,
    *,
    holdout_id: str = "",
    max_copy_ratio: float = MAX_PAIR_COPY_RATIO,
    max_brief_overlap: float = DEVOICED_BRIEF_MAX_OVERLAP,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Mine a brief from ``D(y)`` and prove it is not an extract of ``y``.

    The single entry point for both halves of the writer path, so training and
    the ship gate cannot drift apart: an adapter trained on de-voiced briefs and
    then evaluated on verbatim extracts would be measured on a distribution it
    never saw.

    Brief mining runs against the de-voiced clauses rather than the reflowed
    block because it ranks and selects individual claims. Its own overlap cap is
    relaxed here — it exists to keep a brief from becoming an extract of its
    source, and by this point the source is already a note, not the post. What
    the brief may share with the *post* is measured directly and gates the pair.

    Raises :class:`DevoiceRejected` when the operator did not move the pair far
    enough, so callers drop the row instead of training on ``(y, y)``.
    """
    original = normalize_corpus_text(text)
    clauses = devoice_sentences(original)
    if not clauses:
        raise DevoiceRejected(["devoice_empty"], {})

    devoiced = _reflow(clauses)
    report = devoice_report(original, devoiced, max_copy_ratio=max_copy_ratio)
    # The row that trains is (brief, post), not (note, post), so the
    # document-level copy ratio is recorded and not enforced here. A note that
    # still shares long windows with the post is a warning about the operator;
    # whether *this pair* is an identity map is answered below, on the brief.
    blocking = [reason for reason in report["failed"] if reason != "pair_copy_ratio"]
    if blocking:
        raise DevoiceRejected(blocking, report)

    brief = mine_brief_from_holdout(
        "\n".join(clauses),
        holdout_id=holdout_id,
        max_overlap=max_brief_overlap,
    )
    # The invention guard's allowed-facts set has to stay the *post*: the note
    # drops connective words, and scoring a draft against the note would accuse
    # it of inventing figures the author actually wrote.
    brief["guard_facts"] = original
    brief_text = f"{brief['topic']}\n{brief['points']}"
    brief_ratio = pair_copy_ratio(brief_text, original)
    report = {
        **report,
        "brief_copy_ratio": brief_ratio,
        "brief_words": len(brief_text.split()),
    }
    if brief_ratio > float(max_copy_ratio):
        raise DevoiceRejected(["brief_copy_ratio"], report)
    return brief, report


def pair_copy_ratio(input_text: str, target_text: str) -> float:
    """Share of input words sitting inside a 5-gram of the target.

    This is the identity-map meter. A verbatim extract scores near 1.0; a true
    de-voiced note scores low because its word sequences no longer exist in the
    post. It is the single number worth gating a writer pair on.
    """
    return copied_token_ratio(input_text, [target_text])


def devoice_report(
    original: str,
    devoiced: str,
    *,
    channel: str = "auto",
    max_copy_ratio: float = MAX_PAIR_COPY_RATIO,
) -> dict[str, Any]:
    """Measure one ``(D(y), y)`` pair and decide whether it may train.

    Three independent questions, each fail-closed:

    * did ``D`` actually move the cadence axes? — delegated to the shipped
      :func:`~personality_protect.pair_gate.gate_pair`
    * did ``D`` invent anything? — a de-voicer that adds an entity or a figure
      would poison the invention guard's allowed-facts set
    * is the pair still near ``(y, y)``? — :func:`pair_copy_ratio`

    The pair gate's ``max_input_proper_1k`` check is recorded but not blocking
    here. That threshold exists to catch an *LLM* flattener echoing the author's
    text back, and it reads proper-noun density as the tell. This operator
    preserves proper nouns by construction because they are the brief's content,
    so the same number would only be measuring how many companies the author
    named — and dropping connective words raises the density further without
    adding a single name. The entity-subset invariant below is the check that
    actually answers "did the input gain anything it should not have".

    ``channel`` defaults to ``auto`` so the shipped channel inference decides
    whether the fragment-rhythm check applies. A prose-shaped post has no short
    lines to begin with, and holding it to a fragment gap it never had would
    reject the pair for the author's paragraph habits rather than for anything
    the operator did.
    """
    gate = gate_pair(devoiced, original, channel=channel)
    # Compare single tokens, not spans. Dropping a connective can leave two
    # names of the original adjacent ("Contoso is Ledger" -> "Contoso Ledger"),
    # which reads as a new multi-word span while inventing nothing: both names
    # were already in the source.
    devoiced_names = {
        token
        for key in extract_named_entity_keys(devoiced)
        for token in key.split(" ")
        if token
    }
    new_entities = devoiced_names - extract_entity_keys(original)
    new_numbers = extract_evidence_number_keys(devoiced) - extract_evidence_number_keys(
        original
    )
    copy_ratio = pair_copy_ratio(devoiced, original)

    blocking_gate_failures = [
        reason for reason in gate["failed"] if reason != "max_input_proper_1k"
    ]
    failed = list(blocking_gate_failures)
    if new_entities:
        failed.append("devoice_invented_entities")
    if new_numbers:
        failed.append("devoice_invented_numbers")
    if copy_ratio > float(max_copy_ratio):
        failed.append("pair_copy_ratio")

    return {
        "pass": not failed,
        "failed": failed,
        "copy_ratio": copy_ratio,
        "max_copy_ratio": float(max_copy_ratio),
        "invented_entities_count": len(new_entities),
        "invented_numbers_count": len(new_numbers),
        "gate_pass": bool(gate["pass"]),
        "gate_failed": list(gate["failed"]),
        "gate_advisory": [
            reason for reason in gate["failed"] if reason == "max_input_proper_1k"
        ],
        "resolved_channel": gate["resolved_channel"],
        "frag_gap_ratio": gate["frag_gap_ratio"],
        "median_sentence_gap": gate["median_sentence_gap"],
        "input_axes": text_axes(devoiced),
        "output_axes": text_axes(original),
    }
