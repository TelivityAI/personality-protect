"""Deterministic safety guards for exemplar-assisted writing.

Two different questions need two different capitalized-word policies:

* **Masking** (:func:`mask_exemplar_entities`) asks "could this leak a real
  name out of a retrieved exemplar?" — fail toward masking, so it uses the
  small :data:`_NON_ENTITY_CAPS` allowlist and masks anything unfamiliar.
* **Invention** (:func:`check_invention`) asks "did the model fabricate a
  proper name the brief never gave it?" — fail toward silence, so it uses the
  much larger :data:`_COMMON_CAPITALIZED` allowlist. Flagging ``AI``, ``Match``
  or ``Brief`` as invented companies drowns the real signal, and a guard that
  fires on every draft carries no information.

Both directions are covered by tests in ``tests/test_writer_guards.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable

from personality_protect.eval_compare import extract_evidence_number_keys

# Redaction removes the name instead of substituting a token. A visible
# placeholder like ``[ENTITY]`` is an instruction the model happily follows: a
# dogfood run produced a draft that copied whole exemplars back with dozens of
# literal ``[ENTITY]`` markers in place of the names.
DEFAULT_ENTITY_MASK = ""
DEFAULT_PARROT_NGRAM = 8
# Shorter windows catch reworded copying; only applied to drafts long enough for
# the ratio to mean something, so a single shared phrase never trips it.
PARROT_COVERAGE_NGRAM = 5
PARROT_COVERAGE_LIMIT = 0.3
PARROT_COVERAGE_MIN_TOKENS = 40
# The brief is the draft's own content source, so reusing its wording is
# expected and a shared window means nothing. Only a draft that is *mostly*
# brief text has skipped the writing.
BRIEF_ECHO_LIMIT = 0.6
# Unlike the exemplar check, the reference here is a handful of specific bullets
# rather than general prose, so coverage is already meaningful on a short draft.
BRIEF_ECHO_MIN_TOKENS = 12

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z](?:&[A-Z])+|"
    r"[A-Z][a-z][a-zA-Z0-9'-]*(?:\s+[A-Z][a-z][a-zA-Z0-9'-]*)*)\b"
)
_POSSESSIVE_RE = re.compile(r"['’]s$")
_TRAILING_POSSESSIVE_RE = re.compile(r"^['’]s\b")

# Cleanups after a name is cut out, so the exemplar still reads as prose rather
# than as a form with blanks to fill in.
_TIDY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"[ \t]+([,.;:!?%)\]])"), r"\1"),
    (re.compile(r"([(\[#@])[ \t]+"), r"\1"),
    (re.compile(r"\(\s*\)|\[\s*\]"), ""),
    (re.compile(r"(?m)^[ \t]+|[ \t]+$"), ""),
    # Lines left holding only hashtag/mention symbols or bare punctuation.
    (re.compile(r"(?m)^[#@][#@ \t]*$\n?"), ""),
    (re.compile(r"(?m)^[-—–:,.!?][-—–:,.!? \t]*$\n?"), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
)

# Bracketed all-caps tokens: our own former mask, and anything shaped like it.
# A draft containing one is echoing scaffolding, not writing a post.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_ /-]{2,}\]|<[A-Z][A-Z0-9_ /-]{2,}>")
# Section labels and block separators from the prompt itself.
_SCAFFOLD_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("examples_header", re.compile(r"(?im)^\s*EXAMPLES\b.*:")),
    ("brief_header", re.compile(r"(?im)^\s*BRIEF\s*:")),
    ("write_instruction", re.compile(r"(?i)write the post now")),
    ("block_separator", re.compile(r"(?m)^\s*(?:-{3,}|\*{3,}|={3,}|—{2,})\s*$")),
)

# Sentence-initial words are syntactically capitalized, not necessarily names.
# Keep this deliberately small and generic; unknown capitalized words are safer
# to mask than to leak from a retrieved exemplar.
_NON_ENTITY_CAPS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "here",
        "how",
        "i",
        "if",
        "in",
        "it",
        "its",
        "no",
        "not",
        "of",
        "on",
        "one",
        "or",
        "our",
        "so",
        "stop",
        "that",
        "the",
        "their",
        "these",
        "they",
        "this",
        "those",
        "to",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "we",
        "what",
        "when",
        "where",
        "why",
        "with",
        "you",
        "your",
    }
)

# Ordinary English words that are capitalized only by sentence position, list
# formatting, or emphasis. Never product or company names.
_COMMON_WORDS = frozenset(
    """
    about above across after again against all almost alone along already also although
    always am among another any anyone anything are around away back bad be became because
    become been before began begin behind being below best better between beyond big both
    bring build building built business businesses buy call called came can cannot care
    carry case cases catch change changed changes cheap check choice choose chose clear
    close code come comes coming common company companies compare consider cost costs could
    country course cover create created culture curious current customer customers cut day
    days deal decide decision decisions deep deliver demand did different difficult direct
    do does doing done down draft draw drive drop each earlier early easier easy either
    else end enough enter entire even ever every everyone everything exactly example
    examples except expect experience explain fact facts fail failed failure fair far fast
    feel few field find fine first fix focus follow followed following for force found four
    free friend from front full fund funding further future gave general get give given go
    goal goals goes going gone good got great group grow growth had half hand happen happened
    happens hard has have having head hear heard help her hers high him his hit hold home
    honest hope hour hours house however huge human hundred idea ideas identify important
    improve include included including increase indeed inside instead interest into invest
    is issue issues job jobs join just keep kept key kind know known lack land large last
    late later lead leader leaders leadership learn learned least leave led left less lesson
    lessons let letter level life light like likely line lines list listen little live local
    long look looked looking lose loss lost lot love low made mail main major make makes
    making manage management manager managers many market markets match matter maybe mean
    means meant measure meet meeting member members mention message method middle might mind
    minute minutes miss mission model models moment money month months more most move moved
    movement much must my myself name named names near need needed needs never new news next
    nice night nobody none nor north note nothing notice now number numbers offer office
    often okay old once only open operation operations opportunity option options order
    other others out outcome output over own owner owners page paid part partner partners
    parts pattern pay people per perhaps person picked place plan plans platform play please
    point points poor position possible post posts power practice prefer prepare present
    press pretty price prices pricing print probably problem problems process product
    products program progress project projects promise prove provide public pull purpose
    push put quality question questions quick quickly quiet quite raise ran rate rather read
    ready real reality really reason reasons receive recent record reduce release remember
    remove repeat replace report reports request require research resource resources
    response responsible rest result results return review right rise risk risks role roles
    room round rule rules run running safe said sale sales same saw say saying says scale
    school science score search second section see seem seen sell send sense sent series
    serious serve service services set several share shift ship shipped shipping short
    should show shown side sign signal signals similar simple simply since single site
    situation size skill skills slow small smart social solution solutions solve some
    someone something sometimes soon sorry sort sound source sources speak special specific
    speed spend spent stage stand standard start started state statement stay step steps
    still stop stopped story straight strategy strong structure study stuff style subject
    success such suddenly suggest summer support suppose sure surface system systems take
    taken takes taking talk talked target task tasks teach team teams tech technology tell
    ten term terms test tests than thank thanks then there therefore thing things think
    third though thought thousand three through throw thus time times tiny today together
    told tomorrow tone too took tool tools top total touch toward track trade train
    training transfer treat tree trend trouble true trust truth try trying turn turned type
    under understand unit until up update upon us use used useful user users using usually
    value values various very view voice wait walk want wanted war watch water way ways
    week weeks welcome well went were west what whatever whatsoever whether which while
    white who whole whom whose wide will win window wish within without won word words work
    worked worker workers working works world worry worse worst worth would write writing
    written wrong wrote year years yes yesterday yet young
    """.split()
)

# Calendar words: capitalized by convention, never a fabricated organization.
_CALENDAR_WORDS = frozenset(
    """
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october november december
    q1 q2 q3 q4
    """.split()
)

# Generic business/tech acronyms. Deliberately excludes real vendor acronyms
# (AWS, GCP, IBM) so fabricating one of those is still caught.
_COMMON_ACRONYMS = frozenset(
    """
    ai api apis arr b2b b2c ceo cfo coo cto crm csv cv eu faq gdp hr ip it kpi kpis llm
    llms mba ml mrr mvp nps ok okr okrs pdf pm pr qa r&d rag roi rfp saas sdk seo sla smb
    sms sql tam tl;dr tldr ui uk url us usa ux vc vp
    """.split()
)

# Words this project's own prompt scaffolding puts in front of the model. If a
# draft echoes one it is a prompt-following failure, not a fabricated entity —
# and the parrot/format checks are the right place to notice that.
_PROMPT_SCAFFOLD = frozenset(
    """
    assistant brief briefs cadence constraint constraints entity examples filler lineation
    linkedin paragraph paragraphs response rhythm system topic user
    """.split()
)

_COMMON_CAPITALIZED = (
    _NON_ENTITY_CAPS | _COMMON_WORDS | _CALENDAR_WORDS | _COMMON_ACRONYMS | _PROMPT_SCAFFOLD
)

# Public alias: the de-voicing operator needs the same "capitalized but not a
# name" vocabulary to decide which shouted words are emphasis it may lowercase
# and which are acronyms it must leave alone.
COMMON_CAPITALIZED = _COMMON_CAPITALIZED


@dataclass(frozen=True)
class InventionResult:
    """Entities and evidence figures a draft added beyond its brief."""

    invented_entities: frozenset[str]
    invented_numbers: frozenset[str]

    @property
    def passed(self) -> bool:
        return not self.invented_entities and not self.invented_numbers


def _entity_key(span: str) -> str:
    normalized = re.sub(r"\s+", " ", span.strip()).casefold()
    return _POSSESSIVE_RE.sub("", normalized)


def _looks_like_name(key: str, *, allowlist: frozenset[str]) -> bool:
    """True when no token in the span is an ordinary capitalized word.

    Multi-word spans keep their signal from the unfamiliar token: "Northwind
    Traders" survives on ``northwind`` even though ``traders`` is ordinary,
    while "Keep The Test" is dropped because every token is ordinary.
    """
    tokens = [token for token in key.split(" ") if token]
    if not tokens:
        return False
    return any(token not in allowlist for token in tokens)


def _entity_spans(
    text: str,
    *,
    allowlist: frozenset[str] = _NON_ENTITY_CAPS,
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _ENTITY_RE.finditer(text or ""):
        key = _entity_key(match.group(0))
        if not _looks_like_name(key, allowlist=allowlist):
            continue
        spans.append((match.start(), match.end(), key))
    return spans


def extract_entity_keys(text: str) -> set[str]:
    """Case-insensitive proper-name proxies, masking-strict.

    Treats any unfamiliar capitalized word as a name. Correct for deciding what
    to hide in an exemplar; far too eager for accusing a draft of invention.

    Multi-word spans also yield their individual tokens, so a brief naming
    "Contoso Ledger" grants a draft that says only "Ledger".
    """

    keys: set[str] = set()
    for _, _, key in _entity_spans(text):
        keys.add(key)
        keys.update(token for token in key.split(" ") if token)
    return keys


def extract_named_entity_keys(text: str) -> set[str]:
    """Case-insensitive proper-name proxies, invention-strict.

    Drops ordinary vocabulary, calendar words, generic acronyms, and this
    project's own prompt scaffolding, so what remains is plausibly a real
    proper name. Multi-word spans also shed their ordinary tokens: a
    sentence-initial ``Keep Contoso`` collapses to ``contoso`` rather than
    inventing a phantom ``keep contoso`` company.
    """

    keys: set[str] = set()
    for _, _, key in _entity_spans(text, allowlist=_COMMON_CAPITALIZED):
        name_tokens = [token for token in key.split(" ") if token not in _COMMON_CAPITALIZED]
        if not name_tokens:
            continue
        cleaned = " ".join(name_tokens)
        keys.add(cleaned)
        keys.update(name_tokens)
    return keys


def mask_exemplar_entities(
    exemplar: str,
    brief: str,
    *,
    mask: str = DEFAULT_ENTITY_MASK,
) -> str:
    """Redact exemplar proper names that are not explicitly present in the brief.

    The default ``mask`` is empty: the name is cut out and the surrounding
    whitespace and punctuation tidied, so the exemplar keeps its rhythm and
    lineation without advertising a slot to fill. Passing a visible token is
    still possible for diagnostics, but never for text handed to a model —
    ``[ENTITY]`` in a prompt reliably comes back out in the draft.
    """

    if not exemplar:
        return ""
    allowed = extract_entity_keys(brief)
    pieces: list[str] = []
    cursor = 0
    for start, end, key in _entity_spans(exemplar):
        if key in allowed or start < cursor:
            continue
        pieces.append(exemplar[cursor:start])
        pieces.append(mask)
        cursor = end
        possessive = _TRAILING_POSSESSIVE_RE.match(exemplar[cursor:])
        if possessive:
            cursor += possessive.end()
    pieces.append(exemplar[cursor:])
    redacted = "".join(pieces)
    if mask:
        return redacted
    for pattern, replacement in _TIDY_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted.strip()


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD_RE.finditer(text or "")]


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1")
    tokens = _word_tokens(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _comparison_texts(exemplars: Iterable[str]) -> list[str]:
    """Each exemplar plus its fully redacted form.

    The model is shown redacted exemplars, so a copied chunk matches the
    redacted text, not the original: comparing against the original alone lets
    any window containing a removed name slip through.
    """

    texts: list[str] = []
    seen: set[str] = set()
    for exemplar in exemplars:
        redacted = mask_exemplar_entities(exemplar, "")
        for text in (exemplar, redacted):
            if text and text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def find_parroted_ngrams(
    draft: str,
    exemplars: Iterable[str],
    *,
    n: int = DEFAULT_PARROT_NGRAM,
) -> set[str]:
    """Return normalized draft n-grams copied from any exemplar."""

    draft_ngrams = _ngrams(draft, n)
    if not draft_ngrams:
        return set()
    exemplar_ngrams: set[tuple[str, ...]] = set()
    for exemplar in _comparison_texts(exemplars):
        exemplar_ngrams.update(_ngrams(exemplar, n))
    return {" ".join(ngram) for ngram in draft_ngrams & exemplar_ngrams}


def find_scaffold_markers(draft: str) -> set[str]:
    """Prompt scaffolding a finished post never contains.

    Bracketed placeholders, section headers, the write instruction, and block
    separators all mean the draft is reproducing its own prompt.
    """

    text = draft or ""
    markers = {name for name, pattern in _SCAFFOLD_RES if pattern.search(text)}
    if _PLACEHOLDER_RE.search(text):
        markers.add("placeholder")
    return markers


def copied_token_ratio(
    draft: str,
    exemplars: Iterable[str],
    *,
    n: int = PARROT_COVERAGE_NGRAM,
) -> float:
    """Share of draft words sitting inside an exemplar n-gram.

    An exact-match check on long windows misses a dump that drifts every few
    lines; coverage does not. Near 1.0 means the draft *is* the exemplars.
    """

    tokens = _word_tokens(draft)
    if len(tokens) < n:
        return 0.0
    exemplar_ngrams: set[tuple[str, ...]] = set()
    for exemplar in _comparison_texts(exemplars):
        exemplar_ngrams.update(_ngrams(exemplar, n))
    if not exemplar_ngrams:
        return 0.0
    covered: set[int] = set()
    for index in range(len(tokens) - n + 1):
        if tuple(tokens[index : index + n]) in exemplar_ngrams:
            covered.update(range(index, index + n))
    return round(len(covered) / len(tokens), 4)


def parrot_reject(
    draft: str,
    exemplars: Iterable[str],
    *,
    n: int = DEFAULT_PARROT_NGRAM,
    coverage_limit: float = PARROT_COVERAGE_LIMIT,
    coverage_min_tokens: int = PARROT_COVERAGE_MIN_TOKENS,
) -> bool:
    """Reject a draft that copies its exemplars or echoes its own prompt.

    Three independent signals: prompt scaffolding in the output, one exact
    ``n``-token window shared with an exemplar, or — for drafts long enough to
    measure — more than ``coverage_limit`` of the draft sitting inside exemplar
    5-grams. The last one is what catches an exemplar dump, which reproduces
    many separate passages rather than one long contiguous run.
    """

    if find_scaffold_markers(draft):
        return True
    pool = list(exemplars)
    if find_parroted_ngrams(draft, pool, n=n):
        return True
    if len(_word_tokens(draft)) < coverage_min_tokens:
        return False
    return copied_token_ratio(draft, pool) >= coverage_limit


def brief_echo_reject(
    draft: str,
    brief: str,
    *,
    limit: float = BRIEF_ECHO_LIMIT,
    min_tokens: int = BRIEF_ECHO_MIN_TOKENS,
) -> bool:
    """Reject a draft that is little more than the brief handed back.

    Deliberately looser than :func:`parrot_reject`: a post is *supposed* to use
    the brief's facts and often its phrasing, so only near-total coverage counts.
    A holdout draft that listed the mined bullets verbatim scored the best axis
    distance of its arm while writing nothing at all.
    """

    if len(_word_tokens(draft)) < min_tokens:
        return False
    return copied_token_ratio(draft, [brief]) >= limit


def check_invention(brief: str, draft: str) -> InventionResult:
    """Check draft entities and evidence figures against the brief only.

    The draft side uses the invention-strict extractor and the brief side the
    masking-strict one. The asymmetry is deliberate: a brief should grant as
    much as possible, a draft should be accused of as little as possible, so
    the flags that survive point at genuine fabrication.
    """

    invented_entities = extract_named_entity_keys(draft) - extract_entity_keys(brief)
    invented_numbers = extract_evidence_number_keys(draft) - extract_evidence_number_keys(brief)
    return InventionResult(
        invented_entities=frozenset(invented_entities),
        invented_numbers=frozenset(invented_numbers),
    )


def scrub_invented_sentences(
    draft: str,
    brief: str,
    *,
    normalize: Callable[[str], str] | None = None,
    max_passes: int = 3,
) -> str:
    """Cut the sentences carrying names or figures the brief never granted.

    Last resort after a repair regenerate has already failed. The sentence, not
    the span, is the unit of removal: cutting only the fabricated name out of
    "Fabrikam shipped the migration in nine weeks" leaves the fabricated claim
    standing with its subject missing, which still reads as prose and is still
    false. Dropping the sentence removes the claim.

    Lineation survives — sentences are dropped inside their own line — so a
    scrubbed section keeps the paragraph rhythm the rest of the pipeline
    measures. Returns ``""`` when nothing survives, which is the caller's
    signal to drop the section.

    ``normalize`` is the draft-side case normalizer the caller's invention
    guard applies before checking (sentence-initial verbs are capitalized by
    syntax, not because they are names). It is injected rather than imported
    because it lives above this module in the import order.
    """

    text = draft or ""
    prepare = normalize or (lambda value: value)
    for _ in range(max(1, int(max_passes))):
        if not text.strip() or check_invention(brief, prepare(text)).passed:
            break
        lines: list[str] = []
        dropped = 0
        for line in text.splitlines():
            if not line.strip():
                lines.append("")
                continue
            sentences = [part for part in _SENTENCE_END_RE.split(line) if part.strip()]
            kept = [
                part
                for part in sentences
                if check_invention(brief, prepare(part)).passed
            ]
            dropped += len(sentences) - len(kept)
            lines.append(" ".join(kept).strip())
        if not dropped:
            # The whole draft invents something no single sentence owns; more
            # passes would only repeat this one.
            break
        text = "\n".join(lines)
    for pattern, replacement in _TIDY_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


def brief_allowed_facts(brief: str) -> dict[str, list[str]]:
    """Entities and figures the draft may reuse from the visible brief.

    Returned as sorted lists for Contoso-safe prompts and receipts — never as
    free prose from the author corpus.
    """
    text = brief or ""
    return {
        "entities": sorted(extract_entity_keys(text)),
        "numbers": sorted(extract_evidence_number_keys(text)),
    }
