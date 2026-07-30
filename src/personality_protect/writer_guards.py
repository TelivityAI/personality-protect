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
from dataclasses import dataclass
from typing import Iterable

from personality_protect.eval_compare import extract_evidence_number_keys

DEFAULT_ENTITY_MASK = "[ENTITY]"
DEFAULT_PARROT_NGRAM = 8

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z](?:&[A-Z])+|"
    r"[A-Z][a-z][a-zA-Z0-9'-]*(?:\s+[A-Z][a-z][a-zA-Z0-9'-]*)*)\b"
)
_POSSESSIVE_RE = re.compile(r"['’]s$")

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
    """Mask exemplar proper names that are not explicitly present in the brief."""

    if not exemplar:
        return ""
    allowed = extract_entity_keys(brief)
    pieces: list[str] = []
    cursor = 0
    for start, end, key in _entity_spans(exemplar):
        if key in allowed:
            continue
        pieces.append(exemplar[cursor:start])
        pieces.append(mask)
        cursor = end
    pieces.append(exemplar[cursor:])
    return "".join(pieces)


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD_RE.finditer(text or "")]


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1")
    tokens = _word_tokens(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


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
    for exemplar in exemplars:
        exemplar_ngrams.update(_ngrams(exemplar, n))
    return {" ".join(ngram) for ngram in draft_ngrams & exemplar_ngrams}


def parrot_reject(
    draft: str,
    exemplars: Iterable[str],
    *,
    n: int = DEFAULT_PARROT_NGRAM,
) -> bool:
    """Reject a draft containing an exact normalized exemplar n-gram."""

    return bool(find_parroted_ngrams(draft, exemplars, n=n))


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
