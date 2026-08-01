"""Article channel: outline → section drafts → stitch.

Posts stay on :mod:`personality_protect.write`. Longform needs section budgets
and article-only retrieval — raising ``max_tokens`` on the post path is not
enough and produces one long stub instead of a structured piece.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from personality_protect.chat_prompt import flatten_chat_messages
from personality_protect.config import DEFAULT_MLX_MODEL, ProfilePaths, load_config
from personality_protect.corpus_text import normalize_corpus_text
from personality_protect.draft_trim import (
    drop_repeated_paragraphs,
    drop_restated_sections,
    trim_draft,
    word_count,
)
from personality_protect.models import load_index
from personality_protect.prompt_write import build_write_messages
from personality_protect.style_profile import (
    article_section_count_hint,
    article_section_words,
    article_word_aim,
    article_word_target,
    load_style_profile,
    style_directives,
)
from personality_protect.voice_index import VECTORS_FILENAME, retrieve
from personality_protect.write import (
    DEFAULT_WRITE_K,
    MAX_EXEMPLAR_WORDS,
    GenerateFn,
    PromptSink,
    build_brief,
    clip_exemplar,
    mlx_generate_no_adapter,
    normalize_sentence_case,
)
from personality_protect.writer_guards import (
    brief_allowed_facts,
    check_invention,
    mask_exemplar_entities,
    parrot_reject,
    scrub_invented_sentences,
)

ARTICLE_SOURCES: tuple[str, ...] = ("linkedin_article",)
# Below this the article channel has too little rhythm signal to claim voice.
MIN_ARTICLE_CORPUS = 5
DEFAULT_ARTICLE_SECTION_MAX_TOKENS = 768
MAX_ARTICLE_SECTIONS = 8
MIN_ARTICLE_SECTIONS = 2
# Headroom over the section budget before the tail trim bites. A section that
# lands slightly long is finished prose; one at double the budget is the model
# recycling itself, which is what the trim exists to cut.
SECTION_TRIM_HEADROOM = 1.35
# Tokens per target word. Roughly 1.35 tokens/word for English plus room to
# finish the closing sentence before the budget runs out.
SECTION_TOKENS_PER_WORD = 2.0
# Same clip budget as the post channel for parrot reasons: at 120 words the
# holdout shared 150+ exact 8-token windows with its exemplars. The article
# path still has to spend that budget on prose, not on LinkedIn title /
# "Created on" / "Published on" chrome — otherwise the model never sees the
# author's longform rhythm inside the 60-word window.
ARTICLE_EXEMPLAR_WORDS = MAX_EXEMPLAR_WORDS
_ARTICLE_META_LINE_RE = re.compile(
    r"^(created on|published on|edited on)\b",
    re.IGNORECASE,
)
_ARTICLE_DATE_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def prepare_article_exemplar(text: str, *, max_words: int = ARTICLE_EXEMPLAR_WORDS) -> str:
    """Normalize, drop export chrome, then clip — so the budget is real prose."""
    cleaned = normalize_corpus_text(text)
    lines: list[str] = []
    seen_title = ""
    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _ARTICLE_META_LINE_RE.match(line) or _ARTICLE_DATE_LINE_RE.match(line):
            continue
        if not seen_title:
            seen_title = line.casefold()
            lines.append(line)
            continue
        if line.casefold() == seen_title:
            continue
        lines.append(line)
    body = "\n".join(lines).strip()
    return clip_exemplar(body, max_words=max_words)
# When the visible brief is thin, demanding a full article-length section is
# what forces invention. Cap the per-section aim from brief richness instead.
THIN_BRIEF_WORDS = 80
THIN_SECTION_WORD_FLOOR = 80
THIN_WORDS_PER_BRIEF_WORD = 2.5
# Floor for the repair pass: the budget is already halved, and a section that
# runs out of tokens mid-sentence is a new failure, not a repaired one.
MIN_REPAIR_TOKENS = 256

_BULLET_RE = re.compile(r"^\s*[-*•]\s+")


def count_article_pieces(paths: ProfilePaths) -> int:
    """How many article-source pieces are in the local corpus index."""
    return sum(1 for piece in load_index(paths.index_path) if piece.source in ARTICLE_SOURCES)


def count_indexed_article_pieces(paths: ProfilePaths) -> int | None:
    """Article exemplars retrieval can actually reach, or None with no index."""
    vectors_path = paths.root / "voice_index" / VECTORS_FILENAME
    if not vectors_path.is_file():
        return None
    total = 0
    with vectors_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            piece = json.loads(line).get("piece") or {}
            if str(piece.get("source") or "") in ARTICLE_SOURCES:
                total += 1
    return total


def assert_article_corpus(paths: ProfilePaths, *, minimum: int = MIN_ARTICLE_CORPUS) -> int:
    """Raise when the article channel has too little source material.

    The corpus floor and the retrieval floor are checked separately. The floor
    exists so drafts have rhythm to match, and rhythm arrives through
    retrieval — a corpus of fifty articles with two of them indexed gives the
    article channel nothing.
    """
    n = count_article_pieces(paths)
    if n < minimum:
        raise FileNotFoundError(
            f"Article channel needs at least {minimum} linkedin_article pieces "
            f"in the corpus (found {n}). Ingest more articles, then rebuild "
            "index-voice."
        )
    indexed = count_indexed_article_pieces(paths)
    if indexed is not None and indexed < minimum:
        raise FileNotFoundError(
            f"Article channel needs at least {minimum} linkedin_article pieces "
            f"in the voice index (found {indexed} of {n} in the corpus). "
            "Shrink the article holdout carve, then rebuild index-voice."
        )
    return n


def article_draft_ceiling(style: dict[str, Any], *, sections: int) -> int:
    """Word ceiling a stitched article of ``sections`` can actually reach.

    The single-shot comparison arm has to be trimmed to the same number. Two
    arms edited to different lengths would be separated by the length penalty
    alone, whatever either of them wrote.
    """
    count = max(1, int(sections))
    per_section = article_section_words(style, sections=count)
    return int(round(per_section * SECTION_TRIM_HEADROOM)) * count


def outline_from_brief(topic: str, points: str) -> list[str]:
    """Deterministic section titles from the brief (no model call).

    Each bullet becomes a section. Thin briefs get the topic as a lead section
    plus the bullets so the stitch still has structure.
    """
    topic = topic.strip()
    lines = [line.strip() for line in (points or "").splitlines() if line.strip()]
    bullets: list[str] = []
    for line in lines:
        cleaned = _BULLET_RE.sub("", line).strip()
        if cleaned:
            bullets.append(cleaned)
    if not bullets and points.strip():
        bullets = [points.strip()]

    sections: list[str] = []
    if topic and (not bullets or len(bullets) < MIN_ARTICLE_SECTIONS):
        sections.append(topic)
    for bullet in bullets:
        if bullet not in sections:
            sections.append(bullet)
        if len(sections) >= MAX_ARTICLE_SECTIONS:
            break
    if len(sections) < MIN_ARTICLE_SECTIONS:
        raise ValueError(
            "article brief needs at least two section points "
            "(topic + bullets, or two bullets)"
        )
    return sections


def _section_brief(topic: str, section: str, points: str = "") -> tuple[str, str]:
    """Topic/points pair for one section generation call.

    Only this section's claim goes in the points field. Handing every section
    the full brief is what made section 1 dump the article and sections 2–N
    rewrite it; invent checking still uses the full brief separately.
    ``points`` is accepted for call-site compatibility and ignored.
    """
    del points
    return (
        f"{topic} — {section}" if topic and topic != section else section,
        f"- {section}",
    )


def scale_section_words_for_brief(
    section_words: int,
    *,
    brief_words: int,
    sections: int,
) -> int:
    """Lower the section aim when the brief cannot support a long expand.

    A 60-word brief asking for 280 words per section is the invent pressure the
    holdout measured. The scaled aim still leaves room to write, but stops
    treating length as a hard quota over facts.
    """
    base = max(1, int(section_words))
    n_sections = max(1, int(sections))
    brief_n = max(0, int(brief_words))
    if brief_n >= THIN_BRIEF_WORDS:
        return base
    # Total article aim ≈ brief_words * factor, split across sections.
    thin_total = max(
        THIN_SECTION_WORD_FLOOR * n_sections,
        int(round(brief_n * THIN_WORDS_PER_BRIEF_WORD * n_sections)),
    )
    thin_per = max(THIN_SECTION_WORD_FLOOR, thin_total // n_sections)
    return min(base, thin_per)


def section_structure_directives(
    *,
    section: str,
    index: int,
    total: int,
    word_aim: int,
    section_words: int,
    section_trim_words: int,
    allowed_entities: Sequence[str] = (),
    allowed_numbers: Sequence[str] = (),
    other_sections: Sequence[str] = (),
) -> list[str]:
    """Where this section sits, how long it may run, and which facts are allowed.

    Structure, not voice. Kept separate from the cadence card so the eval's
    control arm can be asked for an article of the same shape without also
    being handed the measured style profile — otherwise the comparison would
    only be establishing that asking for an article produces one.
    """
    lines = [
        f"This is section {index} of {total} in a longform article of about "
        f"{word_aim} words; the other sections cover the rest of the brief.",
        f"Write only the section about: {section}",
        "Do not restate or rewrite other sections. Advance only this section.",
        f"Aim for about {section_words} words in this section, and no more "
        f"than {section_trim_words}. Prefer stopping early over inventing "
        "facts to fill the count.",
    ]
    others = [str(item).strip() for item in other_sections if str(item).strip()]
    if others:
        lines.append(
            "Other sections (do not cover them here): " + " | ".join(others) + "."
        )
    entities = [str(item).strip() for item in allowed_entities if str(item).strip()]
    numbers = [str(item).strip() for item in allowed_numbers if str(item).strip()]
    if entities:
        lines.append(
            "ALLOWED names from the BRIEF only: " + ", ".join(entities) + "."
        )
    else:
        lines.append(
            "The BRIEF names no companies or people. Invent none."
        )
    if numbers:
        lines.append(
            "ALLOWED figures from the BRIEF only: " + ", ".join(numbers) + "."
        )
    else:
        lines.append("The BRIEF has no figures. Write no numbers.")
    return lines


def section_repair_directives(
    *,
    invented_entities: Sequence[str] = (),
    invented_numbers: Sequence[str] = (),
    section_words: int,
) -> list[str]:
    """Name the facts a failed section has to lose before it is written again.

    The regenerate used to differ from the first attempt only by a shorter
    budget, which asks for less invention without saying what was invented; the
    holdout answered with the same fabricated names inside a shorter section.

    Listing the offenders is a calculated risk. A token put in front of this
    model tends to come back out of it — the reason exemplar masking redacts
    names instead of labelling them — so the directive is phrased as removal
    and :func:`~personality_protect.writer_guards.scrub_invented_sentences`
    stands behind it for the case where the model repeats what it was told to
    drop.
    """
    entities = [str(item).strip() for item in invented_entities if str(item).strip()]
    numbers = [str(item).strip() for item in invented_numbers if str(item).strip()]
    lines = [
        "REPAIR: your previous draft of this section stated facts the BRIEF "
        "never gave. Write the section again, from the BRIEF only.",
    ]
    if entities:
        lines.append(
            "Remove these names completely — do not mention them, rename them, "
            "or replace them with other names: " + ", ".join(entities) + "."
        )
    if numbers:
        lines.append(
            "Remove these figures completely and put no figures in their "
            "place: " + ", ".join(numbers) + "."
        )
    lines.append(
        "Add no companies, people, products, places, or figures of your own. "
        "Make the claim smaller instead of sourcing it."
    )
    lines.append(f"Write a shorter section this time: about {section_words} words.")
    return lines


def build_section_messages(
    extra_directives: Sequence[str] = (),
    *,
    topic: str,
    points: str,
    examples: Sequence[str],
    directives: Sequence[str],
) -> list[dict[str, str]]:
    """Article-section prompt with room for an extra directive block.

    Both arms build their sections through here, so a repair directive reaches
    the control arm in the same position it reaches the product arm.
    """
    return build_write_messages(
        topic=topic,
        points=points,
        examples=examples,
        channel="article",
        style_directives=[*directives, *extra_directives],
    )


def _guard_flags(brief: str, draft: str, exemplars: Sequence[str]) -> dict[str, Any]:
    invention = check_invention(brief, normalize_sentence_case(draft))
    return {
        "parrot_reject": parrot_reject(draft, list(exemplars)),
        "invent_reject": not invention.passed,
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
    }


def draft_section_with_repair(
    *,
    build_messages: Callable[[Sequence[str]], list[dict[str, str]]],
    generate_fn: GenerateFn,
    base_model: str,
    invent_brief: str,
    section_words: int,
    section_trim_words: int,
    max_tokens: int,
    exemplars: Sequence[str] = (),
    prompt_sink: PromptSink | None = None,
) -> dict[str, Any]:
    """Draft one section, then repair it, then scrub it, and only then drop it.

    ``status`` says which of those the section survived: ``clean`` (the first
    draft passed the guards), ``repaired`` (the regenerate with the offenders
    named passed), ``scrubbed`` (it passed only after the inventing sentences
    were cut out), or ``dropped`` (nothing usable survived).

    Dropping is last rather than second because it is the expensive outcome. An
    inventing section used to be dropped outright, so a thin brief could take
    every section out and leave an empty article that still counted as a draft
    — the holdout produced exactly that, a zero-word arm scored against a base
    arm that had written 135 words.
    """
    repair_words = max(THIN_SECTION_WORD_FLOOR, int(section_words) // 2)
    draft = ""
    guards: dict[str, Any] = {
        "parrot_reject": False,
        "invent_reject": False,
        "invented_entities": [],
        "invented_numbers": [],
    }
    attempts = 0
    first_messages: list[dict[str, str]] = []
    for attempt in (1, 2):
        if attempt == 1:
            extra: Sequence[str] = ()
            trim_words = int(section_trim_words)
            attempt_tokens = int(max_tokens)
        else:
            # Second pass: half the budget, plus the offender list when there
            # is one. A parrot failure has nothing to list, so it keeps the
            # shorter-budget retry on its own.
            trim_words = repair_words
            attempt_tokens = max(
                MIN_REPAIR_TOKENS, int(round(trim_words * SECTION_TOKENS_PER_WORD))
            )
            extra = (
                section_repair_directives(
                    invented_entities=guards["invented_entities"],
                    invented_numbers=guards["invented_numbers"],
                    section_words=trim_words,
                )
                if guards["invent_reject"]
                else ()
            )
        messages = build_messages(extra)
        if attempt == 1:
            first_messages = messages
        attempts += 1
        raw = str(
            generate_fn(
                messages,
                base_model=base_model,
                max_tokens=attempt_tokens,
                prompt_sink=prompt_sink,
            )
        ).strip()
        draft = trim_draft(raw, max_words=trim_words)
        guards = _guard_flags(invent_brief, draft, exemplars)
        clean = not guards["parrot_reject"] and not guards["invent_reject"]
        if clean and draft.strip():
            return {
                "draft": draft,
                "status": "clean" if attempt == 1 else "repaired",
                "attempts": attempts,
                "guards": guards,
                "messages": first_messages,
            }

    if guards["invent_reject"]:
        scrubbed = scrub_invented_sentences(
            draft, invent_brief, normalize=normalize_sentence_case
        )
        if scrubbed.strip():
            scrub_guards = _guard_flags(invent_brief, scrubbed, exemplars)
            if not scrub_guards["parrot_reject"] and not scrub_guards["invent_reject"]:
                return {
                    "draft": scrubbed,
                    "status": "scrubbed",
                    "attempts": attempts,
                    "guards": scrub_guards,
                    "messages": first_messages,
                }
    return {
        "draft": "",
        "status": "dropped",
        "attempts": attempts,
        "guards": guards,
        "messages": first_messages,
    }


def run_write_article(
    topic: str,
    points: str,
    paths: ProfilePaths,
    *,
    k: int = DEFAULT_WRITE_K,
    max_tokens: int = DEFAULT_ARTICLE_SECTION_MAX_TOKENS,
    generate_fn: GenerateFn | None = None,
    prompt_sink: PromptSink | None = None,
    min_articles: int = MIN_ARTICLE_CORPUS,
) -> dict[str, Any]:
    """Outline → per-section RAG draft → repair/scrub → stitch into one article."""
    topic = topic.strip()
    points = points.strip()
    if not topic:
        raise ValueError("topic must not be empty")
    if not points:
        raise ValueError("points must not be empty")

    article_count = assert_article_corpus(paths, minimum=min_articles)
    sections = outline_from_brief(topic, points)
    config = load_config(paths)
    style = load_style_profile(paths)
    directives = style_directives(style, channel="article")
    # Length comes from the author's own articles, split across the outline —
    # not from the post band, whose ceiling is a LinkedIn character limit.
    word_aim = article_word_aim(style)
    word_ceiling = article_word_target(style)
    full_brief = build_brief(topic, points)
    brief_words = word_count(full_brief)
    section_words = scale_section_words_for_brief(
        article_section_words(style, sections=len(sections)),
        brief_words=brief_words,
        sections=len(sections),
    )
    # Recompute the stitched aim so receipts and the eval ceiling match what
    # sections were actually asked to write.
    word_aim = max(word_aim, section_words * len(sections))
    if brief_words < THIN_BRIEF_WORDS:
        word_aim = section_words * len(sections)
    section_trim_words = int(round(section_words * SECTION_TRIM_HEADROOM))
    section_max_tokens = max(
        int(max_tokens), int(round(section_words * SECTION_TOKENS_PER_WORD))
    )
    allowed = brief_allowed_facts(full_brief)
    generator = generate_fn or mlx_generate_no_adapter
    model_id = config.base_model or DEFAULT_MLX_MODEL

    matches = retrieve(
        full_brief,
        k=k,
        profile=paths.name,
        home=paths.home,
        sources=ARTICLE_SOURCES,
    )
    if k and not matches:
        raise FileNotFoundError(
            f"No article exemplars indexed for profile {paths.name}. "
            "Ingest linkedin_article pieces and run: personality-protect index-voice"
        )
    exemplars = [str(match["text"]) for match in matches]
    masked = [
        mask_exemplar_entities(
            prepare_article_exemplar(exemplar, max_words=ARTICLE_EXEMPLAR_WORDS),
            full_brief,
        )
        for exemplar in exemplars
    ]

    section_drafts: list[str] = []
    dropped_sections: list[str] = []
    repaired_sections: list[str] = []
    scrubbed_sections: list[str] = []
    all_messages: list[list[dict[str, str]]] = []
    attempts_total = 0
    dropped_invented_entities: set[str] = set()
    dropped_invented_numbers: set[str] = set()

    for index, section in enumerate(sections, start=1):
        section_topic, section_points = _section_brief(topic, section, points)
        others = [title for title in sections if title != section]
        outcome = draft_section_with_repair(
            build_messages=partial(
                build_section_messages,
                topic=section_topic,
                points=section_points,
                examples=masked,
                directives=[
                    *directives,
                    *section_structure_directives(
                        section=section,
                        index=index,
                        total=len(sections),
                        word_aim=word_aim,
                        section_words=section_words,
                        section_trim_words=section_trim_words,
                        allowed_entities=allowed["entities"],
                        allowed_numbers=allowed["numbers"],
                        other_sections=others,
                    ),
                ],
            ),
            generate_fn=generator,
            base_model=model_id,
            invent_brief=full_brief,
            section_words=section_words,
            section_trim_words=section_trim_words,
            max_tokens=section_max_tokens,
            exemplars=exemplars,
            prompt_sink=prompt_sink,
        )
        all_messages.append(outcome["messages"])
        attempts_total += int(outcome["attempts"])
        if outcome["status"] == "dropped":
            guards = outcome["guards"]
            dropped_sections.append(section)
            dropped_invented_entities.update(guards.get("invented_entities") or [])
            dropped_invented_numbers.update(guards.get("invented_numbers") or [])
            continue
        section_drafts.append(str(outcome["draft"]))
        if outcome["status"] == "repaired":
            repaired_sections.append(section)
        elif outcome["status"] == "scrubbed":
            scrubbed_sections.append(section)

    # Sections are generated independently, so two of them can arrive as the
    # same argument in different words. Drop restated section drafts first,
    # then collapse near-duplicate paragraphs inside what remains.
    text = drop_repeated_paragraphs(
        "\n\n".join(drop_restated_sections(section_drafts))
    ).strip()
    # Final invent check against the full brief the author supplied. The flag
    # describes the text that ships: a section dropped for inventing is a gap
    # in coverage, not a fabrication in the draft. Forcing invent_reject on any
    # drop disqualified whole articles whose remaining text was clean — and on
    # the holdout that was three of four items. What still fails the article is
    # an empty stitch, or a stitch that invents against the visible brief.
    final_guards = _guard_flags(full_brief, text, exemplars)
    if not text:
        final_guards["invent_reject"] = True

    return {
        "text": text,
        "channel": "article",
        "voice_mode": config.voice_mode,
        "adapter": "none",
        "write_adapter": None,
        "model": model_id,
        "k": len(matches),
        "exemplar_ids": [str(match["id"]) for match in matches],
        "attempts": attempts_total,
        "word_target": word_ceiling,
        "word_aim": word_aim,
        "section_words": section_words,
        "section_trim_words": section_trim_words,
        "section_count_hint": article_section_count_hint(style),
        "article_count": article_count,
        "sections": sections,
        "section_count": len(section_drafts),
        "dropped_sections": dropped_sections,
        "repaired_sections": repaired_sections,
        "scrubbed_sections": scrubbed_sections,
        "dropped_invented_entities": sorted(dropped_invented_entities),
        "dropped_invented_numbers": sorted(dropped_invented_numbers),
        "draft_words": word_count(text),
        "allowed_entities": allowed["entities"],
        "allowed_numbers": allowed["numbers"],
        **final_guards,
        "exemplar_texts": exemplars,
        "messages": all_messages[0] if all_messages else [],
        "prompt": flatten_chat_messages(all_messages[0]) if all_messages else "",
    }
