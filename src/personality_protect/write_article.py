"""Article channel: outline → section drafts → stitch.

Posts stay on :mod:`personality_protect.write`. Longform needs section budgets
and article-only retrieval — raising ``max_tokens`` on the post path is not
enough and produces one long stub instead of a structured piece.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from personality_protect.chat_prompt import flatten_chat_messages
from personality_protect.config import DEFAULT_MLX_MODEL, ProfilePaths, load_config
from personality_protect.draft_trim import trim_draft, word_count
from personality_protect.models import load_index
from personality_protect.prompt_write import build_write_messages
from personality_protect.style_profile import (
    draft_word_target,
    load_style_profile,
    style_directives,
)
from personality_protect.voice_index import retrieve
from personality_protect.write import (
    DEFAULT_WRITE_K,
    GenerateFn,
    PromptSink,
    build_brief,
    clip_exemplar,
    mlx_generate_no_adapter,
    normalize_sentence_case,
)
from personality_protect.writer_guards import (
    check_invention,
    mask_exemplar_entities,
    parrot_reject,
)

ARTICLE_SOURCES: tuple[str, ...] = ("linkedin_article",)
# Below this the article channel has too little rhythm signal to claim voice.
MIN_ARTICLE_CORPUS = 5
DEFAULT_ARTICLE_SECTION_MAX_TOKENS = 768
DEFAULT_ARTICLE_SECTION_WORDS = 280
MAX_ARTICLE_SECTIONS = 8
MIN_ARTICLE_SECTIONS = 2

_BULLET_RE = re.compile(r"^\s*[-*•]\s+")


def count_article_pieces(paths: ProfilePaths) -> int:
    """How many article-source pieces are in the local corpus index."""
    return sum(1 for piece in load_index(paths.index_path) if piece.source in ARTICLE_SOURCES)


def assert_article_corpus(paths: ProfilePaths, *, minimum: int = MIN_ARTICLE_CORPUS) -> int:
    """Raise when the article channel has too little source material."""
    n = count_article_pieces(paths)
    if n < minimum:
        raise FileNotFoundError(
            f"Article channel needs at least {minimum} linkedin_article pieces "
            f"in the corpus (found {n}). Ingest more articles, then rebuild "
            "index-voice."
        )
    return n


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


def _section_brief(topic: str, section: str, points: str) -> tuple[str, str]:
    """Topic/points pair for one section generation call."""
    return (
        f"{topic} — {section}" if topic and topic != section else section,
        f"- {section}\n- Stay on this section only.\n- Brief points:\n{points}",
    )


def _guard_flags(brief: str, draft: str, exemplars: Sequence[str]) -> dict[str, Any]:
    invention = check_invention(brief, normalize_sentence_case(draft))
    return {
        "parrot_reject": parrot_reject(draft, list(exemplars)),
        "invent_reject": not invention.passed,
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
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
    """Outline → per-section RAG draft → stitch into one article."""
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
    directives = style_directives(style)
    # Articles are longer than posts; do not clamp sections to the post ceiling.
    section_words = max(
        DEFAULT_ARTICLE_SECTION_WORDS,
        int(draft_word_target(style)),
    )
    generator = generate_fn or mlx_generate_no_adapter
    model_id = config.base_model or DEFAULT_MLX_MODEL

    full_brief = build_brief(topic, points)
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
        mask_exemplar_entities(clip_exemplar(exemplar), full_brief) for exemplar in exemplars
    ]

    section_drafts: list[str] = []
    all_messages: list[list[dict[str, str]]] = []
    attempts_total = 0
    last_guards: dict[str, Any] = {
        "parrot_reject": False,
        "invent_reject": False,
        "invented_entities": [],
        "invented_numbers": [],
    }

    for section in sections:
        section_topic, section_points = _section_brief(topic, section, points)
        section_brief = build_brief(section_topic, section_points)
        messages = build_write_messages(
            topic=section_topic,
            points=section_points,
            examples=masked,
            style_directives=[
                *directives,
                f"Write only the section about: {section}",
                f"Aim for about {section_words} words in this section.",
            ],
        )
        all_messages.append(messages)
        draft = ""
        for attempt in range(1, 3):
            attempts_total += 1
            raw = str(
                generator(
                    messages,
                    base_model=model_id,
                    max_tokens=max_tokens,
                    prompt_sink=prompt_sink,
                )
            ).strip()
            draft = trim_draft(raw, max_words=section_words * 2)
            last_guards = _guard_flags(section_brief, draft, exemplars)
            if not last_guards["parrot_reject"] and not last_guards["invent_reject"]:
                break
        section_drafts.append(draft)

    text = "\n\n".join(part for part in section_drafts if part.strip()).strip()
    # Final invent check against the full brief the author supplied.
    final_guards = _guard_flags(full_brief, text, exemplars)

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
        "word_target": section_words * len(sections),
        "article_count": article_count,
        "sections": sections,
        "section_count": len(sections),
        "draft_words": word_count(text),
        **final_guards,
        "exemplar_texts": exemplars,
        "messages": all_messages[0] if all_messages else [],
        "prompt": flatten_chat_messages(all_messages[0]) if all_messages else "",
    }
