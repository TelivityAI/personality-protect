"""RAG-backed writing with injectable, adapter-free generation.

MLX is imported only inside :func:`mlx_generate_no_adapter`, behind the
``mlx_runtime`` opt-in gate. Importing this module never touches Metal.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from personality_protect.config import DEFAULT_MLX_MODEL, ProfilePaths, load_config
from personality_protect.prompt_write import build_write_prompt
from personality_protect.voice_index import retrieve
from personality_protect.writer_guards import (
    check_invention,
    mask_exemplar_entities,
    parrot_reject,
)

DEFAULT_WRITE_K = 5
MIN_WRITE_K = 3
MAX_WRITE_K = 5
DEFAULT_WRITE_MAX_TOKENS = 768
MAX_WRITE_ATTEMPTS = 2

GenerateFn = Callable[..., str]


def mlx_generate_no_adapter(
    prompt: str,
    *,
    base_model: str,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
) -> str:
    """Generate from MLX base weights with ``adapter_path=None``.

    The write path is RAG-only: a LoRA adapter is never loaded, even when one
    exists on disk. Tests inject ``generate_fn`` instead of calling this.
    """
    from personality_protect.mlx_runtime import (
        assert_mlx_import_allowed,
        ensure_mlx_wired_cap,
        release_mlx_memory,
    )

    assert_mlx_import_allowed()
    ensure_mlx_wired_cap(memory_gb=16.0)
    try:
        from mlx_lm import generate, load

        model, tokenizer = load(base_model, adapter_path=None)
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max(64, int(max_tokens)),
            verbose=False,
        )
        return str(output).strip()
    finally:
        release_mlx_memory()


_SENTENCE_START_WORD_RE = re.compile(
    r"(?:(?<=\A)|(?<=[.!?:;]\s)|(?<=[.!?:;]\n)|(?<=\n))([A-Z][a-z][a-zA-Z'’-]*)"
)
_MID_SENTENCE_CAP_RE = re.compile(r"(?<=[a-z0-9,'’\-]\s)([A-Z][a-z][a-zA-Z'’-]*)")


def normalize_sentence_case(draft: str) -> str:
    """Lowercase sentence-initial words that are never capitalized mid-sentence.

    Drafts open sentences and lines with ordinary verbs ("Keep the test boring"),
    which the entity guard would otherwise read as invented proper names. A real
    name copied into the draft still appears capitalized mid-sentence, so it
    keeps its case and stays visible to :func:`check_invention`.
    """
    text = draft or ""
    kept = {match.group(1) for match in _MID_SENTENCE_CAP_RE.finditer(text)}
    return _SENTENCE_START_WORD_RE.sub(
        lambda match: match.group(1)
        if match.group(1) in kept
        else match.group(1)[0].lower() + match.group(1)[1:],
        text,
    )


def _guard_flags(brief: str, draft: str, exemplars: list[str]) -> dict[str, Any]:
    invention = check_invention(brief, normalize_sentence_case(draft))
    return {
        "parrot_reject": parrot_reject(draft, exemplars),
        "invent_reject": not invention.passed,
        "invented_entities": sorted(invention.invented_entities),
        "invented_numbers": sorted(invention.invented_numbers),
    }


def build_brief(topic: str, points: str) -> str:
    """Brief text used for retrieval, masking, and the invention guard."""
    return f"Topic: {topic}\nPoints: {points}"


def run_write(
    topic: str,
    points: str,
    paths: ProfilePaths,
    *,
    k: int = DEFAULT_WRITE_K,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    """Retrieve exemplars, generate a draft, regenerate once on guard failure."""
    topic = topic.strip()
    points = points.strip()
    if not topic:
        raise ValueError("topic must not be empty")
    if not points:
        raise ValueError("points must not be empty")
    if not MIN_WRITE_K <= k <= MAX_WRITE_K:
        raise ValueError(f"--k must be between {MIN_WRITE_K} and {MAX_WRITE_K}")

    config = load_config(paths)
    brief = build_brief(topic, points)
    matches = retrieve(brief, k=k, profile=paths.name, home=paths.home)
    if not matches:
        raise FileNotFoundError(
            f"No voice exemplars indexed for profile {paths.name}. "
            "Run: personality-protect index-voice"
        )

    exemplars = [str(match["text"]) for match in matches]
    masked = [mask_exemplar_entities(exemplar, brief) for exemplar in exemplars]
    prompt = build_write_prompt(topic=topic, points=points, examples=masked)

    generator = generate_fn or mlx_generate_no_adapter
    model_id = config.base_model or DEFAULT_MLX_MODEL

    draft = ""
    guards: dict[str, Any] = {}
    attempts = 0
    for attempts in range(1, MAX_WRITE_ATTEMPTS + 1):
        draft = str(
            generator(prompt, base_model=model_id, max_tokens=max_tokens)
        ).strip()
        guards = _guard_flags(brief, draft, exemplars)
        if not guards["parrot_reject"] and not guards["invent_reject"]:
            break

    return {
        "text": draft,
        "voice_mode": config.voice_mode,
        "adapter": "none",
        "write_adapter": None,
        "model": model_id,
        "k": len(matches),
        "exemplar_ids": [str(match["id"]) for match in matches],
        "attempts": attempts,
        **guards,
    }
