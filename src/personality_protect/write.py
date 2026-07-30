"""RAG-backed writing with injectable, adapter-free generation.

MLX is imported only inside :func:`mlx_generate_no_adapter`, behind the
``mlx_runtime`` opt-in gate. Importing this module never touches Metal.
"""

from __future__ import annotations

import re
from collections.abc import Callable, MutableSequence, Sequence
from typing import Any

from personality_protect.chat_prompt import (
    Message,
    flatten_chat_messages,
    render_chat_prompt,
)
from personality_protect.config import DEFAULT_MLX_MODEL, ProfilePaths, load_config
from personality_protect.draft_trim import trim_draft
from personality_protect.prompt_write import build_write_messages
from personality_protect.style_profile import (
    draft_word_target,
    load_style_profile,
    style_directives,
)
from personality_protect.voice_index import retrieve
from personality_protect.writer_guards import (
    check_invention,
    mask_exemplar_entities,
    parrot_reject,
)

# Five 120-word exemplars put ~600 words of the author's prose in front of the
# model and it answered by continuing them: every RAG draft came back as a
# multi-post dump while the exemplar-free arm wrote clean posts. Voice now
# travels mainly as measured cadence targets, so two short excerpts suffice.
DEFAULT_WRITE_K = 2
MIN_WRITE_K = 0
MAX_WRITE_K = 5
# Long LinkedIn posts (~500 words) need headroom past the trim target so the
# model can finish the last paragraph before draft_trim cuts.
DEFAULT_WRITE_MAX_TOKENS = 1536
MAX_WRITE_ATTEMPTS = 2
MAX_EXEMPLAR_WORDS = 60

# Generators take chat messages, not a flat string: the chat template can only
# be applied where the tokenizer lives.
GenerateFn = Callable[..., str]
PromptSink = MutableSequence[str]


def resolve_writer_adapter(paths: ProfilePaths) -> str | None:
    """Return an adapter directory when a writer LoRA is present on disk."""
    adapters = paths.adapters_dir
    weights = adapters / "adapters.safetensors"
    if weights.is_file():
        return str(adapters)
    return None


def mlx_generate_no_adapter(
    messages: Sequence[Message],
    *,
    base_model: str,
    max_tokens: int = DEFAULT_WRITE_MAX_TOKENS,
    adapter_path: str | None = None,
    prompt_sink: PromptSink | None = None,
) -> str:
    """Generate from MLX weights, optionally with a writer LoRA.

    ``adapter_path=None`` keeps the Camp A RAG default (base only). When a
    gated writer adapter directory is supplied, MLX loads it for the call.
    Tests inject ``generate_fn`` instead of calling this.

    ``messages`` are rendered through the tokenizer's chat template. Skipping
    that step makes the instruct model continue the prompt as a document and
    echo its section headers back. ``prompt_sink`` receives the exact final
    prompt string so callers can persist it for human review.
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

        model, tokenizer = load(base_model, adapter_path=adapter_path)
        prompt = render_chat_prompt(
            tokenizer,
            messages,
            fallback=flatten_chat_messages(messages),
        )
        if prompt_sink is not None:
            prompt_sink.append(prompt)
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
        lambda match: (
            match.group(1)
            if match.group(1) in kept
            else match.group(1)[0].lower() + match.group(1)[1:]
        ),
        text,
    )


def clip_exemplar(text: str, *, max_words: int = MAX_EXEMPLAR_WORDS) -> str:
    """First ``max_words`` words of an exemplar, keeping its line breaks."""
    kept: list[str] = []
    used = 0
    for line in (text or "").splitlines():
        words = line.split()
        if used + len(words) > max_words:
            remaining = max_words - used
            if remaining > 0:
                kept.append(" ".join(words[:remaining]))
            break
        kept.append(line)
        used += len(words)
    return "\n".join(kept).strip()


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
    channel: str = "post",
    use_adapter: bool = False,
    generate_fn: GenerateFn | None = None,
    prompt_sink: PromptSink | None = None,
) -> dict[str, Any]:
    """Retrieve exemplars, generate a draft, regenerate once on guard failure.

    ``channel='article'`` delegates to the article outline→section→stitch path.
    ``use_adapter`` loads a local writer LoRA when ``adapters.safetensors`` exists.
    """
    resolved = (channel or "post").strip().lower()
    if resolved == "article":
        from personality_protect.write_article import run_write_article

        return run_write_article(
            topic,
            points,
            paths,
            k=k,
            max_tokens=max_tokens,
            generate_fn=generate_fn,
            prompt_sink=prompt_sink,
        )
    if resolved != "post":
        raise ValueError("channel must be one of: post, article")

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
    matches = retrieve(brief, k=k, profile=paths.name, home=paths.home) if k else []
    if k and not matches:
        raise FileNotFoundError(
            f"No voice exemplars indexed for profile {paths.name}. "
            "Run: personality-protect index-voice"
        )

    exemplars = [str(match["text"]) for match in matches]
    masked = [mask_exemplar_entities(clip_exemplar(exemplar), brief) for exemplar in exemplars]
    style = load_style_profile(paths)
    directives = style_directives(style)
    word_target = draft_word_target(style)
    messages = build_write_messages(
        topic=topic,
        points=points,
        examples=masked,
        style_directives=directives,
    )

    adapter_path = resolve_writer_adapter(paths) if use_adapter else None
    if use_adapter and adapter_path is None:
        raise FileNotFoundError(
            f"No writer adapter at {paths.adapters_dir}/adapters.safetensors. "
            "Train one with: personality-protect train --writer"
        )

    generator = generate_fn or mlx_generate_no_adapter
    model_id = config.base_model or DEFAULT_MLX_MODEL

    draft = ""
    guards: dict[str, Any] = {}
    attempts = 0
    for attempts in range(1, MAX_WRITE_ATTEMPTS + 1):
        raw = str(
            generator(
                messages,
                base_model=model_id,
                max_tokens=max_tokens,
                adapter_path=adapter_path,
                prompt_sink=prompt_sink,
            )
        ).strip()
        # Guards score the edited draft, not the raw stream: the trimmed tail is
        # not part of the post, so letting it drive a rejection would be noise.
        draft = trim_draft(raw, max_words=word_target)
        guards = _guard_flags(brief, draft, exemplars)
        if not guards["parrot_reject"] and not guards["invent_reject"]:
            break

    return {
        "text": draft,
        "channel": "post",
        "voice_mode": config.voice_mode,
        "adapter": "writer" if adapter_path else "none",
        "write_adapter": adapter_path,
        "model": model_id,
        "k": len(matches),
        "exemplar_ids": [str(match["id"]) for match in matches],
        "attempts": attempts,
        "word_target": word_target,
        **guards,
        # Local-only debugging aids; CLI/receipts strip these (personal text).
        "exemplar_texts": exemplars,
        "messages": messages,
        "prompt": flatten_chat_messages(messages),
    }
