"""Synthetic end-to-end demo (safe for public screenshots)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from personality_protect.config import ProfilePaths, init_profile
from personality_protect.filter import filter_draft
from personality_protect.ingest import run_ingest
from personality_protect.select import run_select
from personality_protect.train import run_train

DEMO_PROFILE = "demo"

# Inline fallback if package data missing
_FALLBACK_DOCS = {
    "post_2023_craft.md": """Date: 2023-06-12

I've spent ten years writing for humans, not algorithms. The posts that last
are the ones that sound like someone you would trust over coffee. Short
sentences when the point is sharp. Longer ones when the idea needs room to
breathe. I cut the corporate fog and keep the spine of the argument.

If a paragraph could belong to anyone, I rewrite it until it could only belong
to me. That habit is the whole product.
""",
    "note_2022_voice.txt": """2022-11-03

Voice notes to self: prefer concrete nouns. Avoid "leverage" and "synergy."
Say what happened, then what it meant. Readers forgive bluntness faster than
they forgive vagueness. When I am unsure, I say so — hedging is honest when
it is earned, lazy when it is a costume.
""",
    "article_2024_local.md": """# Why local beats cloud for personal style
Published: 2024-03-18

Your writing style is biometric-adjacent. Shipping it to a rented GPU farm so
someone else's stack can imitate you is a strange bargain. Keep the corpus on
your machine. Train a small adapter. Filter drafts before they go public.

I do not need a trillion-parameter oracle to sound like myself. I need a
faithful mirror that never leaves the desk.
""",
    "comment_2021.txt": """2021-08-21

Appreciate this thread. One caveat from the trenches: most "AI writing tips"
optimize for average. Average is the enemy of a recognizable voice. Ship fewer
words with more fingerprints.
""",
    "email_2020.txt": """Date: 2020-01-15

Subject: Quick note on the draft

Thanks for the pass. Two nits: the opener buries the lede, and the middle
section leans on stock phrases. Punch up the first three lines with what we
actually learned last quarter. Happy to sync Friday.
""",
}


def demo_data_dir() -> Path:
    """Return path to packaged synthetic demo corpus."""
    try:
        root = resources.files("personality_protect").joinpath("data/demo")
        # materialize Traversable to Path when possible
        if hasattr(root, "iterdir"):
            return Path(str(root))
    except Exception:
        pass
    # Repo layout fallback
    here = Path(__file__).resolve().parent
    candidate = here / "data" / "demo"
    if candidate.is_dir():
        return candidate
    return here / "data" / "demo"


def ensure_demo_corpus(target: Path) -> Path:
    """Write synthetic demo files into target (profile cache or temp)."""
    target.mkdir(parents=True, exist_ok=True)
    pkg = demo_data_dir()
    written = 0
    if pkg.is_dir():
        for src in pkg.iterdir():
            if src.is_file() and src.suffix.lower() in {".md", ".txt", ".jsonl"}:
                dest = target / src.name
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                written += 1
    if written == 0:
        for name, body in _FALLBACK_DOCS.items():
            (target / name).write_text(body, encoding="utf-8")
    return target


def run_demo(
    *,
    home: Path | None = None,
    draft: str | None = None,
) -> dict:
    """Full synthetic pipeline: init → ingest → select → mock train → filter."""
    paths, config, _ = init_profile(DEMO_PROFILE, home=home, force=True)
    corpus = ensure_demo_corpus(paths.cache_dir / "demo_corpus")
    added, _ = run_ingest(paths, local=[corpus], source_hint="demo")
    selection, selected = run_select(
        paths,
        min_words=20,  # synthetic pieces are shorter; demo overrides default
        through_year=2024,
        include_undated=True,
    )
    train = run_train(paths, backend="mock", mock=True, max_steps=1)

    sample_draft = draft or (
        "In today's fast-paced digital world, it is important to note that we must "
        "leverage robust synergies to delve into authentic personal branding."
    )
    rewritten, backend = filter_draft(sample_draft, paths, backend="mock")

    return {
        "profile": config.name,
        "home": str(paths.home),
        "ingested": added,
        "selected": len(selected),
        "selection_summary": selection.summary,
        "train_status": train.status,
        "train_backend": train.backend,
        "adapter_dir": train.adapter_dir,
        "filter_backend": backend,
        "draft": sample_draft,
        "rewritten": rewritten,
    }
