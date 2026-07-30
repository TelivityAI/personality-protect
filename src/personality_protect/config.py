"""Profile directories and local config. Corpus/adapters stay on-disk only."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- Quantized happy path (~5–7 GB on disk). NOT full BF16. ---
# Primary local runtime: Q4_K_M GGUF via llama.cpp (~5.6 GB).
DEFAULT_GGUF_REPO = "lmstudio-community/Qwen3.5-9B-GGUF"
DEFAULT_GGUF_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_GGUF_SIZE_HINT = "~5.6 GB"

# Apple Silicon train/filter: MLX 4-bit (~6 GB), not full-precision HF.
DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-9B-4bit"
DEFAULT_MLX_SIZE_HINT = "~6 GB"

# Escape hatch only — full-precision weights are ~18 GB and are NOT the product default.
FULL_PRECISION_BASE_MODEL = "Qwen/Qwen3.5-9B"

# Profile `base_model` defaults to the MLX 4-bit id (train-from-quantized on Mac).
# Filter prefers a local GGUF under models/ when present.
DEFAULT_BASE_MODEL = DEFAULT_MLX_MODEL

DEFAULT_MIN_WORDS = 50
DEFAULT_THROUGH_YEAR = 2024
DEFAULT_PROFILE = "default"
DEFAULT_VOICE_MODE = "rag"
DEFAULT_WRITE_ADAPTER: str | None = None

# Corpus size gates for credible full train (bypass with --force / --smoke)
CORPUS_WARN_BELOW = 50
CORPUS_BLOCK_BELOW = 20

# Explicit smoke / CI low-step default (not a silent full-train stand-in)
SMOKE_MAX_STEPS = 2


def default_home() -> Path:
    """Root for all local PersonalityProtect state (never committed)."""
    override = os.environ.get("PERSONALITY_PROTECT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".personality-protect").resolve()


@dataclass
class ProfileConfig:
    name: str = DEFAULT_PROFILE
    base_model: str = DEFAULT_BASE_MODEL
    gguf_repo: str = DEFAULT_GGUF_REPO
    gguf_file: str = DEFAULT_GGUF_FILE
    voice_mode: str = DEFAULT_VOICE_MODE
    write_adapter: str | None = DEFAULT_WRITE_ADAPTER
    min_words: int = DEFAULT_MIN_WORDS
    through_year: int = DEFAULT_THROUGH_YEAR
    created_at: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileConfig:
        return cls(
            name=str(data.get("name", DEFAULT_PROFILE)),
            base_model=str(data.get("base_model", DEFAULT_BASE_MODEL)),
            gguf_repo=str(data.get("gguf_repo", DEFAULT_GGUF_REPO)),
            gguf_file=str(data.get("gguf_file", DEFAULT_GGUF_FILE)),
            voice_mode=str(data.get("voice_mode", DEFAULT_VOICE_MODE)),
            write_adapter=(
                str(data["write_adapter"]) if data.get("write_adapter") is not None else None
            ),
            min_words=int(data.get("min_words", DEFAULT_MIN_WORDS)),
            through_year=int(data.get("through_year", DEFAULT_THROUGH_YEAR)),
            created_at=str(data.get("created_at", "")),
            sources=list(data.get("sources", [])),
        )


@dataclass
class ProfilePaths:
    """Layout under ~/.personality-protect/profiles/<name>/."""

    home: Path
    name: str

    @property
    def root(self) -> Path:
        return self.home / "profiles" / self.name

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def index_path(self) -> Path:
        return self.root / "index.jsonl"

    @property
    def selection_path(self) -> Path:
        return self.root / "selection.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def sft_dir(self) -> Path:
        return self.root / "sft"

    @property
    def sft_jsonl(self) -> Path:
        return self.sft_dir / "train.jsonl"

    @property
    def adapters_dir(self) -> Path:
        return self.root / "adapters"

    @property
    def adapter_meta(self) -> Path:
        return self.adapters_dir / "adapter_meta.json"

    @property
    def evals_dir(self) -> Path:
        """Local before/after eval receipts (never commit)."""
        return self.root / "evals"

    @property
    def models_dir(self) -> Path:
        """Shared quantized model cache (GGUF etc.) under the home root."""
        return self.home / "models"

    def gguf_path(self, filename: str | None = None) -> Path:
        return self.models_dir / (filename or DEFAULT_GGUF_FILE)

    def ensure(self) -> None:
        for d in (
            self.root,
            self.cache_dir,
            self.sft_dir,
            self.adapters_dir,
            self.evals_dir,
            self.models_dir,
            self.home / "profiles",
        ):
            d.mkdir(parents=True, exist_ok=True)


def get_paths(profile: str = DEFAULT_PROFILE, home: Path | None = None) -> ProfilePaths:
    return ProfilePaths(home=home or default_home(), name=profile)


def load_config(paths: ProfilePaths) -> ProfileConfig:
    if not paths.config_path.is_file():
        raise FileNotFoundError(
            f"No profile at {paths.root}. Run: personality-protect init"
        )
    data = json.loads(paths.config_path.read_text(encoding="utf-8"))
    return ProfileConfig.from_dict(data)


def save_config(paths: ProfilePaths, config: ProfileConfig) -> None:
    paths.ensure()
    paths.config_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def init_profile(
    name: str = DEFAULT_PROFILE,
    *,
    home: Path | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    force: bool = False,
) -> tuple[ProfilePaths, ProfileConfig, bool]:
    """Create profile dirs + config. Returns (paths, config, created)."""
    from datetime import datetime, timezone

    paths = get_paths(name, home=home)
    if paths.config_path.is_file() and not force:
        return paths, load_config(paths), False

    paths.ensure()
    # Privacy notice so users know what lives here
    notice = paths.root / "PRIVACY.txt"
    notice.write_text(
        "This directory holds your personal writing index, SFT JSONL, and LoRA "
        "adapters.\nIt must NEVER be committed to git or uploaded to the cloud.\n"
        "PersonalityProtect keeps all corpus and weights on this machine only.\n",
        encoding="utf-8",
    )
    config = ProfileConfig(
        name=name,
        base_model=base_model,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_config(paths, config)

    home_root = paths.home
    home_root.mkdir(parents=True, exist_ok=True)
    gitignore = home_root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")

    return paths, config, True
