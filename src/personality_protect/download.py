"""Download quantized local models (~5–7 GB). Never the full BF16 default."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from personality_protect.config import (
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_REPO,
    DEFAULT_GGUF_SIZE_HINT,
    DEFAULT_MLX_MODEL,
    DEFAULT_MLX_SIZE_HINT,
    ProfilePaths,
    get_paths,
)

DownloadFormat = Literal["gguf", "mlx"]


@dataclass
class DownloadResult:
    format: str
    status: str
    path: str
    size_hint: str
    notes: str = ""
    repo: str = ""
    filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_gguf_path(
    paths: ProfilePaths,
    *,
    filename: str | None = None,
    explicit: Path | None = None,
) -> Path | None:
    """Return an existing local GGUF path, or None."""
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_file() else None
    candidate = paths.gguf_path(filename)
    if candidate.is_file():
        return candidate
    # Also accept any *.gguf in models/
    if paths.models_dir.is_dir():
        found = sorted(paths.models_dir.glob("*.gguf"))
        if found:
            return found[0]
    return None


def download_gguf(
    paths: ProfilePaths,
    *,
    repo: str = DEFAULT_GGUF_REPO,
    filename: str = DEFAULT_GGUF_FILE,
) -> DownloadResult:
    """Fetch one Q4/Q5-class GGUF into ~/.personality-protect/models/."""
    paths.ensure()
    dest = paths.gguf_path(filename)
    if dest.is_file() and dest.stat().st_size > 1_000_000_000:
        return DownloadResult(
            format="gguf",
            status="exists",
            path=str(dest),
            size_hint=DEFAULT_GGUF_SIZE_HINT,
            notes=f"Already present ({dest.stat().st_size // (1024**3)} GB on disk).",
            repo=repo,
            filename=filename,
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return DownloadResult(
            format="gguf",
            status="missing_deps",
            path=str(dest),
            size_hint=DEFAULT_GGUF_SIZE_HINT,
            notes=(
                "huggingface_hub required. Install: pip install -e \".[models]\"\n"
                f"Then: personality-protect download --format gguf\n"
                f"Or manually place {filename} in {paths.models_dir}"
            ),
            repo=repo,
            filename=filename,
        )

    try:
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(paths.models_dir),
        )
        dest = Path(downloaded)
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(
            format="gguf",
            status="error",
            path=str(dest),
            size_hint=DEFAULT_GGUF_SIZE_HINT,
            notes=f"GGUF download failed: {exc}",
            repo=repo,
            filename=filename,
        )

    size = dest.stat().st_size if dest.is_file() else 0
    gb = max(1, size // (1024**3)) if size else 0
    return DownloadResult(
        format="gguf",
        status="ok",
        path=str(dest),
        size_hint=DEFAULT_GGUF_SIZE_HINT,
        notes=(
            f"Downloaded quantized GGUF (~{gb} GB on disk). "
            "Ready for: personality-protect filter --backend llama"
        ),
        repo=repo,
        filename=filename,
    )


def download_mlx(
    paths: ProfilePaths,
    *,
    repo: str = DEFAULT_MLX_MODEL,
) -> DownloadResult:
    """Prefetch MLX 4-bit weights into the Hugging Face cache (~6 GB)."""
    paths.ensure()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return DownloadResult(
            format="mlx",
            status="missing_deps",
            path=repo,
            size_hint=DEFAULT_MLX_SIZE_HINT,
            notes='huggingface_hub required. Install: pip install -e ".[models]"',
            repo=repo,
        )

    try:
        local = snapshot_download(repo_id=repo)
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(
            format="mlx",
            status="error",
            path=repo,
            size_hint=DEFAULT_MLX_SIZE_HINT,
            notes=f"MLX 4-bit download failed: {exc}",
            repo=repo,
        )

    return DownloadResult(
        format="mlx",
        status="ok",
        path=local,
        size_hint=DEFAULT_MLX_SIZE_HINT,
        notes=(
            f"MLX 4-bit model cached locally ({DEFAULT_MLX_SIZE_HINT}). "
            "Use for train --backend mlx / filter --backend mlx."
        ),
        repo=repo,
    )


def run_download(
    *,
    format: DownloadFormat = "gguf",
    home: Path | None = None,
    profile: str = "default",
    repo: str | None = None,
    filename: str | None = None,
) -> DownloadResult:
    paths = get_paths(profile, home=home)
    if format == "gguf":
        return download_gguf(
            paths,
            repo=repo or DEFAULT_GGUF_REPO,
            filename=filename or DEFAULT_GGUF_FILE,
        )
    if format == "mlx":
        return download_mlx(paths, repo=repo or DEFAULT_MLX_MODEL)
    raise ValueError(f"Unknown format: {format}")
