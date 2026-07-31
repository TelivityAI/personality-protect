"""Shared MLX Metal wired-memory safety for train AND filter/compare.

mlx-lm's ``generate.wired_limit`` and trainer both call
``mx.set_wired_limit(max_recommended_working_set_size)`` (~40 GB on a 48 GB
Mac). That jetsam-kills Python ("quit unexpectedly"). Every MLX entrypoint
must install this cap before load/generate/train.

Importing ``mlx`` / ``mlx_lm`` inside Cursor's sandboxed shell also SIGABRTs
(Metal unavailable → uncaught C++ terminate). MLX is therefore blocked by
default: call ``assert_mlx_import_allowed`` before any ``import mlx*``, and set
``PP_MLX_ALLOW=1`` only in a shell that has a real Metal device.
"""

from __future__ import annotations

import os
from typing import Any

from personality_protect.mlx_train import (
    DEFAULT_WIRED_CAP_BYTES,
    DEFAULT_WIRED_FRACTION,
    detect_device_memory,
    resolve_wired_limit_bytes,
)

_CAP_INSTALLED_FOR: int | None = None


MLX_ALLOW_ENV = "PP_MLX_ALLOW"
MLX_DISABLE_ENV = "PP_MLX_DISABLE"
_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


def mlx_import_allowed() -> bool:
    """True only when MLX use is explicitly opted into and not force-disabled."""
    return _env_flag(MLX_ALLOW_ENV) and not _env_flag(MLX_DISABLE_ENV)


def assert_mlx_import_allowed() -> None:
    """Raise before ``import mlx*`` unless MLX was explicitly opted into.

    Fails closed: importing ``mlx`` in a sandboxed shell has no Metal device and
    SIGABRTs via ``metal::load_device``, killing Python with a crash dialog
    instead of an exception. Callers must set ``PP_MLX_ALLOW=1`` in a real
    GPU-capable shell; ``PP_MLX_DISABLE=1`` still overrides and blocks.
    """
    if _env_flag(MLX_DISABLE_ENV):
        raise RuntimeError(
            f"MLX import blocked ({MLX_DISABLE_ENV}=1). "
            "Do not load mlx/mlx_lm in sandboxed or headless sessions — "
            "Metal abort kills Python. Use mock backend or a full-GPU shell."
        )
    if not _env_flag(MLX_ALLOW_ENV):
        raise RuntimeError(
            f"MLX import blocked by default. Set {MLX_ALLOW_ENV}=1 in a shell with "
            "a real Metal device to load mlx/mlx_lm. Sandboxed or headless "
            "sessions SIGABRT on metal::load_device — use the mock backend there."
        )


def install_wired_cap(limit_bytes: int) -> int:
    """Monkeypatch ``mx.set_wired_limit`` so callers cannot request more than ``limit_bytes``.

    Returns the effective limit applied.
    """
    assert_mlx_import_allowed()
    import mlx.core as mx

    limit = max(1_000_000_000, int(limit_bytes))
    global _CAP_INSTALLED_FOR

    # Avoid stacking wrappers if already capped at this limit.
    current = mx.set_wired_limit
    if (
        _CAP_INSTALLED_FOR == limit
        and getattr(current, "_pp_capped", False)
        and getattr(current, "_pp_limit", None) == limit
    ):
        return limit

    # Unwrap prior PP wrapper so we don't nest indefinitely.
    original = getattr(current, "_pp_original", current)

    def _capped(requested: Any = None, *args: Any, **kwargs: Any) -> Any:
        try:
            req = int(requested) if requested is not None else limit
        except (TypeError, ValueError):
            req = limit
        return original(min(req, limit))

    _capped._pp_capped = True  # type: ignore[attr-defined]
    _capped._pp_limit = limit  # type: ignore[attr-defined]
    _capped._pp_original = original  # type: ignore[attr-defined]
    mx.set_wired_limit = _capped  # type: ignore[method-assign]
    _CAP_INSTALLED_FOR = limit
    try:
        mx.set_wired_limit(limit)
    except Exception:
        pass
    return limit


def ensure_mlx_wired_cap(*, memory_gb: float | None = None) -> int:
    """Install a safe wired cap for the current process (idempotent).

    Honors ``PP_MLX_WIRED_BYTES`` / ``PP_MLX_MEMORY_GB`` env overrides.
    Default hard cap is 16 GB. Prefer ``PP_MLX_MEMORY_GB<=16`` on Studio.
    """
    assert_mlx_import_allowed()
    env_bytes = os.environ.get("PP_MLX_WIRED_BYTES")
    if env_bytes:
        try:
            return install_wired_cap(int(env_bytes))
        except ValueError:
            pass
    if memory_gb is None:
        env_gb = os.environ.get("PP_MLX_MEMORY_GB")
        if env_gb:
            try:
                memory_gb = float(env_gb)
            except ValueError:
                memory_gb = None

    mem_size, max_rec = detect_device_memory()
    wired = resolve_wired_limit_bytes(
        memory_size=mem_size,
        max_recommended=max_rec,
        memory_gb=memory_gb,
        fraction=DEFAULT_WIRED_FRACTION,
        cap_bytes=DEFAULT_WIRED_CAP_BYTES,
    )
    os.environ["PP_MLX_WIRED_BYTES"] = str(wired)
    return install_wired_cap(wired)


def release_mlx_memory() -> None:
    """Best-effort Metal cache clear after filter/generate.

    The opt-in check is not decoration: a Metal-less session aborts inside
    ``metal::load_device`` via C++ ``terminate``, which ``except Exception``
    cannot catch. Without the guard a cleanup call in a sandboxed process takes
    the interpreter down with a crash dialog instead of returning quietly.
    """
    if not mlx_import_allowed():
        return
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass
