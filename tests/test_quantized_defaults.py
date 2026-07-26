"""Tests for quantized model defaults and download helpers."""

from __future__ import annotations

from pathlib import Path

from personality_protect.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_FILE,
    DEFAULT_MLX_MODEL,
    FULL_PRECISION_BASE_MODEL,
    init_profile,
)
from personality_protect.download import resolve_gguf_path, run_download
from personality_protect.train import _train_model_id, backend_docs


def test_defaults_are_quantized_not_full_bf16():
    assert DEFAULT_BASE_MODEL == DEFAULT_MLX_MODEL
    assert "4bit" in DEFAULT_BASE_MODEL.lower() or "gguf" in DEFAULT_BASE_MODEL.lower()
    assert DEFAULT_BASE_MODEL != FULL_PRECISION_BASE_MODEL
    assert DEFAULT_GGUF_FILE.endswith(".gguf")
    assert "Q4" in DEFAULT_GGUF_FILE or "Q5" in DEFAULT_GGUF_FILE


def test_init_profile_stores_gguf_fields(tmp_path: Path):
    paths, config, created = init_profile("t", home=tmp_path)
    assert created
    assert config.base_model == DEFAULT_MLX_MODEL
    assert config.gguf_file == DEFAULT_GGUF_FILE
    assert paths.models_dir.is_dir()


def test_resolve_gguf_path(tmp_path: Path):
    paths, _, _ = init_profile("t", home=tmp_path)
    assert resolve_gguf_path(paths) is None
    fake = paths.models_dir / DEFAULT_GGUF_FILE
    fake.write_bytes(b"not-a-real-gguf")
    found = resolve_gguf_path(paths)
    assert found == fake


def test_download_missing_deps_is_honest(tmp_path: Path, monkeypatch):
    """Never hit the network in unit tests even if huggingface_hub is installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub" or (
            isinstance(name, str) and name.startswith("huggingface_hub.")
        ):
            raise ImportError("blocked in unit tests")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = run_download(format="gguf", home=tmp_path, profile="t")
    assert result.format == "gguf"
    assert result.status == "missing_deps"
    assert "5" in result.size_hint or "6" in result.size_hint or "7" in result.size_hint
    assert "18" not in result.size_hint and "20" not in result.size_hint


def test_backend_docs_never_promise_18_20_gb():
    for name in ("mlx", "cuda", "cpu", "mock"):
        docs = backend_docs(name)
        assert "18" not in docs
        assert "20 GB" not in docs
        assert "20GB" not in docs


def test_train_model_id_coerces_full_precision_on_mlx():
    assert _train_model_id(FULL_PRECISION_BASE_MODEL, "mlx") == DEFAULT_MLX_MODEL
    assert _train_model_id(DEFAULT_MLX_MODEL, "mlx") == DEFAULT_MLX_MODEL
