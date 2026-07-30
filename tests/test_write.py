"""Contoso-safe Lane F write path: retrieve → prompt → mock-MLX → guards."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import init_profile
from personality_protect.models import Piece, save_index
from personality_protect.style_profile import build_style_profile, save_style_profile
from personality_protect.voice_index import build_voice_index
from personality_protect.write import (
    mlx_generate_no_adapter,
    run_write,
)

runner = CliRunner()


def _contoso_pieces() -> list[Piece]:
    return [
        Piece(
            id="contoso-pricing",
            source="linkedin_post",
            text=(
                "Pricing experiments work when teams compare customer value "
                "and renewal signals. Contoso kept the test boring on purpose."
            ),
        ),
        Piece(
            id="contoso-platform",
            source="linkedin_post",
            text=(
                "Platform migrations need careful service boundaries and "
                "boring rollback plans. Name one owner before you start."
            ),
        ),
        Piece(
            id="contoso-ops",
            source="linkedin_post",
            text=(
                "Operations queues improve when Contoso picks one metric "
                "and ignores the rest for a week."
            ),
        ),
        Piece(
            id="contoso-holdout",
            source="linkedin_post",
            text="Customer pricing and packaging lessons belong in the private holdout.",
        ),
    ]


def _seed_contoso_index(tmp_path: Path) -> None:
    paths, _, _ = init_profile("contoso", home=tmp_path)
    pieces = _contoso_pieces()
    save_index(paths.index_path, pieces)
    build_voice_index(paths, holdout_ids={"contoso-holdout"})
    save_style_profile(paths, build_style_profile(pieces))


@pytest.mark.parametrize("module", ["mlx", "mlx.nn", "mlx.core", "mlx_lm"])
def test_conftest_blocks_real_mlx_imports(module: str):
    """Stray mlx imports must raise ImportError, never SIGABRT on metal::load_device."""
    with pytest.raises(ImportError, match="MLX is blocked under pytest"):
        importlib.import_module(module)


def test_mlx_gate_fails_closed_without_explicit_opt_in(monkeypatch):
    from personality_protect import mlx_runtime as rt

    monkeypatch.delenv("PP_MLX_ALLOW", raising=False)
    monkeypatch.delenv("PP_MLX_DISABLE", raising=False)
    assert rt.mlx_import_allowed() is False
    with pytest.raises(RuntimeError, match="PP_MLX_ALLOW"):
        rt.assert_mlx_import_allowed()

    monkeypatch.setenv("PP_MLX_ALLOW", "1")
    assert rt.mlx_import_allowed() is True
    rt.assert_mlx_import_allowed()

    # Explicit disable still wins over opt-in.
    monkeypatch.setenv("PP_MLX_DISABLE", "1")
    assert rt.mlx_import_allowed() is False
    with pytest.raises(RuntimeError, match="PP_MLX_DISABLE"):
        rt.assert_mlx_import_allowed()


def test_write_module_import_does_not_import_mlx():
    import sys

    from personality_protect import write

    assert write.mlx_generate_no_adapter is not None
    assert not [name for name in sys.modules if name.split(".", 1)[0] == "mlx"]


def test_mlx_generate_no_adapter_applies_chat_template_and_never_passes_adapter(
    tmp_path: Path,
):
    """Write-path MLX load must force adapter_path=None and wrap the chat turns."""
    load_calls: list[dict] = []

    fake_model = object()
    fake_tokenizer = MagicMock()
    fake_tokenizer.has_chat_template = True
    fake_tokenizer.chat_template = "unused"
    fake_tokenizer.apply_chat_template.return_value = (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nWrite Contoso.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def fake_load(model_id, adapter_path=None, **kwargs):
        load_calls.append({"model_id": model_id, "adapter_path": adapter_path, **kwargs})
        return fake_model, fake_tokenizer

    fake_mlx_lm = MagicMock()
    fake_mlx_lm.load = fake_load
    fake_mlx_lm.generate = MagicMock(return_value="Contoso ships Ledger quietly.")
    sink: list[str] = []
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Write Contoso."},
    ]

    with (
        patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm}),
        patch("personality_protect.mlx_runtime.assert_mlx_import_allowed"),
        patch("personality_protect.mlx_runtime.ensure_mlx_wired_cap"),
        patch("personality_protect.mlx_runtime.release_mlx_memory"),
    ):
        out = mlx_generate_no_adapter(
            messages,
            base_model="mlx-community/Qwen3.5-9B-4bit",
            max_tokens=64,
            prompt_sink=sink,
        )

    assert out == "Contoso ships Ledger quietly."
    assert load_calls == [{"model_id": "mlx-community/Qwen3.5-9B-4bit", "adapter_path": None}]
    fake_tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = fake_tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["enable_thinking"] is False
    assert call_kwargs["tokenize"] is False
    generate_kwargs = fake_mlx_lm.generate.call_args.kwargs
    assert generate_kwargs["prompt"].startswith("<|im_start|>system")
    assert sink == [generate_kwargs["prompt"]]


def test_run_write_retrieves_masks_and_returns_adapter_none_receipt(tmp_path: Path):
    _seed_contoso_index(tmp_path)
    paths, config, _ = init_profile("contoso", home=tmp_path)
    assert config.write_adapter is None

    def fake_generate(messages, **_kwargs: object) -> str:
        assert [message["role"] for message in messages] == ["system", "user"]
        user = messages[1]["content"]
        assert "EXAMPLES (rhythm reference only" in user
        assert "BRIEF:" in user
        assert "Topic: Contoso pricing" in user
        assert "Fabrikam" not in user  # nothing to invent from fixtures
        # Masked exemplars should not leave holdout text in the prompt.
        assert "private holdout" not in user
        return (
            "Contoso pricing tests work when one owner watches renewal signals.\n"
            "Keep the experiment boring."
        )

    result = run_write(
        topic="Contoso pricing",
        points="Name one owner; watch renewal signals; keep the test boring.",
        paths=paths,
        k=3,
        generate_fn=fake_generate,
    )

    assert result["adapter"] == "none"
    assert result["model"] == config.base_model
    assert result["voice_mode"] == "rag"
    assert result["k"] == 3
    assert "contoso-holdout" not in result["exemplar_ids"]
    assert len(result["exemplar_ids"]) == 3
    assert result["attempts"] == 1
    assert result["parrot_reject"] is False
    assert result["invent_reject"] is False
    assert "Contoso pricing" in result["text"]
    assert result["write_adapter"] is None


def test_run_write_regenerates_once_on_parrot_guard(tmp_path: Path):
    _seed_contoso_index(tmp_path)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    calls: list[str] = []

    parrot = (
        "Pricing experiments work when teams compare customer value "
        "and renewal signals. Contoso kept the test boring on purpose."
    )
    clean = (
        "Contoso should name one owner for pricing tests and watch "
        "renewal signals without copying last quarter's memo."
    )

    def fake_generate(messages, **_kwargs: object) -> str:
        calls.append(messages)
        return parrot if len(calls) == 1 else clean

    result = run_write(
        topic="Contoso pricing ownership",
        points="Name one owner; watch renewal signals.",
        paths=paths,
        k=3,
        generate_fn=fake_generate,
    )

    assert len(calls) == 2
    assert result["attempts"] == 2
    assert result["parrot_reject"] is False
    assert result["text"] == clean
    assert result["adapter"] == "none"


def test_run_write_regenerates_once_on_invent_guard(tmp_path: Path):
    _seed_contoso_index(tmp_path)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    calls: list[int] = []

    invented = "Contoso should copy Fabrikam and cut Ledger exceptions by 18%."
    clean = "Contoso should name one owner and simplify Ledger exceptions."

    def fake_generate(messages, **_kwargs: object) -> str:
        calls.append(1)
        return invented if len(calls) == 1 else clean

    result = run_write(
        topic="Contoso Ledger exceptions",
        points="Name one owner; simplify Ledger exceptions.",
        paths=paths,
        k=3,
        generate_fn=fake_generate,
    )

    assert len(calls) == 2
    assert result["attempts"] == 2
    assert result["invent_reject"] is False
    assert "Fabrikam" not in result["text"]
    assert result["adapter"] == "none"


def test_sentence_initial_words_are_not_invented_entities_but_names_are():
    from personality_protect.write import normalize_sentence_case
    from personality_protect.writer_guards import check_invention

    brief = "Topic: Contoso pricing\nPoints: Name one owner."
    ordinary = "Keep the test boring.\nName one owner before you ship."
    invented = "Contoso should copy Fabrikam before shipping."

    assert check_invention(brief, normalize_sentence_case(ordinary)).passed is True
    result = check_invention(brief, normalize_sentence_case(invented))
    assert result.passed is False
    assert "fabrikam" in result.invented_entities


def test_run_write_reports_guard_failure_after_one_regen(tmp_path: Path):
    _seed_contoso_index(tmp_path)
    paths, _, _ = init_profile("contoso", home=tmp_path)
    calls: list[int] = []

    def fake_generate(messages, **_kwargs: object) -> str:
        calls.append(1)
        return "Contoso should copy Fabrikam and cut exceptions by 18%."

    result = run_write(
        topic="Contoso Ledger exceptions",
        points="Name one owner.",
        paths=paths,
        k=3,
        generate_fn=fake_generate,
    )

    assert len(calls) == 2  # never more than one regen
    assert result["attempts"] == 2
    assert result["invent_reject"] is True
    assert result["invented_entities"] == ["fabrikam"]
    assert result["invented_numbers"] == ["18%"]


def test_cli_write_rejects_adapter_flag(tmp_path: Path):
    _seed_contoso_index(tmp_path)

    result = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "write",
            "--topic",
            "Contoso pricing",
            "--points",
            "Name one owner.",
            "--profile",
            "contoso",
            "--home",
            str(tmp_path),
            "--adapter",
        ],
    )

    assert result.exit_code == 2
    assert "adapter=none" in result.output


def test_cli_write_exits_nonzero_when_guards_still_fail(tmp_path: Path):
    _seed_contoso_index(tmp_path)

    with patch(
        "personality_protect.write.mlx_generate_no_adapter",
        return_value="Contoso should copy Fabrikam today.",
    ):
        result = runner.invoke(
            app,
            [
                "--logo",
                "off",
                "write",
                "--topic",
                "Contoso pricing",
                "--points",
                "Name one owner.",
                "--profile",
                "contoso",
                "--home",
                str(tmp_path),
                "--json",
            ],
        )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["invent_reject"] is True
    assert payload["adapter"] == "none"


def test_cli_write_json_receipt_adapter_none(tmp_path: Path):
    _seed_contoso_index(tmp_path)

    good = (
        "Contoso pricing tests need one owner watching renewal signals.\n"
        "Keep the experiment boring."
    )

    with patch(
        "personality_protect.write.mlx_generate_no_adapter",
        return_value=good,
    ) as gen:
        result = runner.invoke(
            app,
            [
                "--logo",
                "off",
                "write",
                "--topic",
                "Contoso pricing",
                "--points",
                "Name one owner; watch renewal signals; keep the test boring.",
                "--k",
                "3",
                "--profile",
                "contoso",
                "--home",
                str(tmp_path),
                "--no-adapter",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["adapter"] == "none"
    assert payload["write_adapter"] is None
    assert payload["k"] == 3
    assert "contoso-holdout" not in payload["exemplar_ids"]
    assert "Contoso pricing" in payload["text"]
    gen.assert_called()
    # Forced no-adapter path — never an adapter dir string.
    for call in gen.call_args_list:
        assert call.kwargs.get("adapter_path") in (None, ...)
        # positional/kw: function signature has no adapter_path param
        assert "adapter_path" not in call.kwargs or call.kwargs["adapter_path"] is None
