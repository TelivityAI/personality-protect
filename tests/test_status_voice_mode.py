"""Contoso-safe Lane H: status reports Camp A RAG voice_mode + adapter=none."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.config import init_profile

runner = CliRunner()


def test_status_json_shows_rag_voice_mode_adapter_none(tmp_path: Path):
    init_profile("contoso", home=tmp_path)

    result = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "status",
            "--profile",
            "contoso",
            "--home",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["voice_mode"] == "rag"
    assert payload["adapter"] == "none"
    assert payload["write_adapter"] is None


def test_status_text_shows_rag_voice_mode_adapter_none(tmp_path: Path):
    init_profile("contoso", home=tmp_path)

    result = runner.invoke(
        app,
        [
            "--logo",
            "off",
            "status",
            "--profile",
            "contoso",
            "--home",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "voice_mode: rag" in result.output
    assert "adapter: none" in result.output
    assert "write_adapter: none" in result.output
