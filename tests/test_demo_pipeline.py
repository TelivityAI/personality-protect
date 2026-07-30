"""End-to-end demo and mock train/filter tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.demo import run_demo
from personality_protect.train import detect_backend

runner = CliRunner()


def test_demo_pipeline(tmp_path: Path):
    result = run_demo(home=tmp_path)
    assert result["ingested"] >= 1
    assert result["selected"] >= 1
    assert result["train_status"] == "ok"
    assert result["train_backend"] == "mock"
    assert "leverage" not in result["rewritten"].lower() or "rewritten locally" in result["rewritten"]
    assert result["filter_backend"] == "mock"


def test_cli_demo_json(tmp_path: Path):
    res = runner.invoke(
        app,
        ["--logo", "off", "demo", "--home", str(tmp_path), "--json"],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["train_status"] == "ok"
    assert "rewritten" in data


def test_cli_init_ingest_select_train_filter(tmp_path: Path):
    home = str(tmp_path)
    fixtures = Path(__file__).parent / "fixtures"

    r = runner.invoke(app, ["--logo", "off", "init", "--home", home, "--profile", "t", "--json"])
    assert r.exit_code == 0, r.output

    r = runner.invoke(
        app,
        [
            "--logo", "off", "ingest",
            "--home", home, "--profile", "t",
            "--linkedin", str(fixtures / "linkedin"),
            "--path", str(fixtures / "local_docs"),
            "--source", "note",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["added"] > 0

    r = runner.invoke(
        app,
        [
            "--logo", "off", "select",
            "--home", home, "--profile", "t",
            "--min-words", "20",
            "--through-year", "2024",
            "--include-undated",
            "--force",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    sel = json.loads(r.output)
    assert sel["summary"]["pieces"] > 0

    r = runner.invoke(
        app,
        [
            "--logo", "off", "train",
            "--home", home, "--profile", "t",
            "--backend", "mock",
            "--smoke",
            "--force",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["status"] == "ok"

    r = runner.invoke(
        app,
        [
            "--logo", "off", "filter",
            "--home", home, "--profile", "t",
            "--backend", "mock",
            "--text", "In today's fast-paced world we must leverage synergies.",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["backend"] == "mock"
    assert "text" in out


def test_detect_backend_mock():
    assert detect_backend("mock") == "mock"


def test_demo_runs_write_path(tmp_path: Path):
    """Demo must exercise index-voice → style profile → write, not just train/filter."""
    result = run_demo(home=tmp_path)
    assert result["indexed"] >= 1
    assert result["style_pieces"] >= 1
    assert result["banned_ai_filler"] >= 1
    assert result["write_channel"] == "post"
    assert result["write_adapter"] == "none"
    assert result["write_stubbed_model"] is True
    # Guards must have passed — run_write raises/flags rather than return junk.
    assert "Contoso" in result["written"]


def test_style_profile_requires_selection(tmp_path: Path):
    """README quick start must keep `select` before `build-style-profile`."""
    import pytest

    from personality_protect.config import get_paths, init_profile
    from personality_protect.style_profile import run_build_style_profile

    init_profile("t", home=tmp_path)
    with pytest.raises(FileNotFoundError):
        run_build_style_profile(get_paths("t", home=tmp_path))


def test_select_default_includes_current_year(tmp_path: Path):
    """A corpus written this year must not select to zero under bare `select`."""
    from datetime import datetime, timezone

    from personality_protect.config import DEFAULT_THROUGH_YEAR
    from personality_protect.models import Piece
    from personality_protect.select import filter_pieces

    this_year = datetime.now(timezone.utc).year
    assert DEFAULT_THROUGH_YEAR >= this_year

    recent = Piece(
        id="r1",
        source="linkedin_post",
        text="word " * 200,
        year=this_year,
        word_count=200,
    )
    kept = filter_pieces(
        [recent],
        min_words=50,
        through_year=DEFAULT_THROUGH_YEAR,
        include_undated=False,
    )
    assert [p.id for p in kept] == ["r1"]


def test_api_health(tmp_path: Path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from personality_protect.api import make_handler
    from personality_protect.config import init_profile

    init_profile("default", home=tmp_path)
    # run demo so filter has adapter
    run_demo(home=tmp_path)

    handler = make_handler("demo")
    # Patch get_paths home via env
    import os

    os.environ["PERSONALITY_PROTECT_HOME"] = str(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            data = json.loads(resp.read().decode())
        assert data["ok"] is True
        assert data["local_only"] is True

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/filter",
            data=json.dumps({"text": "We must leverage robust synergies."}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
        assert body["ok"] is True
        assert body["local_only"] is True
        assert body["text"]
    finally:
        httpd.shutdown()
        os.environ.pop("PERSONALITY_PROTECT_HOME", None)
