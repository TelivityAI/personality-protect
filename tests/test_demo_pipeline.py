"""End-to-end demo and mock train/filter tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from personality_protect.cli import app
from personality_protect.demo import run_demo
from personality_protect.filter import filter_draft
from personality_protect.train import detect_backend, run_train

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


def test_api_health(tmp_path: Path):
    from personality_protect.api import make_handler
    from personality_protect.config import init_profile
    from http.server import ThreadingHTTPServer
    import threading
    import urllib.request

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
