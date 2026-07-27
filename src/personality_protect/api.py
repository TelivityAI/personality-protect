"""Local-only HTTP API stub for a future browser extension.

Binds to 127.0.0.1 only. Never accepts remote connections by default.
Does not upload corpus or adapters.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from personality_protect import __version__
from personality_protect.config import DEFAULT_PROFILE, get_paths, load_config
from personality_protect.filter import filter_draft

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(profile: str = DEFAULT_PROFILE) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/health"):
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "service": "personality-protect",
                        "version": __version__,
                        "local_only": True,
                        "profile": profile,
                        "endpoints": {
                            "GET /health": "liveness",
                            "POST /v1/filter": "rewrite draft with local adapter",
                        },
                    },
                )
                return
            _json_response(self, 404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/v1/filter":
                _json_response(self, 404, {"ok": False, "error": "not_found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                _json_response(self, 400, {"ok": False, "error": "invalid_json"})
                return
            draft = str(data.get("text") or data.get("draft") or "")
            if not draft.strip():
                _json_response(self, 400, {"ok": False, "error": "missing_text"})
                return
            try:
                paths = get_paths(profile)
                load_config(paths)
                mt = data.get("max_tokens")
                rewritten, backend = filter_draft(
                    draft,
                    paths,
                    backend=str(data.get("backend") or "auto"),  # type: ignore[arg-type]
                    max_tokens=int(mt) if mt is not None else None,
                    force=bool(data.get("force")),
                )
            except FileNotFoundError as exc:
                _json_response(self, 404, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                _json_response(self, 500, {"ok": False, "error": str(exc)})
                return
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "backend": backend,
                    "text": rewritten,
                    "local_only": True,
                },
            )

    return Handler


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    profile: str = DEFAULT_PROFILE,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            f"Refusing to bind to {host!r}. Local API is 127.0.0.1-only for privacy."
        )
    httpd = ThreadingHTTPServer((host, port), make_handler(profile))
    print(f"personality-protect API on http://{host}:{port} (local only)")
    print("POST /v1/filter  GET /health   Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
