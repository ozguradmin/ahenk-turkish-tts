#!/usr/bin/env python3
"""Serve site/ on localhost for testing, optionally cross-origin isolated (multi-threaded WebAssembly).

    python tools/web/serve.py [--port 8000] [--isolate]

Plain mode is what GitHub Pages gives (no COOP/COEP headers; site/coi-serviceworker.js adds them in browsers that
allow service workers). --isolate sends Cross-Origin-Opener-Policy: same-origin and
Cross-Origin-Embedder-Policy: require-corp itself, for browsers or panes without service workers.
On localhost the page reads the model from site/models/ (see site/js/config.js).
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

SITE = Path(__file__).resolve().parents[2] / "site"


class Handler(http.server.SimpleHTTPRequestHandler):
    isolate = False
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".js": "text/javascript", ".onnx": "application/octet-stream", ".py": "text/plain; charset=utf-8",
                      ".json": "application/json"}

    def end_headers(self):
        if self.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--isolate", action="store_true")
    a = ap.parse_args(argv)
    Handler.isolate = a.isolate
    handler = functools.partial(Handler, directory=str(SITE))
    with http.server.ThreadingHTTPServer((a.bind, a.port), handler) as httpd:
        print(f"serving {SITE} at http://localhost:{a.port}/ {'(cross-origin isolated)' if a.isolate else ''}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
