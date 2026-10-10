"""Local stand-in for the Qollab runtime. Run from the project root: python dev/serve.py [port]

GET  /                 index.html wrapped in a full document, import map points qec.bridge at dev/bridge_shim.js
GET  /<path>           static files from the project root
POST /api/<fn>         {"args": [...]} -> qec.bridge.<fn>(*args); draw returns SVG; run executes the built circuits on Aer
"""
import io
import json
import mimetypes
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from qec import bridge
from qec.aer_compat import simulate, stand_in_noise

# Same stand-in noise and simulator choice as tools/record_take.py so the page looks like a noisy device.
NOISE = stand_in_noise()
BACKEND_NAME = "local dev harness (Aer, depolarizing stand-in)"

mimetypes.add_type("text/javascript", ".js")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Logical teleportation on a tiny surface code (dev harness)</title>
<link rel="stylesheet" href="/main.css">
<script type="importmap">{"imports": {"qec.bridge": "/dev/bridge_shim.js"}}</script>
</head><body>
%s
<script type="module" src="/main.js"></script>
</body></html>"""


def run(shots):
    return simulate(bridge.circuits(), int(shots), noise_model=NOISE)


def api(fn, args):
    """Return (content_type, bytes)."""
    if fn == "run":
        return "application/json", json.dumps(run(*args)).encode()
    if fn == "circuits":
        return "application/json", json.dumps(len(bridge.circuits())).encode()
    if fn == "circuits_subset":   # run() still executes every built circuit; the harness fits all of them
        return "application/json", json.dumps(len(bridge.circuits_subset(*args))).encode()
    if fn == "draw":
        buf = io.BytesIO()
        bridge.draw(*args).savefig(buf, format="svg", bbox_inches="tight")
        return "image/svg+xml", buf.getvalue()
    if fn in ("build", "analyze", "load_recorded", "export_take", "spacetime"):
        return "application/json", getattr(bridge, fn)(*args).encode()
    raise KeyError(fn)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (PAGE % (ROOT / "index.html").read_text(encoding="utf-8")).encode()
            self._send("text/html; charset=utf-8", body)
        else:
            super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        args = json.loads(self.rfile.read(n) or b"{}").get("args", [])
        try:
            ctype, body = api(self.path[len("/api/"):], args)
        except Exception as e:  # surface the Python error in the page banner
            self._send("text/plain; charset=utf-8", f"{type(e).__name__}: {e}".encode(), 500)
            return
        self._send(ctype, body)

    def _send(self, ctype, body, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"dev harness at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
