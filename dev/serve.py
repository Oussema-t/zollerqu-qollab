"""Local stand-in for the Qollab runtime. Run from the project root: python dev/serve.py [port]

GET  /                 index.html wrapped in a full document, import map points qopt.bridge at dev/bridge_shim.js
GET  /<path>           static files from the project root
POST /api/<fn>         {"args": [...]} -> qopt.bridge.<fn>(*args); sweep_circuits_py and dh_next_circuit keep their
                       circuits server side and return how many; run executes them on Aer and returns the counts
"""
import json
import mimetypes
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qopt import bridge
from qopt.aer_run import aer_run

bridge.load_data((ROOT / "data" / "champaign.json").read_text(encoding="utf-8"))
CURRENT = []   # circuits built by the last sweep_circuits_py / dh_next_circuit call

mimetypes.add_type("text/javascript", ".js")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Store placement by quantum minimum finding (dev harness)</title>
<link rel="stylesheet" href="/main.css">
<script type="importmap">{"imports": {"qopt.bridge": "/dev/bridge_shim.js"}}</script>
</head><body>
%s
<script type="module" src="/main.js"></script>
</body></html>"""

TEXT_FNS = ("load_data", "cost_table_json", "classical", "sweep", "sweep_analyze_json", "dh_start", "dh_feed",
            "placement_sites", "load_recorded", "export_take")


def api(fn, args):
    """Return (content_type, bytes)."""
    if fn in ("sweep_circuits_py", "dh_next_circuit"):
        built = getattr(bridge, fn)(*args)
        CURRENT[:] = built if isinstance(built, list) else [built]
        return "application/json", json.dumps(len(CURRENT)).encode()
    if fn == "run":
        shots = int(args[0])
        return "application/json", json.dumps(aer_run(list(CURRENT), shots, seed=None)).encode()
    if fn in TEXT_FNS:
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
