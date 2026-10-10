# Dev harness and demo video plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the finished page outside Qollab, with the real Python core behind it, so it can be polished and a demo video recorded without a Qollab login.

**Architecture:** A local Python HTTP server (`dev/serve.py`) serves the repo files, wraps the body-only `index.html` in a full document with an import map, and exposes the six `qec.bridge` entry points plus an Aer run endpoint under `/api/`. A JS shim (`dev/bridge_shim.js`) replaces the Qollab `qec.bridge` module and the injected `backend` global with fetch calls to that server. `main.js`, `index.html`, `main.css`, `ui/*` and `qec/*` are not modified. A Playwright script (`dev/record_demo.py`) drives the page with on-screen captions and writes an mp4.

**Tech stack:** Python 3.12 `http.server`, qiskit-aer 0.17 (already in `requirements-dev.txt:2`), matplotlib (via `qiskit[visualization]`, `requirements.txt:1`), Playwright 1.61 for Python (installed, chromium-1234 and ffmpeg-1011 present under `%LOCALAPPDATA%\ms-playwright`).

---

## Facts this plan rests on

| Fact | Source |
|---|---|
| `main.js` imports the bridge as a bare specifier `'qec.bridge'` and expects `build, circuits, analyze, draw, load_recorded, export_take` | `main.js:1,67,69,144,128,178,196` |
| Every bridge call is `bridge.<fn>.callPromising(...args)` with string/number args | `main.js:67,69,128,130,144,178,196` |
| `backend` is a global; `backend.name` may be a string or function | `main.js:40` |
| `backend.run(circuits, {shots})` returns a job; `job.status()` returns an object whose `.name` is read; terminal names contain done/completed/error/failed/cancel | `main.js:72,78,84,86` |
| `job.result()` returns an object with `get_counts(i)`; a `toJs()` method is optional | `main.js:102,105,10-14` |
| `draw` result is passed straight to `URL.createObjectURL`, so a `Blob` works | `main.js:182` |
| Bridge signatures: `build(settings_json)->str`, `circuits()->list`, `analyze(meta_json, counts_json)->str`, `draw(index)->Figure`, `load_recorded()->str`, `export_take(backend_name, settings_json, meta_json, counts_json, date)->str`; `export_take` swallows the missing `js` module | `qec/bridge.py:13-56` |
| Built circuits live in `_LAST["circuits"]` after `build` | `qec/bridge.py:10,15,21` |
| `index.html` is body-only (starts with `<div class="app">`) | `index.html:1` |
| Stand-in noise model: depolarizing 5e-4 on 1q gates, 6e-3 on `cx`, `optimization_level=0` | `tools/record_take.py:24-28` |
| Timing measured today: noisy Aer, 16 circuits x 500 shots = 56 s; 200 shots ~22 s; `draw` + SVG = 1.3 s, 47 KB | measured in this session |
| Shots input: min 100, max 2000, step 100 | `index.html:12` |
| Element ids used by the recorder: `shots`, `inject-pauli`, `inject-qubit`, `submit`, `load-recorded`, `status`, `tab-1..4`, `circuit-details`, `panel` | `index.html:12-47` |
| CSS has a dark default and a light override | `main.css:1-7` |

## Decisions

- Default demo shots: 200 (22 s wait, shown with a caption). 500 is too slow for a video.
- Harness uses the same noise model as `tools/record_take.py` so flagged fractions are non-zero and the page looks like a noisy device. Constant at the top of `serve.py`, no flag.
- Job status in the shim is `RUNNING` until the fetch resolves, then `DONE`, so the polling loop in `main.js:91-100` is exercised.
- Video: 1280x720, dark colour scheme, captions injected as a fixed bottom bar by the recorder (not part of the shipped page). Honesty caption at the start: "Recorded against a local Aer stand-in. The same page runs unchanged on Qollab."
- `dev/` is not uploaded to Qollab. README gets a two-line "Run locally" note.
- `playwright>=1.40` added to `requirements-dev.txt`.

---

### Task 1: Dev server

**Files:**
- Create: `dev/serve.py`
- Create: `dev/__init__.py` (empty)
- Test: `tests/test_dev_server.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_dev_server.py
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from dev.serve import Handler


@pytest.fixture(scope="module")
def base():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def post(base, fn, *args):
    req = urllib.request.Request(f"{base}/api/{fn}", data=json.dumps({"args": list(args)}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.headers.get_content_type(), r.read()


def test_root_wraps_index_with_import_map(base):
    html = urllib.request.urlopen(base + "/").read().decode()
    assert html.startswith("<!doctype html>")
    assert '"qec.bridge": "/dev/bridge_shim.js"' in html
    assert '<div class="app" id="app">' in html
    assert '<script type="module" src="/main.js">' in html
    assert 'href="/main.css"' in html


def test_static_files_served(base):
    assert urllib.request.urlopen(base + "/ui/patches.js").status == 200
    assert urllib.request.urlopen(base + "/dev/bridge_shim.js").headers.get_content_type() == "text/javascript"


def test_build_run_analyze_round_trip(base):
    _, meta = post(base, "build", json.dumps({"shots": 100, "inject": None}))
    meta = json.loads(meta)
    assert len(meta) == 16
    _, n = post(base, "circuits")
    assert json.loads(n) == 16
    _, counts = post(base, "run", 100)
    counts = json.loads(counts)
    assert len(counts) == 16 and all(sum(c.values()) == 100 for c in counts)
    _, rows = post(base, "analyze", json.dumps(meta), json.dumps(counts))
    rows = json.loads(rows)
    assert len(rows) == 16 and rows[0]["n_total"] == 100


def test_draw_returns_svg(base):
    post(base, "build", json.dumps({"shots": 100, "inject": None}))
    ctype, body = post(base, "draw", 0)
    assert ctype == "image/svg+xml"
    assert body.lstrip().startswith(b"<?xml") or b"<svg" in body[:500]


def test_load_recorded_and_export(base):
    _, take = post(base, "load_recorded")
    take = json.loads(take)
    assert take["counts"] and take["meta"]
    _, text = post(base, "export_take", "dev", json.dumps(take["settings"]), json.dumps(take["meta"]), json.dumps(take["counts"]), "2026-10-10")
    assert json.loads(text)["source"] == "qollab-run"
```

- [ ] **Step 2: Run, expect ImportError** `python -m pytest tests/test_dev_server.py -q`

- [ ] **Step 3: Implement**

```python
# dev/serve.py
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
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qec import bridge

# Same stand-in noise as tools/record_take.py so the page looks like a noisy device.
_noise = NoiseModel()
_noise.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["h", "x", "y", "z", "t", "sdg", "ry"])
_noise.add_all_qubit_quantum_error(depolarizing_error(0.006, 2), ["cx"])
SIM = AerSimulator(noise_model=_noise)
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
    circuits = bridge.circuits()
    result = SIM.run(transpile(circuits, SIM, optimization_level=0), shots=int(shots)).result()
    return [dict(result.get_counts(i)) for i in range(len(circuits))]


def api(fn, args):
    """Return (content_type, bytes)."""
    if fn == "run":
        return "application/json", json.dumps(run(*args)).encode()
    if fn == "circuits":
        return "application/json", json.dumps(len(bridge.circuits())).encode()
    if fn == "draw":
        buf = io.BytesIO()
        bridge.draw(*args).savefig(buf, format="svg", bbox_inches="tight")
        return "image/svg+xml", buf.getvalue()
    if fn in ("build", "analyze", "load_recorded", "export_take"):
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
```

- [ ] **Step 4: Run tests, expect 5 passed** (the shim file from Task 2 must exist for `test_static_files_served`; create it first or run Task 2 before this step)
- [ ] **Step 5: Commit** `git add dev/__init__.py dev/serve.py tests/test_dev_server.py && git commit -m "Add local dev harness server"`

### Task 2: Bridge shim

**Files:**
- Create: `dev/bridge_shim.js`

- [ ] **Step 1: Implement** (no unit test; exercised by the browser smoke test in Task 3)

```js
// dev/bridge_shim.js
// Stand-in for the Qollab module 'qec.bridge' and the injected 'backend' global. Every call goes to dev/serve.py.
async function call(fn, args, as = 'text') {
  const r = await fetch(`/api/${fn}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ args }) });
  if (!r.ok) throw new Error(await r.text());
  return as === 'blob' ? r.blob() : as === 'json' ? r.json() : r.text();
}
const fn = (name, as) => ({ callPromising: (...args) => call(name, args, as) });

export const build = fn('build');
export const circuits = fn('circuits', 'json');
export const analyze = fn('analyze');
export const draw = fn('draw', 'blob');          // Blob of SVG, so URL.createObjectURL works as on Qollab
export const load_recorded = fn('load_recorded');
export const export_take = fn('export_take');

globalThis.backend = {
  name: 'local dev harness (Aer, depolarizing stand-in)',
  async run(_circuits, { shots }) {
    let counts = null, error = null;
    const done = call('run', [shots], 'json').then((c) => { counts = c; }, (e) => { error = e; });
    return {
      async status() { return { name: error ? 'ERROR' : counts ? 'DONE' : 'RUNNING' }; },
      async result() { await done; if (error) throw error; return { get_counts: async (i) => counts[i] }; },
    };
  },
};
```

- [ ] **Step 2: `node --check dev/bridge_shim.js`**
- [ ] **Step 3: Commit** `git add dev/bridge_shim.js && git commit -m "Add bridge shim for the dev harness"`

### Task 3: Browser smoke test and visual review

- [ ] **Step 1:** `python dev/serve.py 8765` in the background; open `http://127.0.0.1:8765/` in the in-app browser.
- [ ] **Step 2:** Load recorded take: tabs appear, lesson 1 shows the injected X on q1 (top cap and ZZZZ ancilla at 100 percent).
- [ ] **Step 3:** Submit 200 shots, no injection: status counts up, then all four lessons render; open "Show the physical circuit" and confirm an SVG appears.
- [ ] **Step 4:** Screenshot each lesson at 1280 wide and at 400 wide; list visual defects. Any fix that changes `main.css`, `ui/*` or `index.html` is listed for Tushar before it is made (cosmetic fixes under 10 lines may go straight in and are listed in the commit message).

### Task 4: Demo recorder

**Files:**
- Create: `dev/record_demo.py`
- Modify: `requirements-dev.txt` (add `playwright>=1.40`)
- Modify: `README.md` (add "Run locally" and "Demo video" notes under the files section)

- [ ] **Step 1: Implement**

```python
# dev/record_demo.py
"""Record the demo video. Run from the project root with dev/serve.py already running:
    python dev/record_demo.py [--url http://127.0.0.1:8765/] [--shots 200] [--out dev/out]
Writes dev/out/demo.webm and dev/out/demo.mp4 (mp4 via Playwright's bundled ffmpeg).
"""
import argparse
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CAPTION_CSS = """
#demo-cap { position: fixed; left: 0; right: 0; bottom: 0; padding: 14px 24px; background: rgba(10,12,16,.92);
  color: #fff; font: 20px/1.4 system-ui, sans-serif; z-index: 9999; border-top: 2px solid #4f9cff; }
"""

STEPS = [
    # (caption, action name, seconds to hold after the action)
    ("Logical teleportation on a tiny surface code. Three logical qubits, each a 2x2 patch of the [[4,1,2]] code.", None, 4),
    ("Recorded against a local Aer stand-in with depolarising noise. The same page runs unchanged on Qollab.", None, 4),
    ("One job carries all 16 circuits. We submit with 200 shots per circuit.", "submit", 3),
    ("While it runs: the page polls the job and shows progress. Noise-model simulators run one shot at a time.", "wait_results", 2),
    ("Lesson 1. One patch, no injected error. Face and caps light up with the rate each check fired.", "tab-1", 6),
    ("Hover a logical edge: Z_L is the left edge, X_L the top edge.", "hover-edge", 4),
    ("Lesson 2. A transversal CNOT between two patches is a logical CNOT. Correlations survive post-selection.", "tab-2", 7),
    ("Lesson 3. Teleportation of a logical state through a logical Bell pair, Pauli frame applied in post-processing.", "tab-3", 7),
    ("Lesson 4. A T gate by teleporting through a magic state. Kept shots are those with the right measurement outcome.", "tab-4", 7),
    ("Every lesson shows the same experiment on bare physical qubits, so you can see what the code buys.", None, 4),
    ("The physical circuit behind each lesson is one click away.", "circuit", 5),
    ("Now inject an X error on data qubit 1 and resubmit.", "inject", 3),
    ("Lesson 1: the top Z check and the ZZZZ ancilla fire every time. Distance 2 detects every single error.", "wait_lesson1", 7),
    ("Team ZollerQu. This effort is supported by Qollab & IonQ.", None, 4),
]


def caption(page, text):
    page.evaluate("""(t) => { let c = document.getElementById('demo-cap');
        if (!c) { c = document.createElement('div'); c.id = 'demo-cap'; document.body.appendChild(c); }
        c.textContent = t; }""", text)


def wait_showing(page, timeout=180_000):
    page.wait_for_function("() => /^Showing /.test(document.getElementById('status').textContent)", timeout=timeout)


def act(page, name, shots):
    if name == "submit":
        page.fill("#shots", str(shots))
        page.click("#submit")
    elif name == "wait_results":
        wait_showing(page)
    elif name.startswith("tab-"):
        page.click("#" + name)
        page.locator("#panel").scroll_into_view_if_needed()
    elif name == "hover-edge":
        page.locator("#patches line[tabindex]").first.hover()
    elif name == "circuit":
        page.locator("#circuit-details summary").click()
        page.wait_for_selector("#circuit-img img", timeout=30_000)
        page.locator("#circuit-img").scroll_into_view_if_needed()
    elif name == "inject":
        page.locator("#setup").evaluate("d => d.open = true")
        page.select_option("#inject-pauli", "X")
        page.select_option("#inject-qubit", "1")
        page.click("#submit")
    elif name == "wait_lesson1":
        wait_showing(page)
        page.click("#tab-1")
        page.locator("#patches").scroll_into_view_if_needed()


def ffmpeg():
    for p in glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright", "ffmpeg-*", "ffmpeg-win64.exe")):
        return p
    return shutil.which("ffmpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--out", default="dev/out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, color_scheme="dark",
                                  record_video_dir=str(out), record_video_size={"width": 1280, "height": 720})
        page = ctx.new_page()
        page.goto(a.url)
        page.wait_for_selector("#submit")
        page.add_style_tag(content=CAPTION_CSS)
        for text, action, hold in STEPS:
            caption(page, text)
            if action:
                act(page, action, a.shots)
            time.sleep(hold)
        video = page.video
        ctx.close()
        webm = Path(video.path())
        browser.close()
    final = out / "demo.webm"
    shutil.move(str(webm), final)
    ff = ffmpeg()
    if ff:
        subprocess.run([ff, "-y", "-i", str(final), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "demo.mp4")], check=True)
    print("wrote", final, "and demo.mp4" if ff else "(no ffmpeg found, webm only)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run with the server up; open `dev/out/demo.mp4`, check length (target 2 to 3 min), captions readable, no clipped panels. Adjust `STEPS` holds, never the shipped page, from this script.
- [ ] **Step 3:** Add `dev/out/` to `.gitignore`.
- [ ] **Step 4:** README: under files, add `dev/` (local harness, not uploaded) and a three-line "Run locally" (`pip install -r requirements-dev.txt`, `python dev/serve.py`, open the URL) and "Demo video" (`python dev/record_demo.py`).
- [ ] **Step 5:** Commit `git add dev/record_demo.py requirements-dev.txt README.md .gitignore && git commit -m "Add scripted demo recorder"`

### Task 5: Review

One fresh reviewer (Fable) reads the diff plus a change summary, checks it against the facts table, ten-minute cap.

## Out of scope

- Changing `main.js`, `index.html`, `main.css`, `ui/*`, `qec/*` (except fixes found in Task 3, which are listed first).
- Recording on Qollab itself. `--url` exists so the same script can be pointed at the Qollab page later if the page is reachable without an iframe; if it is in an iframe the recorder needs a frame locator, which is a plan change.
- Voice-over.
