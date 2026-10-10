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
    assert '"three": "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js"' in html
    assert '"three/addons/": "https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/"' in html
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
