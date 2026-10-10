"""Entry points called from main.js. JSON strings in and out; circuits and figures stay proxies."""
import json
from pathlib import Path

from qec.analyze import analyze_all
from qec.lessons import build_all
from qec.spacetime import layout

RECORDED = "data/recorded_aria1.json"   # literal path so the Qollab runtime loads the file

_LAST = {"circuits": [], "meta": [], "figures": {}, "layouts": {}}


def build(settings_json):
    circuits, meta = build_all(json.loads(settings_json))
    _LAST["circuits"], _LAST["meta"] = circuits, meta
    _LAST["figures"] = {}
    _LAST["layouts"] = {}
    return json.dumps(meta)


def circuits():
    return _LAST["circuits"]


def analyze(meta_json, counts_json):
    return json.dumps(analyze_all(json.loads(meta_json), json.loads(counts_json)))


def draw(index):
    index = int(index)
    if index not in _LAST["figures"]:
        qc = _LAST["circuits"][index]
        _LAST["figures"][index] = qc.draw(output="mpl", fold=-1, scale=0.6)
    return _LAST["figures"][index]


def spacetime(index):
    index = int(index)
    if index not in _LAST["layouts"]:
        _LAST["layouts"][index] = json.dumps(layout(_LAST["circuits"][index], _LAST["meta"][index]))
    return _LAST["layouts"][index]


def load_recorded():
    path = Path(RECORDED)
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / RECORDED
    with open(path, encoding="utf-8") as f:
        return f.read()


def export_take(backend_name, settings_json, meta_json, counts_json, date):
    """Hand a JSON file to the Qollab console (copy button) so a real run can be saved as data/recorded_aria1.json."""
    take = {
        "source": "qollab-run", "backend": backend_name, "date": date,
        "settings": json.loads(settings_json), "meta": json.loads(meta_json), "counts": json.loads(counts_json),
    }
    text = json.dumps(take)
    try:
        from js import sendFile  # only exists inside the Qollab runtime
        sendFile("recorded_take.json", "application/json", text)
    except ImportError:
        pass
    return text
