import json

from qec import bridge
from qec.lessons import build_all


def test_build_then_circuits_and_analyze_round_trip():
    meta_json = bridge.build(json.dumps({"shots": 50, "inject": None}))
    meta = json.loads(meta_json)
    circuits = bridge.circuits()
    assert len(meta) == len(circuits) == 16
    # fake counts: every shot reads all zeros
    counts = [{"0" * qc.num_clbits: 50} for qc in circuits]
    res = json.loads(bridge.analyze(meta_json, json.dumps(counts)))
    assert len(res) == 16
    assert res[0]["flagged_fraction"] == 0 and res[0]["p1_accepted"] == 0


def test_recorded_take_is_loadable_and_consistent():
    take = json.loads(bridge.load_recorded())
    assert set(take) >= {"source", "backend", "date", "settings", "meta", "counts"}
    assert len(take["meta"]) == len(take["counts"]) == 16
    assert build_all(take["settings"])[1] == take["meta"]
    res = json.loads(bridge.analyze(json.dumps(take["meta"]), json.dumps(take["counts"])))
    assert len(res) == 16


def test_draw_returns_matplotlib_figure():
    bridge.build(json.dumps({"shots": 50, "inject": None}))
    fig = bridge.draw(0)
    assert fig.__class__.__name__ == "Figure"
