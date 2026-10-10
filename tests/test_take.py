import json
from pathlib import Path

import numpy as np

from qopt import bridge
from qopt.cost import cost_table

ROOT = Path(__file__).resolve().parents[1]
TAKE = json.loads((ROOT / "data" / "recorded_local.json").read_text(encoding="utf-8"))
DATA = json.loads((ROOT / "data" / "champaign.json").read_text(encoding="utf-8"))


def test_take_shape():
    assert TAKE["source"] == "local-aer" and TAKE["params"] == {"m": 10}
    assert len(TAKE["sweep_counts"]) == sum(len(s["js"]) for s in TAKE["plan"])
    assert TAKE["dh"]["seed"] == 7 and TAKE["dh"]["rounds"]


def test_sweep_matches_theory():
    bridge.load_data(json.dumps(DATA))
    out = json.loads(bridge.sweep_analyze_json(json.dumps(TAKE["params"]), json.dumps(TAKE["plan"]),
                                               json.dumps(TAKE["sweep_counts"])))
    assert all(s["max_dev"] < 0.05 for s in out)


def test_dh_result_is_brute_force_optimum():
    table = cost_table(DATA, {}, TAKE["params"]["m"])
    assert abs(TAKE["dh"]["result"]["best_cost"] - np.nanmin(table[np.isfinite(table)])) < 1e-6
