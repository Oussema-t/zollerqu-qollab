import json
from math import comb
from pathlib import Path

import numpy as np

from qopt import bridge
from qopt.cost import cost_table

ROOT = Path(__file__).resolve().parents[1]
TAKE = json.loads((ROOT / "data" / "recorded_local.json").read_text(encoding="utf-8"))
DATA = json.loads((ROOT / "data" / "champaign.json").read_text(encoding="utf-8"))


def test_take_shape():
    assert TAKE["source"] == "local-aer" and TAKE["params"] == {"m": 8}
    assert len(TAKE["sweep_counts"]) == sum(len(s["js"]) for s in TAKE["plan"])
    assert TAKE["dh"]["seed"] == 7 and TAKE["dh"]["rounds"]


def test_sweep_matches_theory():
    bridge.load_data(json.dumps(DATA))
    out = json.loads(bridge.sweep_analyze_json(json.dumps(TAKE["params"]), json.dumps(TAKE["plan"]),
                                               json.dumps(TAKE["sweep_counts"])))
    # 1000 shots per point: every point within 4 standard errors of the theory value (plus 0.005)
    shots = sum(TAKE["sweep_counts"][0].values())
    for s in out:
        for p in s["points"]:
            se = (p["theory"] * (1 - p["theory"]) / shots) ** 0.5
            assert abs(p["measured"] - p["theory"]) <= 4 * se + 0.005, (s["M"], p)
        assert s["max_dev"] < 0.06


def test_take_costs_are_eighths():
    res = TAKE["dh"]["result"]
    assert res["best_cost"] * 8 == round(res["best_cost"] * 8)
    assert all(s["T"] * 8 == round(s["T"] * 8) for s in TAKE["plan"])


def test_dh_result_is_feasible_and_not_below_brute_force():
    table = cost_table(DATA, {}, TAKE["params"]["m"])
    exact = np.nanmin(table[np.isfinite(table)])
    best = TAKE["dh"]["result"]["best_cost"]
    assert best is not None and np.isfinite(best) and best >= exact - 1e-9
    print(f"recorded dh best_cost {best:.6f} vs brute force {exact:.6f}: exact={abs(best - exact) <= 1e-9}")


def test_dh_one_shot_per_round():
    assert TAKE["dh"]["dh_shots"] == 1
    res = TAKE["dh"]["result"]
    assert res["oracle_calls"] == 1 + res["grover_iterations"] + len(TAKE["dh"]["rounds"])
    assert res["N"] == 70


def test_scaling_study():
    study = TAKE["study"]
    ms = [e["m"] for e in study]
    assert ms == [6, 8, 10]
    for e in study:
        assert set(e) == {"m", "N", "seeds", "calls_mean", "calls_min", "calls_max", "grover_mean",
                          "rounds_mean", "exact_fraction"}
        assert e["N"] == comb(e["m"], 4)  # 15, 70, 210
        assert e["calls_min"] <= e["calls_mean"] <= e["calls_max"]
        assert e["exact_fraction"] >= 0.5
        if e["m"] >= 10:
            assert e["calls_mean"] < e["N"]
