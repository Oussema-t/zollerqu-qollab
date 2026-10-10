import json

import numpy as np
from qiskit import QuantumCircuit

from qopt import bridge
from qopt.aer_run import aer_run
from qopt.classical import brute_force
from qopt.cost import cost_table
from tests.conftest import random_data

PARAMS = {"m": 6, "k": 3}


def setup_module():
    bridge.load_data(json.dumps(random_data(30, 12, seed=11)))


def test_load_data_returns_json():
    out = json.loads(bridge.load_data(json.dumps(random_data(30, 12, seed=11))))
    assert out == {"blocks": 30, "sites": 12}


def test_cost_table_json():
    out = json.loads(bridge.cost_table_json(json.dumps(PARAMS)))
    assert set(out) == {"m", "k", "n_feasible", "best_index", "best_cost", "best_sites"}
    assert out["m"] == 6 and out["k"] == 3 and out["n_feasible"] == 20
    table = cost_table(random_data(30, 12, seed=11), PARAMS, 6)
    assert out["best_index"] == brute_force(table)[0]
    assert out["best_sites"] == [i for i in range(6) if out["best_index"] >> i & 1]


def test_cost_table_json_default_m_is_10():
    out = json.loads(bridge.cost_table_json(json.dumps({"k": 2})))
    assert out["m"] == 10 and out["n_feasible"] == 45


def test_classical():
    out = json.loads(bridge.classical(json.dumps(PARAMS)))
    assert set(out) >= {"brute_force", "greedy", "local_search", "milp"}
    assert set(out["brute_force"]) >= {"index", "cost", "sites", "evaluations"}
    assert len(out["greedy"]["sites"]) == 3
    assert out["local_search"]["best_cost"] >= out["brute_force"]["cost"]
    assert out["milp"] is None or out["milp"]["scope"] == "travel part only"


def test_sweep_round_trip():
    plan = json.loads(bridge.sweep(json.dumps(PARAMS)))
    assert [s["M"] for s in plan] == [1, 4, 16]
    circuits = bridge.sweep_circuits_py(json.dumps(PARAMS))
    assert all(isinstance(c, QuantumCircuit) for c in circuits)
    assert len(circuits) == sum(len(s["js"]) for s in plan)
    counts = aer_run(circuits, 2000, seed=3)
    out = json.loads(bridge.sweep_analyze_json(json.dumps(PARAMS), json.dumps(plan), json.dumps(counts)))
    assert [s["M"] for s in out] == [1, 4, 16]
    assert all(s["max_dev"] < 0.05 for s in out)


def test_dh_loop():
    start = json.loads(bridge.dh_start(json.dumps(PARAMS), 5))
    assert set(start) >= {"N", "T", "best_index"}
    done, rounds, seed = False, 0, 0
    while not done:
        circuit = bridge.dh_next_circuit()
        assert isinstance(circuit, QuantumCircuit)
        out = json.loads(bridge.dh_feed(json.dumps(aer_run([circuit], 100, seed=seed)[0])))
        seed += 1
        rounds += 1
        assert set(out["round"]) >= {"T", "j", "best_sample_cost", "improved", "oracle_calls_cum"}
        done = out["done"]
        assert rounds <= 30
    res = out["result"]
    table = cost_table(random_data(30, 12, seed=11), PARAMS, 6)
    assert res["best_index"] == brute_force(table)[0]
    assert res["converged"] is True and len(res["rounds"]) == rounds
    assert res["best_sites"] == [i for i in range(6) if res["best_index"] >> i & 1]


def test_placement_sites():
    assert json.loads(bridge.placement_sites(0b100101)) == [0, 2, 5]
