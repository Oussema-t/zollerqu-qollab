from qiskit import transpile
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
    assert isinstance(out["blocks"], list) and isinstance(out["sites"], list)
    assert len(out["blocks"]) == 30 and len(out["sites"]) == 12


def test_load_data_reads_the_shipped_file_when_no_text_is_given():
    out = json.loads(bridge.load_data())
    assert len(out["blocks"]) == 158 and len(out["sites"]) == 12
    assert bridge._STATE["data"] == out
    bridge.load_data(json.dumps(random_data(30, 12, seed=11)))


def test_page_does_not_fetch_project_files():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "main.js").read_text(encoding="utf-8")
    for needle in ("fetch('data/", 'fetch("data/', "fetch(`data/"):
        assert needle not in js


def test_cost_table_json():
    out = json.loads(bridge.cost_table_json(json.dumps(PARAMS)))
    assert set(out) == {"m", "k", "N", "n_feasible", "best_index", "best_cost", "best_sites", "travel_constant"}
    assert out["travel_constant"] * 8 == round(out["travel_constant"] * 8)  # reported to the nearest 1/8
    assert out["best_cost"] * 8 == round(out["best_cost"] * 8)
    assert out["m"] == 6 and out["k"] == 3 and out["n_feasible"] == out["N"] == 20
    table = cost_table(random_data(30, 12, seed=11), PARAMS, 6)
    assert out["best_index"] == brute_force(table)[0]
    assert out["best_sites"] == [i for i in range(6) if out["best_index"] >> i & 1]


def test_cost_table_json_default_m_is_8():
    out = json.loads(bridge.cost_table_json(json.dumps({"k": 2})))
    assert out["m"] == 8 and out["n_feasible"] == out["N"] == 28


def test_classical():
    out = json.loads(bridge.classical(json.dumps(PARAMS)))
    assert set(out) == {"brute_force", "greedy", "local_search"}
    assert set(out["brute_force"]) >= {"index", "cost", "sites", "evaluations", "unit"}
    assert out["brute_force"]["evaluations"] == 20 and out["brute_force"]["unit"] == "placements scored"
    assert len(out["greedy"]["sites"]) == 3
    assert out["greedy"]["evaluations"] == 6 + 5 + 4 and out["greedy"]["unit"] == "partial placements scored"
    assert out["local_search"]["best_cost"] >= out["brute_force"]["cost"]
    assert out["local_search"]["unit"] == "placements scored" and out["local_search"]["evaluations"] > 0


def test_sweep_round_trip():
    plan = json.loads(bridge.sweep(json.dumps(PARAMS)))
    assert [s["M"] for s in plan] == [1, 4, 16]
    assert all(s["N"] == 20 for s in plan)
    circuits = bridge.sweep_circuits_py(json.dumps(PARAMS))
    assert all(isinstance(c, QuantumCircuit) for c in circuits)
    assert len(circuits) == sum(len(s["js"]) for s in plan)
    counts = aer_run(circuits, 2000, seed=3)
    out = json.loads(bridge.sweep_analyze_json(json.dumps(PARAMS), json.dumps(plan), json.dumps(counts)))
    assert [s["M"] for s in out] == [1, 4, 16]
    assert all(s["N"] == 20 for s in out)
    assert all(s["max_dev"] < 0.05 for s in out)


def test_live_run_is_one_circuit_that_finds_the_optimum():
    """The live run is the loop's last search: one circuit, threshold marking only the optimum (M = 1),
    j chosen to maximise the chance of measuring it (j = 3 for M = 1 of 20). Measured most often: the exact optimum."""
    plan = json.loads(bridge.sweep(json.dumps(PARAMS), True))
    assert [(s["M"], s["js"]) for s in plan] == [(1, [3])]
    circuits = bridge.sweep_circuits_py(json.dumps(PARAMS), True)
    assert len(circuits) == 1
    counts = aer_run(circuits, 2000, seed=3)
    out = json.loads(bridge.live_result(json.dumps(PARAMS), json.dumps(counts)))
    exact = json.loads(bridge.cost_table_json(json.dumps(PARAMS)))
    assert out["cost"] == exact["best_cost"] and out["exact"] is True
    assert out["share"] > 0.9 and out["theory"] > 0.99 and out["j"] == 3


def test_ionq_gets_high_level_gates():
    """IonQ takes controlled phases and Toffolis as single gates; sending them undecomposed keeps the job small."""
    from qiskit.providers.basic_provider import BasicSimulator

    class Spy:
        name = "ionq_simulator"

        def run(self, circuits, shots):
            self.sent = circuits
            return BasicSimulator().run(transpile(circuits, basis_gates=["rz", "sx", "x", "cx"]), shots=shots)
    spy = Spy()
    bridge.run_sweep(spy, json.dumps(PARAMS), 10, True)
    ops = spy.sent.count_ops()
    assert "cp" in ops and "ccx" not in ops and all(len(i.qubits) <= 2 for i in spy.sent.data if i.operation.name != "measure")


def test_run_sweep_one_circuit_is_submitted_bare():
    """IonQ rejects multi-circuit jobs, so a one-circuit sweep sends the circuit itself, not a list of one."""
    from qiskit.providers.basic_provider import BasicSimulator

    class Spy:
        def run(self, circuits, shots):
            self.sent = circuits
            return BasicSimulator().run(circuits, shots=shots)
    spy = Spy()
    counts = json.loads(bridge.run_sweep(spy, json.dumps(PARAMS), 10, True))
    assert isinstance(spy.sent, QuantumCircuit)
    assert len(counts) == 1 and sum(counts[0].values()) == 10 and len(next(iter(counts[0]))) == 6


def test_page_does_not_run_the_loop_live():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "main.js").read_text(encoding="utf-8")
    for needle in ("liveDh", "dh_run_round", "dh_start", "dh_feed", "DH_SHOTS"):
        assert needle not in js


def test_run_sweep_counts_per_circuit(monkeypatch):
    from functools import partial

    from qiskit.providers.basic_provider import BasicSimulator
    from qopt import search
    # two circuits per series (j = 0, 1): BasicSimulator needs minutes for the full 17-qubit sweep
    monkeypatch.setattr(search, "sweep_plan", partial(search.sweep_plan, max_circuits=2))
    counts = json.loads(bridge.run_sweep(BasicSimulator(), json.dumps(PARAMS), 50))
    assert len(counts) == len(bridge.sweep_circuits_py(json.dumps(PARAMS)))
    assert all(isinstance(c, dict) and sum(c.values()) == 50 for c in counts)
    assert all(len(key) == 6 for c in counts for key in c)  # only the m site qubits are measured


def test_dh_run_round_counts():
    from qiskit.providers.basic_provider import BasicSimulator
    bridge.dh_start(json.dumps(PARAMS), 5)
    c = json.loads(bridge.dh_run_round(BasicSimulator(), 1))
    assert isinstance(c, dict) and sum(c.values()) == 1
    out = json.loads(bridge.dh_feed(json.dumps(c)))
    assert "round" in out


def test_dh_loop():
    start = json.loads(bridge.dh_start(json.dumps(PARAMS), 5))
    assert set(start) >= {"N", "T", "best_index"}
    assert start["N"] == 20
    done, rounds, seed = False, 0, 0
    while not done:
        circuit = bridge.dh_next_circuit()
        assert isinstance(circuit, QuantumCircuit)
        out = json.loads(bridge.dh_feed(json.dumps(aer_run([circuit], 100, seed=seed)[0])))
        seed += 1
        rounds += 1
        assert set(out["round"]) >= {"T", "j", "best_sample_cost", "improved", "oracle_calls_cum"}
        done = out["done"]
        assert rounds <= 120
    res = out["result"]
    assert res["grover_iterations"] == sum(r["j"] for r in res["rounds"])
    assert res["oracle_calls"] == 1 + res["grover_iterations"] + rounds
    assert res["N"] == 20
    table = cost_table(random_data(30, 12, seed=11), PARAMS, 6)
    assert res["best_index"] == brute_force(table)[0]
    assert res["converged"] is True and len(res["rounds"]) == rounds
    assert res["best_sites"] == [i for i in range(6) if res["best_index"] >> i & 1]


def test_placement_sites():
    assert json.loads(bridge.placement_sites(0b100101)) == [0, 2, 5]


def test_export_take_round_trips():
    plan, counts, dh = [{"M": 1, "T": 2.0, "js": [0]}], [{"00": 5}], {"seed": 7, "rounds": [], "result": {}}
    text = bridge.export_take("Aer", json.dumps(PARAMS), json.dumps(plan), json.dumps(counts), json.dumps(dh), "2026-10-10")
    assert json.loads(text) == {"source": "qollab-run", "backend": "Aer", "date": "2026-10-10",
                                "params": PARAMS, "plan": plan, "sweep_counts": counts, "dh": dh}


def test_load_recorded_parses():
    take = json.loads(bridge.load_recorded())
    assert take["source"] == "local-aer" and take["params"] == {"m": 8}


def test_load_recorded_returns_given_text_unchanged():
    text = '{"source": "qollab-run",  "x": [1, 2]}'
    assert bridge.load_recorded(text) == text


def test_load_recorded_rejects_invalid_text():
    import pytest
    with pytest.raises(json.JSONDecodeError):
        bridge.load_recorded("not json")
