import json

from qiskit.providers.basic_provider import BasicSimulator

from qec import bridge
from qec.lessons import build_all


def test_build_then_circuits_and_analyze_round_trip():
    meta_json = bridge.build(json.dumps({"shots": 50, "inject": None}))
    meta = json.loads(meta_json)
    circuits = bridge.circuits()
    assert len(meta) == len(circuits) == 16
    assert [m["nqubits"] for m in meta] == [qc.num_qubits for qc in circuits]
    # fake counts: every shot reads all zeros
    counts = [{"0" * qc.num_clbits: 50} for qc in circuits]
    res = json.loads(bridge.analyze(meta_json, json.dumps(counts)))
    assert len(res) == 16
    assert res[0]["discard_fraction"] == 0 and res[0]["p1_corrected"] == 0 and res[0]["n_clean"] == 50


def test_recorded_take_is_loadable_and_consistent():
    take = json.loads(bridge.load_recorded())
    assert set(take) >= {"source", "backend", "date", "settings", "meta", "counts"}
    assert len(take["meta"]) == len(take["counts"]) == 16
    assert build_all(take["settings"])[1] == take["meta"]
    assert all(sum(c.values()) == take["settings"]["shots"] for c in take["counts"])
    res = json.loads(bridge.analyze(json.dumps(take["meta"]), json.dumps(take["counts"])))
    assert len(res) == 16
    assert all(r["n_clean"] + r["n_corrected"] + r["n_discarded"] == r["n_total"] for r in res)


def test_draw_returns_matplotlib_figure():
    bridge.build(json.dumps({"shots": 50, "inject": None}))
    fig = bridge.draw(0)
    assert fig.__class__.__name__ == "Figure"


def test_circuits_subset_returns_requested_circuits_in_order():
    bridge.build(json.dumps({"shots": 50, "inject": None}))
    sub = bridge.circuits_subset("[0, 12]")
    assert [qc.num_qubits for qc in sub] == [9, 27]


def test_run_subset_one_circuit_is_submitted_bare():
    """IonQ takes only single-circuit jobs, so one index sends the circuit itself, not a list of one."""
    class Spy:
        def run(self, circuits, shots):
            self.sent = circuits
            return BasicSimulator().run(circuits, shots=shots)
    meta = json.loads(bridge.build(json.dumps({"shots": 50, "inject": None})))
    spy = Spy()
    counts = json.loads(bridge.run_subset(spy, json.dumps([0]), 10))
    assert not isinstance(spy.sent, list)
    assert len(counts) == 1 and sum(counts[0].values()) == 10 and len(next(iter(counts[0]))) == meta[0]["nqubits"]


def test_run_subset_counts_in_order():
    meta = json.loads(bridge.build(json.dumps({"shots": 50, "inject": None})))
    indices = [i for i, m in enumerate(meta) if m["lesson"] == 1]
    assert indices
    counts = json.loads(bridge.run_subset(BasicSimulator(), json.dumps(indices), 20))
    assert len(counts) == len(indices)
    assert all(sum(c.values()) == 20 for c in counts)
    assert [len(next(iter(c))) for c in counts] == [meta[i]["nqubits"] for i in indices]
