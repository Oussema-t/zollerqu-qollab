from itertools import combinations
from math import comb

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector

from qopt.aer_run import aer_run
from qopt.classical import brute_force
from qopt.cost import feasible_mask, model_cost_int, model_table
from qopt.search import (BASIS, accumulator_bits, dh_init, dicke_circuit, feasible_count, find_min, grover_circuit,
                         index_from_key, marked_count, oracle, success_theory, sweep_analyze, sweep_circuits,
                         sweep_plan, to_basis)


def random_model(m, k, seed, a_range=20, b_range=12, distinct=True):
    """Random integer coefficients (eighths); distinct: redraw until all feasible costs differ (M targets exact)."""
    rng = np.random.default_rng(seed)
    while True:
        A = [int(v) for v in rng.integers(-a_range, a_range + 1, m)]
        B = {p: int(v) for p in combinations(range(m), 2) if (v := rng.integers(-2, b_range + 1)) != 0}
        mdl = {"m": m, "k": k, "A": A, "B": B, "scale": 8}
        costs = [model_cost_int(mdl, s) for s in combinations(range(m), k)]
        if not distinct or len(set(costs)) == len(costs):
            return mdl


def _oracle_action(mdl, T, strict, x):
    """Apply the oracle to |x> on the sites with accumulator and AND ancilla at |0>; return the output statevector."""
    o = oracle(mdl, T, strict)
    qc = QuantumCircuit(o.num_qubits)
    for i in range(mdl["m"]):
        if x >> i & 1:
            qc.x(i)
    qc.compose(o, inplace=True)
    return Statevector(qc).data


@pytest.mark.parametrize("strict", [True, False])
def test_oracle_marks_exactly_the_placements_below_threshold(strict):
    m, k = 5, 2
    mdl = random_model(m, k, seed=0)
    table = model_table(mdl)
    T = float(np.sort(table)[3])
    marked = 0
    for x in range(2 ** m):
        out = _oracle_action(mdl, T, strict, x)
        # every basis state, feasible or not, comes back as itself: accumulator and ancilla return to |0>
        assert np.isclose(abs(out[x]), 1.0, atol=1e-9)
        if bin(x).count("1") == k:
            below = table[x] < T if strict else table[x] <= T
            assert np.isclose(out[x].real, -1.0 if below else 1.0, atol=1e-9)
            marked += below
    assert marked == marked_count(table, T, strict)
    assert marked == (3 if strict else 4)


def test_oracle_register_widths_and_gates():
    mdl = random_model(5, 2, seed=1)
    T = float(np.sort(model_table(mdl))[2])
    n = accumulator_bits(mdl, round(T * 8))
    o = oracle(mdl, T)
    assert o.num_qubits == 5 + n + 1
    ops = set(o.count_ops())
    assert "diagonal" not in ops and "unitary" not in ops
    assert ops <= {"h", "p", "cp", "ccx", "z", "swap"}


def test_accumulator_bits_cover_every_feasible_difference():
    mdl = random_model(6, 3, seed=2)
    costs = [model_cost_int(mdl, s) for s in combinations(range(6), 3)]
    for t in (min(costs), max(costs) + 1, 0, 37):
        n = accumulator_bits(mdl, t)
        assert all(-(2 ** (n - 1)) <= c - t <= 2 ** (n - 1) - 1 for c in costs)


def test_circuits_are_built_from_coefficients_not_the_table(monkeypatch):
    import qopt.cost
    import qopt.search

    def forbidden(*args, **kwargs):
        raise AssertionError("circuit construction must not tabulate the placements")

    monkeypatch.setattr(qopt.cost, "model_table", forbidden)
    monkeypatch.setattr(qopt.cost, "cost_table", forbidden)
    monkeypatch.setattr(qopt.search, "model_table", forbidden, raising=False)
    mdl = random_model(6, 3, seed=3)
    qc = grover_circuit(mdl, 0.5, 2, strict=True)
    assert qc.num_clbits == 6


def test_grover_circuit_measures_only_the_sites():
    mdl = random_model(6, 3, seed=4)
    qc = grover_circuit(mdl, 0.0, 1)
    assert qc.num_clbits == 6
    measured = [qc.find_bit(inst.qubits[0]).index for inst in qc.data if inst.operation.name == "measure"]
    assert measured == list(range(6))


@pytest.mark.parametrize("m,k", [(4, 2), (6, 3), (8, 4), (10, 4)])
def test_dicke_circuit_is_uniform_over_weight_k(m, k):
    amps = Statevector(dicke_circuit(m, k)).data
    expected = np.where(feasible_mask(m, k), 1 / np.sqrt(comb(m, k)), 0)
    assert np.allclose(amps, expected, atol=1e-9)


def test_feasible_count():
    assert feasible_count(model_table(random_model(8, 4, seed=0, distinct=False))) == 70


def test_index_from_key_little_endian():
    # site 0 = qubit 0 = last character
    assert index_from_key("0001") == 1
    assert index_from_key("1000") == 8
    assert index_from_key("10 01") == 9


def test_uniform_start_samples_only_feasible():
    m, k = 8, 4
    mdl = random_model(m, k, seed=6, distinct=False)
    counts = aer_run([grover_circuit(mdl, float(np.sort(model_table(mdl))[0]), 0)], 4000, seed=2)[0]
    assert all(len(key) == m and bin(index_from_key(key)).count("1") == k for key in counts)
    assert len(counts) > 60  # spread over the 70 feasible placements


def test_success_probability_matches_theory():
    m, k = 6, 3
    mdl = random_model(m, k, seed=2)
    table = model_table(mdl)
    N = comb(m, k)
    assert feasible_count(table) == N == 20
    finite = np.sort(table[np.isfinite(table)])
    circuits, expected = [], []
    for M in (1, 4):
        T = float(finite[M - 1])
        for j in range(5):
            circuits.append(grover_circuit(mdl, T, j))
            expected.append((T, success_theory(M, N, j)))
    counts_list = aer_run(circuits, 4000, seed=7)
    for counts, (T, theory) in zip(counts_list, expected):
        hits = sum(c for key, c in counts.items() if table[index_from_key(key)] <= T)
        assert abs(hits / 4000 - theory) < 0.03


def test_sweep_plan_and_analyze_round_trip():
    m, k = 6, 3
    mdl = random_model(m, k, seed=3)
    table = model_table(mdl)
    plan = sweep_plan(table, M_targets=(1, 4), max_circuits=4)
    assert [s["M"] for s in plan] == [1, 4]
    assert all(s["N"] == comb(6, 3) for s in plan)  # feasible count, not 2^m
    assert plan[0]["js"] == [0, 1, 2, 3]  # ceil(1.2*pi/4*sqrt(20))=5, capped at max_circuits-1
    assert plan[1]["js"] == [0, 1, 2, 3]  # ceil(1.2*pi/4*sqrt(5))=3
    assert plan[0]["T"] == np.sort(table)[0]
    circuits = sweep_circuits(mdl, plan)
    assert len(circuits) == 8 and all(c.num_clbits == m for c in circuits)
    analysis = sweep_analyze(table, plan, aer_run(circuits, 4000, seed=1))
    assert [s["M"] for s in analysis] == [1, 4]
    for s in analysis:
        assert s["N"] == comb(6, 3)
        assert [p["j"] for p in s["points"]] == [0, 1, 2, 3]
        assert all(p["theory"] == success_theory(s["M"], comb(6, 3), p["j"]) for p in s["points"])
        assert s["max_dev"] == max(abs(p["measured"] - p["theory"]) for p in s["points"])
        assert s["max_dev"] < 0.03


def test_sweep_plan_iteration_count():
    m = 8
    table = model_table(random_model(m, 3, seed=3, a_range=400, b_range=300))
    plan = sweep_plan(table, M_targets=(16,), max_circuits=12)
    # C(8,3) = 56 placements: ceil(1.2 * pi/4 * sqrt(56/16)) = ceil(1.76) = 2
    assert plan[0]["js"] == [0, 1, 2] and plan[0]["M"] == 16


def test_find_min_reaches_brute_force_optimum():
    m, k = 6, 3
    hits = 0
    for seed in range(10):
        mdl = random_model(m, k, seed=100 + seed)
        table = model_table(mdl)

        aer_seeds = iter(range(1000 * seed, 1000 * seed + 1000))

        def run(circuits, shots):
            return aer_run(circuits, shots, seed=next(aer_seeds))

        res = find_min(mdl, run, np.random.default_rng(seed))  # default: one shot per round
        assert res["N"] == comb(m, k) and res["rounds"] and res["oracle_calls"] > 0
        assert res["grover_iterations"] == sum(r["j"] for r in res["rounds"])
        assert res["oracle_calls"] == 1 + res["grover_iterations"] + len(res["rounds"])
        assert res["rounds"][-1]["oracle_calls_cum"] == res["oracle_calls"]
        assert res["best_cost"] == table[res["best_index"]]
        hits += res["best_index"] == brute_force(table)[0]
    assert hits >= 9


def test_find_min_round_records():
    mdl = random_model(6, 3, seed=5)
    res = find_min(mdl, lambda c, s: aer_run(c, s, seed=0), np.random.default_rng(0), max_rounds=4)
    assert len(res["rounds"]) <= 4
    r = res["rounds"][0]
    assert set(r) >= {"T", "j", "best_sample_cost", "improved", "oracle_calls_cum"}
    assert "converged" in res
    assert all(isinstance(r["best_index"], int) for r in res["rounds"])
    assert res["rounds"][-1]["best_index"] == res["best_index"]


def test_find_min_defaults_one_shot_and_80_rounds():
    import inspect
    from qopt.search import dh_done
    sig = inspect.signature(find_min).parameters
    assert sig["shots"].default == 1 and sig["max_rounds"].default == 120
    assert inspect.signature(dh_done).parameters["max_rounds"].default == 120


def test_find_min_counts_one_call_per_round():
    mdl = random_model(6, 3, seed=5)
    shots_seen = []

    def run(circuits, shots):
        shots_seen.append(shots)
        return aer_run(circuits, shots, seed=len(shots_seen))

    res = find_min(mdl, run, np.random.default_rng(1))
    assert set(shots_seen) == {1}
    cum = 1  # the random start is one placement scored
    for r in res["rounds"]:
        cum += r["j"] + 1
        assert r["oracle_calls_cum"] == cum
    assert res["oracle_calls"] == cum
    assert res["grover_iterations"] == cum - len(res["rounds"]) - 1


def test_dh_init_counts_the_random_start():
    mdl = random_model(6, 3, seed=9)
    table = model_table(mdl)
    state = dh_init(mdl, np.random.default_rng(3))
    assert state["oracle_calls"] == 1 and state["N"] == comb(6, 3)
    assert np.isfinite(table[state["best_index"]]) and state["T"] == table[state["best_index"]]


def _small_sweep():
    mdl = random_model(5, 2, seed=5)
    table = model_table(mdl)
    return sweep_circuits(mdl, sweep_plan(table, M_targets=(1, 4), max_circuits=3))


def test_to_basis_leaves_only_basic_gates():
    circuits = _small_sweep()
    out = to_basis(circuits)
    assert len(out) == len(circuits)
    for qc in out:
        assert set(qc.count_ops()) <= set(BASIS) | {"measure"}


def test_to_basis_runs_on_basic_simulator():
    circuit = to_basis(_small_sweep())[-1]
    counts = BasicSimulator().run(circuit, shots=100).result().get_counts()
    assert sum(counts.values()) == 100 and all(len(key) == 5 for key in counts)
