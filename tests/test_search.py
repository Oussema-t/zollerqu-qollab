from math import comb

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector

from qopt.aer_run import aer_run
from qopt.classical import brute_force
from qopt.cost import feasible_mask
from qopt.search import (BASIS, dh_init, dicke_circuit, feasible_count, find_min, grover_circuit, index_from_key,
                         marked_count, oracle, success_theory, sweep_analyze, sweep_circuits, sweep_plan, to_basis)


def random_table(m, k, seed):
    rng = np.random.default_rng(seed)
    table = rng.uniform(10, 100, 2 ** m)
    table[~feasible_mask(m, k)] = np.inf
    return table


def test_oracle_phases_match_threshold():
    m = 6
    table = random_table(m, 3, seed=0)
    T = np.sort(table)[5]
    qc = QuantumCircuit(m)
    qc.h(range(m))
    qc.append(oracle(table, T), range(m))
    amps = Statevector(qc).data * np.sqrt(2 ** m)
    assert np.allclose(amps, np.where(table <= T, -1, 1))
    assert marked_count(table, T, strict=False) == 6
    assert marked_count(table, T, strict=True) == 5


def test_strict_oracle_excludes_threshold():
    m = 6
    table = random_table(m, 3, seed=1)
    T = np.sort(table)[3]
    qc = QuantumCircuit(m)
    qc.h(range(m))
    qc.append(oracle(table, T, strict=True), range(m))
    amps = Statevector(qc).data * np.sqrt(2 ** m)
    assert np.allclose(amps, np.where(table < T, -1, 1))


def test_oracle_never_marks_infeasible_entries():
    m = 6
    table = random_table(m, 3, seed=4)
    qc = QuantumCircuit(m)
    qc.h(range(m))
    qc.append(oracle(table, np.inf), range(m))
    amps = Statevector(qc).data * np.sqrt(2 ** m)
    assert np.allclose(amps, np.where(np.isfinite(table), -1, 1))
    assert marked_count(table, np.inf) == comb(6, 3)


@pytest.mark.parametrize("m,k", [(4, 2), (6, 3), (8, 4), (10, 4)])
def test_dicke_circuit_is_uniform_over_weight_k(m, k):
    amps = Statevector(dicke_circuit(m, k)).data
    expected = np.where(feasible_mask(m, k), 1 / np.sqrt(comb(m, k)), 0)
    assert np.allclose(amps, expected, atol=1e-9)


def test_feasible_count():
    assert feasible_count(random_table(8, 4, seed=0)) == 70


def test_index_from_key_little_endian():
    # site 0 = qubit 0 = last character
    assert index_from_key("0001") == 1
    assert index_from_key("1000") == 8
    assert index_from_key("10 01") == 9


def test_uniform_start_samples_only_feasible():
    m, k = 8, 4
    table = random_table(m, k, seed=6)
    counts = aer_run([grover_circuit(table, np.sort(table)[0], 0)], 4000, seed=2)[0]
    assert all(bin(index_from_key(key)).count("1") == k for key in counts)
    assert len(counts) > 60  # spread over the 70 feasible placements


def test_grover_circuit_k_inferred_equals_explicit():
    table = random_table(6, 3, seed=8)
    T = np.sort(table)[2]
    a = Statevector(grover_circuit(table, T, 2).remove_final_measurements(inplace=False))
    b = Statevector(grover_circuit(table, T, 2, k=3).remove_final_measurements(inplace=False))
    assert a.equiv(b)


def test_success_probability_matches_theory():
    m = 8
    table = random_table(m, 4, seed=2)
    N = comb(8, 4)
    assert feasible_count(table) == N == 70
    finite = np.sort(table[np.isfinite(table)])
    circuits, expected = [], []
    for M in (1, 4, 16):
        T = finite[M - 1]
        for j in range(7):
            circuits.append(grover_circuit(table, T, j))
            expected.append((T, success_theory(M, N, j)))
    counts_list = aer_run(circuits, 4000, seed=7)
    for counts, (T, theory) in zip(counts_list, expected):
        hits = sum(c for key, c in counts.items() if table[index_from_key(key)] <= T)
        assert abs(hits / 4000 - theory) < 0.03


def test_sweep_plan_and_analyze_round_trip():
    m = 8
    table = random_table(m, 3, seed=3)
    plan = sweep_plan(table, M_targets=(1, 4), max_circuits=5)
    assert [s["M"] for s in plan] == [1, 4]
    assert all(s["N"] == comb(8, 3) for s in plan)  # feasible count, not 2^m
    assert plan[0]["js"] == [0, 1, 2, 3, 4]  # ceil(1.2*pi/4*sqrt(56))=8, capped at max_circuits-1
    assert plan[1]["js"] == [0, 1, 2, 3, 4]  # ceil(1.2*pi/4*sqrt(14))=4
    assert plan[0]["T"] == np.sort(table)[0]
    circuits = sweep_circuits(table, plan)
    assert len(circuits) == 10 and all(c.num_clbits == m for c in circuits)
    analysis = sweep_analyze(table, plan, aer_run(circuits, 4000, seed=1))
    assert [s["M"] for s in analysis] == [1, 4]
    for s in analysis:
        assert s["N"] == comb(8, 3)
        assert [p["j"] for p in s["points"]] == [0, 1, 2, 3, 4]
        assert all(p["theory"] == success_theory(s["M"], comb(8, 3), p["j"]) for p in s["points"])
        assert s["max_dev"] == max(abs(p["measured"] - p["theory"]) for p in s["points"])
        assert s["max_dev"] < 0.03


def test_sweep_plan_iteration_count():
    m = 8
    table = random_table(m, 3, seed=3)
    plan = sweep_plan(table, M_targets=(16,), max_circuits=12)
    # N = C(8,3) = 56: ceil(1.2 * pi/4 * sqrt(56/16)) = ceil(1.76) = 2
    assert plan[0]["js"] == [0, 1, 2] and plan[0]["M"] == 16


def test_find_min_reaches_brute_force_optimum():
    m = 8
    hits = 0
    for seed in range(10):
        table = random_table(m, 3, seed=100 + seed)

        aer_seeds = iter(range(1000 * seed, 1000 * seed + 1000))

        def run(circuits, shots):
            return aer_run(circuits, shots, seed=next(aer_seeds))

        res = find_min(table, run, np.random.default_rng(seed))  # default: one shot per round
        assert res["N"] == comb(8, 3) and res["rounds"] and res["oracle_calls"] > 0
        assert res["grover_iterations"] == sum(r["j"] for r in res["rounds"])
        assert res["oracle_calls"] == 1 + res["grover_iterations"] + len(res["rounds"])
        assert res["rounds"][-1]["oracle_calls_cum"] == res["oracle_calls"]
        assert res["best_cost"] == table[res["best_index"]]
        hits += res["best_index"] == brute_force(table)[0]
    assert hits >= 9


def test_find_min_round_records():
    table = random_table(6, 3, seed=5)
    res = find_min(table, lambda c, s: aer_run(c, s, seed=0), np.random.default_rng(0), max_rounds=4)
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
    table = random_table(6, 3, seed=5)
    shots_seen = []

    def run(circuits, shots):
        shots_seen.append(shots)
        return aer_run(circuits, shots, seed=len(shots_seen))

    res = find_min(table, run, np.random.default_rng(1))
    assert set(shots_seen) == {1}
    cum = 1  # the random start is one lookup
    for r in res["rounds"]:
        cum += r["j"] + 1
        assert r["oracle_calls_cum"] == cum
    assert res["oracle_calls"] == cum
    assert res["grover_iterations"] == cum - len(res["rounds"]) - 1


def test_dh_init_counts_the_random_start():
    table = random_table(6, 3, seed=9)
    state = dh_init(table, np.random.default_rng(3))
    assert state["oracle_calls"] == 1 and state["N"] == comb(6, 3)
    assert np.isfinite(table[state["best_index"]]) and state["T"] == table[state["best_index"]]


def _small_sweep():
    table = random_table(6, 3, seed=5)
    return sweep_circuits(table, sweep_plan(table, M_targets=(1, 4), max_circuits=3))


def test_to_basis_leaves_only_basic_gates():
    circuits = _small_sweep()
    out = to_basis(circuits)
    assert len(out) == len(circuits)
    for qc in out:
        assert set(qc.count_ops()) <= set(BASIS) | {"measure"}


def test_to_basis_runs_on_basic_simulator():
    circuit = to_basis(_small_sweep())[-1]
    counts = BasicSimulator().run(circuit, shots=100).result().get_counts()
    assert sum(counts.values()) == 100
