"""Grover threshold search and Durr-Hoyer minimum finding over a cost table.

Bitstring convention: site i = qubit i = bit i of the index. In a Qiskit counts key
(little-endian) site i is the character at position m-1-i, so int(key, 2) is the index.
"""
from math import asin, ceil, pi, sin, sqrt

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate


def index_from_key(key):
    return int(key.replace(" ", ""), 2)


def _marked(cost, T, strict):
    return cost < T if strict else cost <= T


def marked_count(cost, T, strict=False):
    return int(_marked(cost, T, strict).sum())


def oracle(cost, T, strict=False):
    return DiagonalGate(np.where(_marked(cost, T, strict), -1.0, 1.0).tolist())


def _diffusion(m):
    qc = QuantumCircuit(m, name="D")
    qc.h(range(m))
    qc.x(range(m))
    qc.h(m - 1)
    qc.mcx(list(range(m - 1)), m - 1)
    qc.h(m - 1)
    qc.x(range(m))
    qc.h(range(m))
    return qc.to_gate()


def grover_circuit(cost, T, j, strict=False):
    m = int(np.log2(len(cost)))
    qc = QuantumCircuit(m, m)
    qc.h(range(m))
    if j:
        o, d = oracle(cost, T, strict), _diffusion(m)
        for _ in range(j):
            qc.append(o, range(m))
            qc.append(d, range(m))
    qc.measure(range(m), range(m))
    return qc


def success_theory(M, N, j):
    return sin((2 * j + 1) * asin(sqrt(M / N))) ** 2


def sweep_plan(cost, M_targets=(1, 4, 16), max_circuits=12):
    N = len(cost)
    finite = np.sort(cost[np.isfinite(cost)])
    plan = []
    for target in M_targets:
        T = float(finite[target - 1])
        M = marked_count(cost, T)
        J = min(max_circuits - 1, ceil(1.2 * pi / 4 * sqrt(N / M)))
        plan.append({"M": M, "T": T, "js": list(range(J + 1))})
    return plan


def sweep_circuits(cost, plan):
    return [grover_circuit(cost, s["T"], j) for s in plan for j in s["js"]]


def sweep_analyze(cost, plan, counts_list):
    N = len(cost)
    counts_iter = iter(counts_list)
    out = []
    for s in plan:
        points = []
        for j in s["js"]:
            counts = next(counts_iter)
            shots = sum(counts.values())
            hits = sum(c for key, c in counts.items() if cost[index_from_key(key)] <= s["T"])
            points.append({"j": j, "measured": hits / shots, "theory": success_theory(s["M"], N, j)})
        out.append({"M": s["M"], "T": s["T"], "points": points,
                    "max_dev": max(abs(p["measured"] - p["theory"]) for p in points)})
    return out


# Durr-Hoyer minimum finding, split into steps so the browser can drive it one circuit at a time.

LAM = 6 / 5


def dh_init(cost, rng):
    x = int(rng.choice(np.flatnonzero(np.isfinite(cost))))
    return {"N": len(cost), "T": float(cost[x]), "best_index": x, "mstep": 1.0,
            "stagnant": 0, "j": None, "grover_iterations": 0, "oracle_calls": 0, "rounds": []}


def dh_next(cost, state, rng):
    """Pick j uniformly in [0, mstep) and return the strict-threshold Grover circuit."""
    state["j"] = int(rng.integers(0, ceil(state["mstep"])))
    return grover_circuit(cost, state["T"], state["j"], strict=True)


def dh_feed(cost, state, counts):
    """Take the best sample of the last circuit's counts; returns the round record.

    Oracle calls per round = j Grover iterations + 1 classical query scoring the measured sample.
    """
    j = state["j"]
    state["grover_iterations"] += j
    state["oracle_calls"] += j + 1
    idx = min((index_from_key(k) for k in counts), key=lambda i: cost[i])
    sample = float(cost[idx])
    improved = sample < state["T"]
    record = {"T": state["T"], "j": j, "best_sample_cost": sample, "improved": bool(improved),
              "oracle_calls_cum": state["oracle_calls"]}
    if improved:
        state["T"], state["best_index"] = sample, idx
        state["mstep"], state["stagnant"] = 1.0, 0
    else:
        state["mstep"] = min(LAM * state["mstep"], sqrt(state["N"]))
        if state["mstep"] == sqrt(state["N"]):
            state["stagnant"] += 1
    record["best_index"] = state["best_index"]
    state["rounds"].append(record)
    return record


def dh_done(state, max_rounds=80, max_oracle_calls=None):
    """Stop after 3 consecutive non-improving rounds at mstep = sqrt(N), or on a budget."""
    return (state["stagnant"] >= 3 or len(state["rounds"]) >= max_rounds
            or (max_oracle_calls is not None and state["oracle_calls"] >= max_oracle_calls))


def dh_result(cost, state):
    return {"best_index": state["best_index"], "best_cost": state["T"], "rounds": state["rounds"],
            "grover_iterations": state["grover_iterations"], "oracle_calls": state["oracle_calls"], "N": state["N"],
            "converged": marked_count(cost, state["T"], strict=True) == 0}


def find_min(cost, run, rng, shots=1, max_rounds=80, max_oracle_calls=None):
    state = dh_init(cost, rng)
    while not dh_done(state, max_rounds, max_oracle_calls):
        circuit = dh_next(cost, state, rng)
        dh_feed(cost, state, run([circuit], shots)[0])
    return dh_result(cost, state)
