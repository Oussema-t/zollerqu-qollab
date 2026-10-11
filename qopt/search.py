"""Grover threshold search and Durr-Hoyer minimum finding over the feasible placements.

The search space is the C(m,k) placements with exactly k stores opened among the m candidate locations:
the site register starts in the Dicke state |D^m_k> (uniform over them) and the diffusion reflects about
that state.

The threshold oracle computes the cost inside the circuit from the per-site integers A_i and the per-pair
integers B_ij of qopt.cost.model (Draper constant additions in the Fourier basis of a two's-complement
accumulator, a Toffoli AND ancilla for each pair), subtracts the threshold, flips the phase on the sign bit
and uncomputes. No table of the placements is used to build any circuit.

Bitstring convention: site i = qubit i = bit i of the index. In a Qiskit counts key
(little-endian) site i is the character at position m-1-i, so int(key, 2) is the index. Only the m site
qubits are measured.
"""
from math import acos, asin, ceil, comb, pi, sin, sqrt

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate

from qopt.cost import model_cost, model_table


BASIS = ["rz", "sx", "x", "cx"]


def to_basis(circuits):
    """Unroll to standard gates only (Qollab's BasicSimulator rejects ccry). Takes and returns a list."""
    return transpile(circuits, basis_gates=BASIS, optimization_level=0)


def index_from_key(key):
    return int(key.replace(" ", ""), 2)


def _marked(cost, T, strict):
    return np.isfinite(cost) & (cost < T if strict else cost <= T)


def feasible_count(cost):
    return int(np.isfinite(cost).sum())


def marked_count(cost, T, strict=False):
    return int(_marked(cost, T, strict).sum())


def _threshold_int(mdl, T, strict):
    """cost < T (strict) is cost_int < T_int; cost <= T is cost_int < T_int + 1. T is a multiple of 1/scale."""
    return int(round(T * mdl["scale"])) + (0 if strict else 1)


def accumulator_bits(mdl, t):
    """Two's-complement width that holds cost_int - t for every feasible placement.

    From coefficient bounds only: the k smallest (largest) A plus the k(k-1)/2 smallest (largest) pair
    coefficients, absent pairs counting 0. Never from the table.
    """
    m, k = mdl["m"], mdl["k"]
    A = sorted(mdl["A"])
    B = sorted(list(mdl["B"].values()) + [0] * (comb(m, 2) - len(mdl["B"])))
    pairs = k * (k - 1) // 2
    lo = sum(A[:k]) + sum(B[:pairs])
    hi = sum(A[m - k:]) + sum(B[len(B) - pairs:])
    return 1 + max(max(hi - t, 0).bit_length(), max(t - lo - 1, 0).bit_length())


def _qft(n):
    """Standard QFT on n qubits, little-endian: |x> -> sum_y exp(2 pi i x y / 2^n) |y> / sqrt(2^n)."""
    qc = QuantumCircuit(n, name="QFT")
    for j in reversed(range(n)):
        qc.h(j)
        for q in reversed(range(j)):
            qc.cp(pi / 2 ** (j - q), q, j)
    for q in range(n // 2):
        qc.swap(q, n - 1 - q)
    return qc


def _add_const(qc, c, acc, ctrl=None):
    """Add the integer c to the accumulator held in the Fourier basis (Draper): one phase per accumulator qubit."""
    n = len(acc)
    for l, q in enumerate(acc):
        angle = 2 * pi * ((c * 2 ** l) % 2 ** n) / 2 ** n
        if angle == 0:
            continue
        if ctrl is None:
            qc.p(angle, q)
        else:
            qc.cp(angle, ctrl, q)


def oracle(mdl, T, strict=False):
    """Phase -1 on the placements with cost < T (strict) or cost <= T, computed in-circuit.

    Qubits: m sites, then an n-qubit accumulator (two's complement), then one AND ancilla. The accumulator
    and the ancilla start and end in |0>.
    """
    m = mdl["m"]
    t = _threshold_int(mdl, T, strict)
    n = accumulator_bits(mdl, t)
    acc, anc = list(range(m, m + n)), m + n
    compute = QuantumCircuit(m + n + 1)
    compute.h(acc)                                # the QFT of |0>
    for i, a in enumerate(mdl["A"]):
        _add_const(compute, a, acc, ctrl=i)
    for (i, j), b in mdl["B"].items():
        compute.ccx(i, j, anc)
        _add_const(compute, b, acc, ctrl=anc)
        compute.ccx(i, j, anc)
    _add_const(compute, -t, acc)
    compute.compose(_qft(n).inverse(), acc, inplace=True)   # accumulator = cost_int - t; sign bit = cost_int < t
    qc = QuantumCircuit(m + n + 1, name="O")
    qc.compose(compute, inplace=True)
    qc.z(acc[-1])
    qc.compose(compute.inverse(), inplace=True)
    return qc


def _scs(qc, n, k):
    """Split-and-cyclic-shift block SCS_{n,k} of Bartschi and Eidenbenz (arXiv:1904.07358).

    Paper qubits 1..n are circuit qubits 0..n-1; the block acts on paper qubits n-k..n.
    """
    q = lambda p: p - 1  # noqa: E731
    qc.cx(q(n - 1), q(n))
    qc.cry(2 * acos(sqrt(1 / n)), q(n), q(n - 1))
    qc.cx(q(n - 1), q(n))
    for l in range(2, k + 1):
        qc.cx(q(n - l), q(n))
        qc.append(RYGate(2 * acos(sqrt(l / n))).control(2, annotated=False), [q(n), q(n - l + 1), q(n - l)])
        qc.cx(q(n - l), q(n))


def dicke_circuit(m, k):
    """Deterministic |D^m_k> preparation, no ancilla, O(km) gates (Bartschi and Eidenbenz, Lemma 2).

    Start from |0..0 1..1> with ones on the k top-index qubits, then apply
    SCS_{l,k} for l = m..k+1 followed by SCS_{l,l-1} for l = k..2.
    """
    qc = QuantumCircuit(m, name="Dicke")
    qc.x(range(m - k, m))
    for n in range(m, k, -1):
        _scs(qc, n, k)
    for n in range(k, 1, -1):
        _scs(qc, n, n - 1)
    return qc


def _diffusion(prep):
    """D = A (2|0><0| - I) A^dagger, up to a global phase, with A the feasible-uniform preparation."""
    m = prep.num_qubits
    qc = QuantumCircuit(m, name="D")
    qc.compose(prep.inverse(), inplace=True)
    qc.x(range(m))
    qc.h(m - 1)
    qc.mcx(list(range(m - 1)), m - 1)
    qc.h(m - 1)
    qc.x(range(m))
    qc.compose(prep, inplace=True)
    return qc.to_gate()


def grover_circuit(mdl, T, j, strict=False):
    """Dicke start, j Grover iterations with the in-circuit threshold oracle, measure the m site qubits."""
    m, k = mdl["m"], mdl["k"]
    n = accumulator_bits(mdl, _threshold_int(mdl, T, strict))
    sites = QuantumRegister(m, "site")
    qc = QuantumCircuit(sites, QuantumRegister(n, "acc"), QuantumRegister(1, "and"), ClassicalRegister(m, "c"))
    prep = dicke_circuit(m, k)
    qc.compose(prep, sites, inplace=True)
    if j:
        o, d = oracle(mdl, T, strict).to_gate(), _diffusion(prep)
        for _ in range(j):
            qc.append(o, qc.qubits)
            qc.append(d, sites)
    qc.measure(sites, qc.clbits)
    return qc


def success_theory(M, N, j):
    return sin((2 * j + 1) * asin(sqrt(M / N))) ** 2


def sweep_plan(cost, M_targets=(1, 4, 16), max_circuits=12):
    N = feasible_count(cost)
    finite = np.sort(cost[np.isfinite(cost)])
    plan = []
    for target in M_targets:
        T = float(finite[target - 1])
        M = marked_count(cost, T)
        J = min(max_circuits - 1, ceil(1.2 * pi / 4 * sqrt(N / M)))
        plan.append({"M": M, "N": N, "T": T, "js": list(range(J + 1))})
    return plan


def sweep_circuits(mdl, plan):
    return [grover_circuit(mdl, s["T"], j) for s in plan for j in s["js"]]


def sweep_analyze(cost, plan, counts_list):
    N = feasible_count(cost)
    counts_iter = iter(counts_list)
    out = []
    for s in plan:
        points = []
        for j in s["js"]:
            counts = next(counts_iter)
            shots = sum(counts.values())
            hits = sum(c for key, c in counts.items() if cost[index_from_key(key)] <= s["T"])
            points.append({"j": j, "measured": hits / shots, "theory": success_theory(s["M"], N, j)})
        out.append({"M": s["M"], "N": N, "T": s["T"], "points": points,
                    "max_dev": max(abs(p["measured"] - p["theory"]) for p in points)})
    return out


# Durr-Hoyer minimum finding, split into steps so the browser can drive it one circuit at a time.

LAM = 6 / 5


def dh_init(mdl, rng):
    """Random feasible start; scoring it is one placement scored, so oracle_calls starts at 1."""
    sites = sorted(int(i) for i in rng.choice(mdl["m"], mdl["k"], replace=False))
    return {"N": comb(mdl["m"], mdl["k"]), "T": model_cost(mdl, sites), "best_index": sum(1 << i for i in sites),
            "mstep": 1.0, "stagnant": 0, "j": None, "grover_iterations": 0, "oracle_calls": 1, "rounds": []}


def dh_next(mdl, state, rng):
    """Pick j uniformly in [0, mstep) and return the strict-threshold Grover circuit."""
    state["j"] = int(rng.integers(0, ceil(state["mstep"])))
    return grover_circuit(mdl, state["T"], state["j"], strict=True)


def _score(mdl, index):
    sites = [i for i in range(mdl["m"]) if index >> i & 1]
    return model_cost(mdl, sites) if len(sites) == mdl["k"] else float("inf")


def dh_feed(mdl, state, counts):
    """Take the best sample of the last circuit's counts; returns the round record.

    Oracle calls per round = j Grover iterations + 1 classical score of the measured sample
    (plus the 1 for the random start counted in dh_init).
    """
    j = state["j"]
    state["grover_iterations"] += j
    state["oracle_calls"] += j + 1
    idx = min((index_from_key(k) for k in counts), key=lambda i: _score(mdl, i))
    sample = _score(mdl, idx)
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


def dh_done(state, max_rounds=120, max_oracle_calls=None):
    """Stop after 3 consecutive non-improving rounds at mstep = sqrt(N), or on a budget."""
    return (state["stagnant"] >= 3 or len(state["rounds"]) >= max_rounds
            or (max_oracle_calls is not None and state["oracle_calls"] >= max_oracle_calls))


def dh_result(mdl, state):
    """converged is a check against the classical table after the run; the run itself never uses the table."""
    return {"best_index": state["best_index"], "best_cost": state["T"], "rounds": state["rounds"],
            "grover_iterations": state["grover_iterations"], "oracle_calls": state["oracle_calls"], "N": state["N"],
            "converged": marked_count(model_table(mdl), state["T"], strict=True) == 0}


def find_min(mdl, run, rng, shots=1, max_rounds=120, max_oracle_calls=None):
    state = dh_init(mdl, rng)
    while not dh_done(state, max_rounds, max_oracle_calls):
        circuit = dh_next(mdl, state, rng)
        dh_feed(mdl, state, run([circuit], shots)[0])
    return dh_result(mdl, state)
