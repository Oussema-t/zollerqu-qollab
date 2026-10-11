"""Helpers for the ZollerQu mesh IonQ batch: rebuild the circuits, run ONE job, grade every shot against the exact optimum."""
import time

import numpy as np
from qiskit import QuantumCircuit, transpile


def qaoa_circuit(J_scaled, h_scaled, gammas, betas, measure=True):
    """H on all qubits; per layer rzz(2 gamma J_ij) on every pair, rz(2 gamma h_i), rx(-2 beta) (mixer exp(+i beta sum X))."""
    J, h, n = np.asarray(J_scaled), np.asarray(h_scaled), len(h_scaled)
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    circuit.rzz(float(2 * gamma * J[i, j]), i, j)
        for q in range(n):
            if h[q] != 0:
                circuit.rz(float(2 * gamma * h[q]), q)
        for q in range(n):
            circuit.rx(float(-2 * beta), q)
    if measure:
        circuit.measure_all()
    return circuit


def varqite_circuit(gates, params, n, measure=True):
    """The trained varQITE circuit: H on all qubits, then ry / rx / cx / rzz / zy gates (zy = exp(i theta Z_c Y_t))."""
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for kind, qubits, index in gates:
        value = None if index is None else float(params[index])
        if kind == "ry":
            circuit.ry(value, qubits[0])
        elif kind == "rx":
            circuit.rx(value, qubits[0])
        elif kind == "cx":
            circuit.cx(*qubits)
        elif kind == "rzz":
            circuit.rzz(value, *qubits)
        elif kind == "zy":
            control, target = qubits
            circuit.sdg(target)
            circuit.h(target)
            circuit.rzz(-2.0 * value, control, target)
            circuit.h(target)
            circuit.s(target)
    if measure:
        circuit.measure_all()
    return circuit


def build_batch(batch, include_varqite=True):
    """All circuits of the batch, in a fixed order, with their labels."""
    circuits, labels = [], []
    for scenario in batch["scenarios"]:
        for spec in scenario["circuits"]:
            if spec["kind"] == "varqite" and not include_varqite:
                continue
            if spec["kind"] == "qaoa":
                circuits.append(qaoa_circuit(scenario["J_scaled"], scenario["h_scaled"], spec["gammas"], spec["betas"]))
            else:
                circuits.append(varqite_circuit(spec["gates"], spec["params"], batch["n"]))
            labels.append((scenario["name"], spec["name"]))
    return circuits, labels


def bits_of_key(key, n):
    """Qiskit count key -> bits in region order (the RIGHTMOST character is qubit / region 0)."""
    key = key.replace(" ", "")[-n:]
    return np.array([int(b) for b in key[::-1]])


def energies_of_bits(bits, J, h, const):
    z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    return z @ np.asarray(J) @ z + np.asarray(h) @ z + const


def check_exact_optimum(scenario, n):
    """Brute force over all 2^n splits: the minimum must equal the exported exact optimum."""
    states = np.arange(2**n)
    z = 1.0 - 2.0 * ((states[:, None] >> np.arange(n)) & 1)
    energies = np.einsum("si,ij,sj->s", z, np.asarray(scenario["J"]), z) + z @ np.asarray(scenario["h"]) + scenario["const"]
    assert abs(energies.min() - scenario["e_min"]) < 1e-9, "exported optimum does not match brute force"
    assert set(np.flatnonzero(energies <= energies.min() + 1e-9).tolist()) == set(scenario["optimal_states"])
    return energies


def grade(counts, scenario, n, near_factor=1.05, epsilon=0.05):
    """Metrics of one circuit's counts (every shot counted; nothing filtered)."""
    work = np.asarray(scenario["work"], dtype=float)
    shots = sum(counts.values())
    optimal = set(scenario["optimal_states"])
    e_min = scenario["e_min"]
    p_opt = p_near = p_bal = mean_cost = 0.0
    best = None
    for key, count in counts.items():
        bits = bits_of_key(key, n)
        state = int(np.dot(bits, 1 << np.arange(n)))
        cost = energies_of_bits(bits, scenario["J"], scenario["h"], scenario["const"])
        share_b = float(work @ bits / work.sum())
        p_opt += count * (state in optimal)
        p_near += count * (cost <= near_factor * e_min + 1e-12)
        p_bal += count * (abs(share_b - scenario["alpha"]) <= epsilon)
        mean_cost += count * cost
        if best is None or cost < best[0]:
            best = (cost, state, share_b)
    return dict(shots=int(shots), p_opt=float(p_opt / shots), p_near=float(p_near / shots), p_balanced=float(p_bal / shots),
                mean_cost=float(mean_cost / shots), best_cost=float(best[0]), best_gap_percent=float(100.0 * (best[0] / e_min - 1.0)),
                best_state=int(best[1]), best_share_b=float(best[2]), distinct_outcomes=len(counts))


def run_batch(circuits, backend, shots):
    """ONE job for all circuits. Returns a list of count dicts, or None if no backend or the run failed."""
    if backend is None:
        return None
    compiled = transpile(circuits, backend, optimization_level=1)
    try:
        job = backend.run(compiled, shots=shots)
        start = time.time()
        while True:
            try:
                status = job.status()
                name = getattr(status, "name", str(status))
            except Exception as error:                 # a status error must not stop the run
                name = f"unknown ({type(error).__name__})"
            print(f"  job status after {time.time() - start:5.0f} s: {name}", flush=True)
            if name in ("DONE", "ERROR", "CANCELLED") or "DONE" in str(name).upper():
                break
            time.sleep(10)
        result = job.result()
        return [result.get_counts(i) for i in range(len(compiled))]
    except Exception as error:
        print(f"IonQ run failed: {type(error).__name__}: {error}")
        return None


def local_counts(circuits, shots, seed=2026):
    """Local mode without IonQ: exact noiseless sampling of each circuit (measurement removed)."""
    from qiskit.quantum_info import Statevector
    counts = []
    for index, circuit in enumerate(circuits):
        bare = circuit.remove_final_measurements(inplace=False)
        state = Statevector(bare)
        state.seed(seed + index)
        counts.append({str(key): int(value) for key, value in state.sample_counts(shots).items()})
    return counts
