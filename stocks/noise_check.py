"""LOCAL-ONLY optional check: rerun the shortlisted circuits on qiskit-aer with a simple noise model.

Approximate noise model, NOT IonQ: depolarizing 0.5 % on every two-qubit gate, 0.05 % on every one-qubit gate except
the virtual rz, 0.5 % symmetric readout error. Purpose: confirm the ranking of the crude estimate, nothing more.
Run from the repo root:  python -m stocks.noise_check
"""
import json
import sys

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from stocks import select_circuits as sc

SHOTS = 4000
TWO_QUBIT, ONE_QUBIT, READOUT = 0.005, 0.0005, 0.005


def noise_model():
    nm = NoiseModel(basis_gates=sc.BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(TWO_QUBIT, 2), ["rzz", "cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(ONE_QUBIT, 1), ["rx", "ry", "h", "sx", "x"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - READOUT, READOUT], [READOUT, 1 - READOUT]]))
    return nm


def noisy_scores(circuit, c, backend):
    compiled = transpile(circuit, basis_gates=sc.BASIS, optimization_level=1, seed_transpiler=2026)
    counts = backend.run(compiled, shots=SHOTS, seed_simulator=2026).result().get_counts()
    hits = near = total = 0
    for key, count in counts.items():
        index = int(key.replace(" ", ""), 2)
        total += count
        if c.feasible[index] and c.objective[index] <= c.e_min + c.tolerance:
            hits += count
        if c.feasible[index] and c.objective[index] <= c.e_min + sc.WITHIN * (c.e_max - c.e_min):
            near += count
    return hits / total, near / total


def main():
    rows, _, _, _, built = sc.candidates()
    data = json.loads(sc.CANDIDATES_PATH.read_text())
    eligible = [r for r in rows if r["two_qubit_gates"] <= sc.MAX_TWO_QUBIT_GATES and not r["generic_state_preparation"]]
    shortlist = sorted(eligible, key=lambda r: -r["crude_noisy_p_optimum"])[:14]
    names = {r["name"] for r in shortlist} | set(sc.PREFERRED_BATCH)
    backend = AerSimulator(noise_model=noise_model())
    print(f"approximate noise model, not IonQ: 2q depolarizing {TWO_QUBIT:.1%}, 1q {ONE_QUBIT:.2%}, readout {READOUT:.1%}; {SHOTS} shots each")
    print(f"{'circuit':<40} {'2q':>4} {'noiseless':>10} {'crude':>8} {'Aer noisy':>10} {'<5% noiseless':>14} {'<5% Aer':>8}")
    results = {}
    for r in sorted((r for r in rows if r["name"] in names), key=lambda r: -r["crude_noisy_p_optimum"]):
        circuit, c = built[r["name"]]
        p_opt, p_near = noisy_scores(circuit, c, backend)
        results[r["name"]] = {"aer_p_optimum": p_opt, "aer_p_within_5pct": p_near, "shots": SHOTS}
        print(f"{r['name']:<40} {r['two_qubit_gates']:4d} {r['p_optimum_noiseless']:10.4f} {r['crude_noisy_p_optimum']:8.4f} {p_opt:10.4f} "
              f"{r['p_within_5pct_noiseless']:14.3f} {p_near:8.3f}")
    for row in data["rows"]:
        if row["name"] in results:
            row.update(results[row["name"]])
    data["aer_noise_model"] = {"two_qubit_depolarizing": TWO_QUBIT, "one_qubit_depolarizing": ONE_QUBIT, "readout": READOUT, "shots": SHOTS,
                              "note": "approximate noise model, not IonQ; local qiskit-aer check only"}
    sc.CANDIDATES_PATH.write_text(json.dumps(data, indent=1, default=float) + "\n")
    crude = [r["crude_noisy_p_optimum"] for r in rows if r["name"] in results]
    aer = [results[r["name"]]["aer_p_optimum"] for r in rows if r["name"] in results]
    from scipy.stats import spearmanr
    print("rank agreement crude vs Aer (Spearman):", round(float(spearmanr(crude, aer).statistic), 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
