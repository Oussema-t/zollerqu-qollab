"""Two-qubit gates of our circuits on all-to-all connectivity (IonQ: one native ZZ per rzz) versus the same circuits routed
on a heavy-hex coupling map (SWAP routing). MEASURED with the Qiskit transpiler only (no simulation, no hardware), with the
settings of the stocks project so the two are comparable: optimisation level 1, five routing seeds, basis rzz / rz / rx / sx / x.
Circuits: one dense QAOA cost + mixer layer at 12 and 29 qubits (every pair of regions is coupled by the balance term), and the
trained varQITE ZY circuit that ran on the IonQ Forte 1 noise model (from qollab/selected_circuits.json).
usage: python scripts/heavy_hex_gate_counts.py   ->  results/ionq_selection/heavy_hex_gate_counts.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qollab"))
import mesh_ionq  # noqa: E402

BASIS = ["rzz", "rz", "rx", "sx", "x"]
SEEDS = [2026, 2027, 2028, 2029, 2030]


def two_qubit(circuit):
    return sum(1 for item in circuit.data if item.operation.num_qubits == 2)


def dense_layer(n):
    """One QAOA layer on n qubits with every pair coupled (angles do not matter for gate counts)."""
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for layer_round in mesh_ionq.round_robin_pairs(n):
        for i, j in layer_round:
            circuit.rzz(0.3, i, j)
    circuit.rx(0.4, range(n))
    return circuit


def measure(name, circuit):
    n = circuit.num_qubits
    distance = next(d for d in (3, 5, 7) if CouplingMap.from_heavy_hex(d).size() >= n)
    heavy_hex = CouplingMap.from_heavy_hex(distance)
    flat = transpile(circuit, basis_gates=BASIS, optimization_level=1)
    routed = [transpile(circuit, coupling_map=heavy_hex, basis_gates=BASIS, optimization_level=1, seed_transpiler=seed) for seed in SEEDS]
    counts = [two_qubit(c) for c in routed]
    row = dict(circuit=name, qubits=n, all_to_all_two_qubit_gates=two_qubit(flat),
               all_to_all_two_qubit_depth=flat.depth(lambda item: item.operation.num_qubits == 2),
               heavy_hex_qubits=heavy_hex.size(), heavy_hex_distance=distance, heavy_hex_seeds=SEEDS,
               heavy_hex_two_qubit_gates=dict(min=int(min(counts)), median=float(np.median(counts)), max=int(max(counts))),
               heavy_hex_two_qubit_depth_min=int(min(c.depth(lambda item: item.operation.num_qubits == 2) for c in routed)))
    row["ratio_median"] = row["heavy_hex_two_qubit_gates"]["median"] / row["all_to_all_two_qubit_gates"]
    print(f"{name}: all-to-all {row['all_to_all_two_qubit_gates']}, heavy-hex {row['heavy_hex_two_qubit_gates']} ({row['ratio_median']:.1f}x)")
    return row


if __name__ == "__main__":
    selection = json.loads((ROOT / "qollab" / "selected_circuits.json").read_text())
    zy = next(c for c in selection["circuits"] if c["name"].startswith("varQITE ZY"))
    rows = [measure("one dense QAOA layer, 12 qubits", dense_layer(12)),
            measure("one dense QAOA layer, 29 qubits", dense_layer(29)),
            measure("varQITE ZY (the circuit run on the IonQ Forte 1 noise model), 12 qubits", mesh_ionq.circuit_from_spec(zy["spec"], measure=False))]
    out = ROOT / "results" / "ionq_selection" / "heavy_hex_gate_counts.json"
    out.write_text(json.dumps(dict(method="Qiskit transpiler only: optimisation level 1, basis " + " / ".join(BASIS) + ", SWAP routing on "
                                          "CouplingMap.from_heavy_hex, five seeds; all-to-all = the same basis without a coupling map "
                                          "(one rzz = one native ZZ on IonQ Forte)", rows=rows), indent=1))
    print(f"wrote {out}")
