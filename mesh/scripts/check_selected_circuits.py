"""Check qollab/selected_circuits.json on its own (no notebook): every circuit is rebuilt with qollab/mesh_ionq.py, its noiseless
P(optimum) and P(within 5 %) must equal the saved predictions, and 1,000 sampled shots decoded with region_bits_of_key and graded
with grade() (the main.py path) must give consistent numbers. Also re-checks the exact optimum by brute force.
usage: python scripts/check_selected_circuits.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qollab"))
import mesh_ionq  # noqa: E402

selection = json.loads((ROOT / "qollab" / "selected_circuits.json").read_text())
n = selection["n"]
for name, instance in selection["instances"].items():
    energies = mesh_ionq.check_exact_optimum(instance, n)                      # brute force: optimum and optimal states
    near = np.flatnonzero(energies <= 1.05 * instance["e_min"] + 1e-12)
    assert set(near.tolist()) == set(instance["near_states"]), f"{name}: near-optimal set differs"
print(f"exact optima re-checked by brute force for {len(selection['instances'])} scenarios")
for block in selection["circuits"] + [selection["fallback"]]:
    spec = block["spec"]
    scenarios = list(selection["instances"]) if block["graded_under"] == "both scenarios" else [block["graded_under"]]
    circuit = mesh_ionq.circuit_from_spec(spec, measure=False)
    probs_qubits = Statevector(circuit).probabilities()
    state = Statevector(circuit)
    state.seed(2026 + block["index"])
    counts = {str(k): int(v) for k, v in state.sample_counts(1000).items()}
    for scenario in scenarios:
        instance = selection["instances"][scenario]
        full = np.zeros(2**n)
        for index, p in enumerate(probs_qubits):
            bits = mesh_ionq.region_bits_of_key(format(index, f"0{spec['n_qubits']}b"), spec, n)
            full[int(np.dot(bits, 1 << np.arange(n)))] += p
        p_opt, p_near = float(full[instance["optimal_states"]].sum()), float(full[instance["near_states"]].sum())
        if block["graded_under"] != "both scenarios":
            assert abs(p_opt - block["predicted"]["p_opt"]) < 1e-9 and abs(p_near - block["predicted"]["p_near"]) < 1e-9, f"{block['name']}: prediction mismatch"
        decoded = {}
        for key, count in counts.items():
            bits = mesh_ionq.region_bits_of_key(key, spec, n)
            region_key = "".join(str(b) for b in bits[::-1])                  # back to a Qiskit-style key over all regions
            decoded[region_key] = decoded.get(region_key, 0) + count
        graded = mesh_ionq.grade(decoded, instance, n)
        print(f"  ok  [{scenario}] {block['name']}: exact P(opt) {p_opt:.4f}, P(<=5%) {p_near:.4f}; 1,000 sampled shots: "
              f"P(opt) {graded['p_opt']:.3f}, best gap {graded['best_gap_percent']:.1f} %")
print(f"all {len(selection['circuits'])} batch circuits and the fallback rebuild exactly from qollab/selected_circuits.json")
