"""Aer stand-in for the injected `run(circuits, shots)` backend."""
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

# Built once: rebuilding the simulator target and pass manager per call dominates small runs.
_SIM = AerSimulator(method="statevector")
_PM = generate_preset_pass_manager(optimization_level=1, backend=_SIM)


def aer_run(circuits, shots, seed=None):
    result = _SIM.run(_PM.run(circuits), shots=shots, seed_simulator=seed).result()
    return [result.get_counts(i) for i in range(len(circuits))]
