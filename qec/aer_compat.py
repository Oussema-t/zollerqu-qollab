"""Local Aer runs for tools/record_take.py, dev/serve.py and the tests. Not used by the Qollab runtime.

Each circuit runs on the fastest exact Aer method for it: stabilizer for Clifford-only circuits,
statevector up to 20 qubits, matrix product state above (the 27-qubit lesson 3 circuits).
Aer's extended_stabilizer is not used: its Metropolis sampler returns strings outside the code
space for these circuits (lesson 3 ideal P(1) came out 0.46 instead of 0.146).
"""
CLIFFORD = {"h", "s", "sdg", "x", "y", "z", "cx", "measure", "barrier"}
MAX_STATEVECTOR_QUBITS = 20


def stand_in_noise():
    """Depolarizing stand-in for a trapped-ion device."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["h", "x", "y", "z", "s", "sdg", "t", "ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.006, 2), ["cx"])
    return noise


def method_for(qc):
    if {ins.operation.name for ins in qc.data} <= CLIFFORD:
        return "stabilizer"
    if qc.num_qubits <= MAX_STATEVECTOR_QUBITS:
        return "statevector"
    return "matrix_product_state"


def simulate(circuits, shots, noise_model=None, seed=None):
    """Counts dicts, one per circuit, in the order given."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    groups = {}
    for k, qc in enumerate(circuits):
        groups.setdefault(method_for(qc), []).append(k)
    out = [None] * len(circuits)
    for method, ks in groups.items():
        sim = AerSimulator(method=method, noise_model=noise_model)
        tcs = transpile([circuits[k] for k in ks], sim, optimization_level=0)
        result = sim.run(tcs, shots=int(shots), seed_simulator=seed).result()
        for j, k in enumerate(ks):
            out[k] = dict(result.get_counts(j))
    return out
