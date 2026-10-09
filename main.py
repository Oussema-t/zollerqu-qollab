"""ZollerQu - Qollab x IonQ hackathon: 2-qubit smoke test.

This effort is supported by Qollab & IonQ.
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


def bell_circuit():
    """Two qubits in the Bell state: expect only '00' and '11', about 50/50."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def run_local(qc):
    """Exact probabilities, no simulator needed (works locally and on Qollab)."""
    probs = {k: round(float(v), 3) for k, v in Statevector(qc).probabilities_dict().items()}
    print("Exact probabilities:", probs)


def run_ionq(qc, ionq_backend, shots=100):
    """One small job on the backend chosen in Qollab's Run dialog."""
    measured = qc.copy()
    measured.measure_all()
    compiled = transpile(measured, ionq_backend, optimization_level=1)
    try:
        job = ionq_backend.run(compiled, shots=shots)
        print("Backend counts:", job.result().get_counts())
    except Exception as err:  # qiskit-ionq can hide the real error
        print("Backend run failed:", type(err).__name__, err)


def main():
    qc = bell_circuit()
    run_local(qc)
    ionq_backend = globals().get("backend")  # Qollab pre-creates `backend`
    if ionq_backend is None:
        print("No Qollab backend found: local test only.")
    else:
        print("Running on:", getattr(ionq_backend, "name", ionq_backend))
        run_ionq(qc, ionq_backend)


main()
