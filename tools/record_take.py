"""Generate data/recorded_aria1.json locally with Aer as a stand-in until the real IonQ Aria 1 run.
Run from the project root: python tools/record_take.py
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qec.lessons import build_all

SETTINGS = {"shots": 500, "inject": {"pauli": "X", "qubit": 1}}


def main():
    circuits, meta = build_all(SETTINGS)
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["h", "x", "y", "z", "t", "sdg", "ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.006, 2), ["cx"])
    sim = AerSimulator(noise_model=noise)
    result = sim.run(transpile(circuits, sim, optimization_level=0), shots=SETTINGS["shots"], seed_simulator=1234).result()
    counts = [dict(result.get_counts(i)) for i in range(len(circuits))]
    take = {
        "source": "local-aer-stand-in",
        "backend": "AerSimulator depolarizing (stand-in, replace with IonQ Aria 1 noise model)",
        "date": datetime.date.today().isoformat(),
        "settings": SETTINGS, "meta": meta, "counts": counts,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "recorded_aria1.json"
    out.write_text(json.dumps(take))
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
