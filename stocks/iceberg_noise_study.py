"""Local prediction (NOT IonQ): plain QAOA vs the Iceberg-protected circuit under an approximate depolarizing noise model.

    python -m stocks.iceberg_noise_study     # writes results/iceberg_noise_study.json

The circuits are built by qollab/main.py (the code that runs on Qollab). Noise model (qiskit-aer): depolarizing error
`rate` on every two-qubit gate (rzz, rxx, cx), rate/10 on every one-qubit gate except rz, 0.5 % readout flips. The
two-qubit rate is CALIBRATED so that the plain B CVaR 0.1 p=3 circuit reproduces the one IonQ Forte 1 noise-model run
(7 best portfolios in 1,000 shots); the uncalibrated 0.5 % model of stocks/noise_check.py is shown as well. Both are
approximations: the real noise model is unknown, and correlated or coherent errors would change the Iceberg numbers.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "iceberg_noise_study.json"
READOUT = 0.005
TARGET = ("B CVaR 0.1 p=3", 7 / 1000)
RATES = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02)


def load_main():
    code = (ROOT / "qollab" / "main.py").read_text()
    code = code[:code.index('if sys.platform != "emscripten":')]
    ns = {"__name__": "qollab_main"}
    exec(compile(code, "qollab/main.py", "exec"), ns)
    return ns


def noise_model(rate):
    nm = NoiseModel(basis_gates=["rzz", "rxx", "cx", "rz", "rx", "ry", "h", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 2), ["rzz", "rxx", "cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate / 10, 1), ["rx", "ry", "h", "sx", "x"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - READOUT, READOUT], [READOUT, 1 - READOUT]]))
    return nm


def counts_of(qc, rate, shots, seed, method):
    sim = AerSimulator(method=method, noise_model=noise_model(rate), seed_simulator=seed)
    compiled = transpile(qc, basis_gates=["rzz", "rxx", "cx", "rz", "rx", "ry", "h", "sx", "x"], optimization_level=1, seed_transpiler=seed)
    return sim.run(compiled, shots=shots).result().get_counts()


def score(ns, counts, spec):
    a = ns["analyse"](counts, spec["model"], spec["mode"]) if counts else None
    return a


def main():
    ns = load_main()
    out = {"note": __doc__.strip().splitlines()[0], "readout": READOUT, "calibration": [], "study": []}
    cvar = next(x for x in ns["CIRCUITS"] if x["name"] == TARGET[0])
    plain = ns["build_circuit"](cvar)
    for rate in RATES:
        a = score(ns, counts_of(plain, rate, 40000, 1, "density_matrix"), cvar)
        out["calibration"].append({"rate": rate, "p_optimum": a["p_optimum"]})
        print(f"calibration: two-qubit depolarizing {rate:.2%}: plain {TARGET[0]} P(opt) {a['p_optimum']:.4f} (IonQ run: {TARGET[1]:.4f})", flush=True)
    xs = np.array([c["rate"] for c in out["calibration"]])
    ys = np.log([max(c["p_optimum"], 1e-6) for c in out["calibration"]])
    calibrated = float(np.interp(np.log(TARGET[1]), ys[::-1], xs[::-1]))
    out["calibrated_rate"] = calibrated
    print(f"calibrated two-qubit depolarizing rate: {calibrated:.4%}", flush=True)
    for rate, label in ((0.005, "uncalibrated 0.5 %"), (calibrated, "calibrated to the IonQ run")):
        for p in (1, 2, 3, 4, 5, 6):
            spec = ns["lr_spec"](p)
            plain = ns["build_circuit"](spec)
            ice, info, _ = ns["build_iceberg"](spec, 1)
            started = time.time()
            a_plain = score(ns, counts_of(plain, rate, 20000, 2, "density_matrix"), spec)
            raw = counts_of(ice, rate, 4000, 3, "statevector")
            logical, tally = ns["iceberg_decode"](raw, info)
            a_ice = score(ns, logical, spec)
            kept = tally["accepted"] / tally["shots"]
            row = {"noise": label, "rate": rate, "layers": p, "noiseless": spec["p_optimum_noiseless"],
                   "plain": {"p_optimum": a_plain["p_optimum"], "quality": a_plain["quality"], **ns["resources"](plain)},
                   "iceberg": {"kept_share": kept, "p_optimum_kept": a_ice["p_optimum"] if a_ice else 0.0,
                               "p_optimum_per_shot_sent": (a_ice["p_optimum"] if a_ice else 0.0) * kept, "quality_kept": a_ice["quality"] if a_ice else None,
                               "tally": tally, **ns["resources"](ice)},
                   "seconds": round(time.time() - started, 1)}
            out["study"].append(row)
            print(f"{label}, p={p}: noiseless {row['noiseless']:.4f} | plain {a_plain['p_optimum']:.4f} | Iceberg kept {kept:.1%}, "
                  f"P(opt|kept) {row['iceberg']['p_optimum_kept']:.4f}, per shot sent {row['iceberg']['p_optimum_per_shot_sent']:.4f} ({row['seconds']} s)", flush=True)
            OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("saved", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
