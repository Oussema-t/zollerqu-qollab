"""Gate counts and Aer wall time for the shipped instance: data/champaign.json, default params,
m = 10 and 12, one Grover circuit with j = 10 iterations, 1000 shots, threshold = the minimum (M = 1).

Counts are after transpiling to rz, sx, x, cx at optimization level 1.
Run from the repo root: python tools/time_grover.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qopt.cost import DEFAULTS, cost_table  # noqa: E402
from qopt.search import _diffusion, dicke_circuit, feasible_count, grover_circuit, oracle  # noqa: E402

J, SHOTS = 10, 1000
BASIS = ["rz", "sx", "x", "cx"]
sim = AerSimulator(method="statevector")
data = json.loads((Path(__file__).resolve().parents[1] / "data" / "champaign.json").read_text(encoding="utf-8"))
k = DEFAULTS["k"]


def counts(circuit):
    ops = dict(transpile(circuit, basis_gates=BASIS, optimization_level=1).count_ops())
    ops.pop("measure", None)
    return ops.get("cx", 0), sum(ops.values())


for m in (10, 12):
    cost = cost_table(data, {}, m)
    T = float(np.min(cost))
    circuit = grover_circuit(cost, T, J)

    prep = dicke_circuit(m, k)
    iteration = QuantumCircuit(m)
    iteration.append(oracle(cost, T), range(m))
    iteration.append(_diffusion(prep), range(m))

    t0 = time.perf_counter()
    tc = transpile(circuit, basis_gates=BASIS, optimization_level=1)
    t_transpile = time.perf_counter() - t0
    ops = dict(tc.count_ops())
    t0 = time.perf_counter()
    sim.run(tc, shots=SHOTS, seed_simulator=1).result()
    t_run = time.perf_counter() - t0

    it_cx, it_ops = counts(iteration)
    dk_cx, dk_ops = counts(prep)
    n_ops = sum(v for key, v in ops.items() if key != "measure")
    print(f"m={m} k={k} N={feasible_count(cost)} j={J}: full circuit cx={ops.get('cx', 0)} ops={n_ops} "
          f"depth={tc.depth()} | per iteration cx={it_cx} ops={it_ops} | dicke alone cx={dk_cx} ops={dk_ops} | "
          f"transpile={t_transpile:.2f}s aer_run={t_run:.2f}s ({SHOTS} shots)")
    print(f"  full circuit ops: {ops}")
