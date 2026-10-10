"""Gate counts and Aer wall time for one Grover circuit (j = 10, 1000 shots) at m = 10 and 12.

Uses a random cost table with k = 3 feasibility so it does not depend on data/champaign.json.
Run from the repo root: python tools/time_grover.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qopt.cost import feasible_mask  # noqa: E402
from qopt.search import grover_circuit  # noqa: E402

J, SHOTS = 10, 1000
sim = AerSimulator(method="statevector")

for m in (10, 12):
    cost = np.random.default_rng(0).uniform(10, 100, 2 ** m)
    cost[~feasible_mask(m, 3)] = np.inf
    T = np.sort(cost)[0]
    circuit = grover_circuit(cost, T, J)

    t0 = time.perf_counter()
    tc = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    t_transpile = time.perf_counter() - t0
    ops = dict(tc.count_ops())

    t0 = time.perf_counter()
    sim.run(tc, shots=SHOTS, seed_simulator=1).result()
    t_run = time.perf_counter() - t0

    print(f"m={m} j={J}: cx={ops.get('cx', 0)} total_ops={sum(ops.values())} depth={tc.depth()} "
          f"transpile={t_transpile:.2f}s aer_run={t_run:.2f}s ({SHOTS} shots)")
    print(f"  ops: {ops}")
