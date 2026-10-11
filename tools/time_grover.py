"""Qubits, gate counts and wall time of one Grover iteration on the shipped instance: data/champaign.json,
default params, k = 4 stores opened among m = 8, 10, 12 candidate locations, threshold = the minimum (M = 1).

One Grover iteration = the in-circuit cost oracle (compute, phase on the sign bit, uncompute) and the
diffusion (Dicke preparation twice and a reflection). Counts are after transpiling to rz, sx, x, cx at
optimization level 1, and at level 0 (what the bridge ships, translation only). Wall time: a j = 1 circuit
minus a j = 0 circuit, one shot, on qiskit's BasicSimulator (Qollab's built-in simulator) and on Aer.
Run from the repo root: python tools/time_grover.py [m ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qopt.cost import DEFAULTS, model, model_table  # noqa: E402
from qopt.search import (BASIS, _diffusion, accumulator_bits, dicke_circuit, feasible_count, grover_circuit,  # noqa: E402
                         oracle, to_basis)

BASIC_MAX_QUBITS = 24
aer = AerSimulator(method="statevector")
basic = BasicSimulator()
data = json.loads((Path(__file__).resolve().parents[1] / "data" / "champaign.json").read_text(encoding="utf-8"))
k = DEFAULTS["k"]


def counts(circuit, level):
    ops = dict(transpile(circuit, basis_gates=BASIS, optimization_level=level).count_ops())
    ops.pop("measure", None)
    return ops.get("cx", 0), sum(ops.values())


def seconds(backend, circuit):
    t0 = time.perf_counter()
    backend.run(circuit, shots=1).result()
    return time.perf_counter() - t0


for m in [int(a) for a in sys.argv[1:]] or [8, 10, 12]:
    mdl = model(data, {}, m)
    T = float(np.min(model_table(mdl)))     # only to pick the threshold for the timing
    n = accumulator_bits(mdl, int(round(T * mdl["scale"])) + 1)
    o = oracle(mdl, T)
    iteration = QuantumCircuit(o.num_qubits)
    iteration.compose(o, inplace=True)
    iteration.append(_diffusion(dicke_circuit(m, k)), range(m))
    c1, o1 = counts(iteration, 1)
    c0, o0 = counts(iteration, 0)
    j0, j1 = to_basis([grover_circuit(mdl, T, 0), grover_circuit(mdl, T, 1)])
    line = (f"m={m} k={k} placements={feasible_count(model_table(mdl))} qubits={o.num_qubits} "
            f"(sites {m} + accumulator {n} + AND 1), pairs={len(mdl['B'])} | per iteration: level 1 cx={c1} ops={o1}, "
            f"level 0 cx={c0} ops={o0}")
    t_aer = seconds(aer, j1) - seconds(aer, j0)
    line += f" | Aer {t_aer:.2f}s"
    if o.num_qubits <= BASIC_MAX_QUBITS:
        t_basic = seconds(basic, j1) - seconds(basic, j0)
        line += f" | BasicSimulator {t_basic:.1f}s per iteration"
    print(line, flush=True)
