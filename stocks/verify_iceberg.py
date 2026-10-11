"""Verify the Iceberg error-detection circuits in qollab/main.py (the exact code that runs on Qollab).

    python -m stocks.verify_iceberg            # writes results/iceberg_verification.json

1. Noiseless equivalence: with no noise every shot passes every check, and the decoded logical distribution equals the
   plain 12-qubit QAOA distribution (total variation distance ~ 1e-15), for several circuits, depths and check counts.
   The shipped decoder (iceberg_decode) is tested on the exact distribution written as counts.
2. Single-fault injection: after every gate, every single-qubit Pauli (1-qubit gates) or every one of the 15 two-qubit
   Paulis (2-qubit gates), plus a bit flip on every qubit at the start. Each fault is classified as
   detected (no shot passes), harmless (shots pass and the decoded distribution is unchanged) or harmful (shots pass
   and the distribution changes). The Iceberg code has distance 2, so XX, YY and ZZ faults on a gate's own pair are
   expected to be the only harmful ones.
3. Readout: a flipped bit on any single qubit must make every shot fail a check.
4. Transpiling with optimization_level=1 (as main.py does before sending) keeps 1.
Exact statevector simulation in NumPy (complex128); faults are run in parallel.
"""
import json
import multiprocessing as mp
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "iceberg_verification.json"
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0]).astype(complex)}


def load_main():
    code = (ROOT / "qollab" / "main.py").read_text()
    code = code[:code.index('if sys.platform != "emscripten":')]
    ns = {"__name__": "qollab_main"}
    exec(compile(code, "qollab/main.py", "exec"), ns)
    return ns


_CACHE = {}


def matrix(op):
    key = (op.name, tuple(float(p) for p in op.params))
    if key not in _CACHE:
        _CACHE[key] = Operator(op).data
    return _CACHE[key]


def apply(psi, n, u, qubits):
    """Apply a 1- or 2-qubit matrix (Qiskit convention: qargs[0] is the least significant) to qubits."""
    psi = psi.reshape([2] * n)
    if len(qubits) == 1:
        axis = n - 1 - qubits[0]
        psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [axis])), 0, axis)
    else:
        a0, a1 = n - 1 - qubits[0], n - 1 - qubits[1]
        out = np.tensordot(u.reshape(2, 2, 2, 2), psi, axes=([2, 3], [a1, a0]))
        psi = np.moveaxis(out, [0, 1], [a1, a0])
    return psi.reshape(-1)


def gate_list(qc):
    return [(inst.operation, [qc.find_bit(q).index for q in inst.qubits]) for inst in qc.data if inst.operation.name not in ("measure", "barrier")]


def run(gates, n, start=0, psi=None):
    if psi is None:
        psi = np.zeros(1 << n, complex)
        psi[0] = 1.0
    for op, qubits in gates[start:]:
        psi = apply(psi, n, matrix(op), qubits)
    return psi


def decode_vector(prob, info, n):
    """Probability that a shot passes every check, and the decoded logical distribution of the passing shots."""
    idx = np.arange(1 << n, dtype=np.int64)
    bit = lambda q: (idx >> q) & 1
    k, b = info["k"], info["b"]
    ok = bit(info["flag"]) == 0
    for a1, a2 in info["ancillas"]:
        ok &= (bit(a1) == 0) & (bit(a2) == 0)
    parity = np.zeros_like(idx)
    for q in range(k + 2):
        parity ^= bit(q)
    ok &= parity == 0
    logical = np.zeros_like(idx)
    for i in range(k):
        logical |= (bit(i) ^ bit(b)) << i
    dist = np.bincount(logical[ok], weights=prob[ok], minlength=1 << k)
    return float(prob[ok].sum()), dist


def tvd(a, b):
    return 0.5 * float(np.abs(a / max(a.sum(), 1e-300) - b / max(b.sum(), 1e-300)).sum())


def classify(gate):
    op, qubits = gate
    k = STATE["info"]["k"]
    b, flag = STATE["info"]["b"], STATE["info"]["flag"]
    if any(q > flag for q in qubits):
        return "syndrome check"
    if flag in qubits:
        return "flag (state-preparation check)"
    if op.name == "rxx":
        return "logical mixer rxx(t, i)"
    if op.name == "rzz":
        return "logical field rzz(i, b)" if b in qubits else "logical coupling rzz(i, j)"
    return "state preparation (GHZ + H)"


STATE = {}


def fault_job(task):
    position, labels, qubits = task
    gates, n, ideal = STATE["gates"], STATE["n"], STATE["ideal"]
    psi = STATE["prefix"][position].copy()
    for label, q in zip(labels, qubits):
        if label != "I":
            psi = apply(psi, n, PAULI[label], [q])
    psi = run(gates, n, start=position, psi=psi)
    p_pass, dist = decode_vector(np.abs(psi) ** 2, STATE["info"], n)
    if p_pass < 1e-9:
        verdict = "detected"
    elif tvd(dist, ideal) < 1e-7:
        verdict = "harmless"
    else:
        verdict = "harmful"
    return {"position": position, "pauli": "".join(labels), "qubits": qubits, "p_pass": p_pass,
            "tvd": tvd(dist, ideal) if p_pass >= 1e-9 else None, "verdict": verdict}


def fault_injection(ns, spec, checks, workers=8):
    plain = ns["build_circuit"](spec)
    ice, info, _ = ns["build_iceberg"](spec, checks)
    n = ice.num_qubits
    gates = gate_list(ice)
    ideal = np.abs(run(gate_list(plain), plain.num_qubits)) ** 2
    prefix = [np.zeros(1 << n, complex)]
    prefix[0][0] = 1.0
    for op, qubits in gates:
        prefix.append(apply(prefix[-1], n, matrix(op), qubits))
    STATE.update(gates=gates, n=n, ideal=ideal, info=info, prefix=prefix)
    tasks = [(0, ("X",), [q]) for q in range(n)]   # a wrong initial bit on any qubit
    kinds = {}
    for g, (op, qubits) in enumerate(gates):
        position = g + 1   # fault right after gate g
        labels_set = [(p,) for p in "XYZ"] if len(qubits) == 1 else [pq for pq in product("IXYZ", repeat=2) if pq != ("I", "I")]
        for labels in labels_set:
            tasks.append((position, labels, qubits))
            kinds[(position, labels)] = classify((op, qubits))
    results = []
    with mp.get_context("fork").Pool(workers) as pool:
        for r in pool.imap_unordered(fault_job, tasks, chunksize=8):
            results.append(r)
            if len(results) % 500 == 0:
                print(f"    {len(results)} of {len(tasks)} faults done", flush=True)
    summary = {}
    for r in results:
        where = "initial bit flip" if r["position"] == 0 else kinds[(r["position"], tuple(r["pauli"]))]
        weight = sum(1 for c in r["pauli"] if c != "I")
        bucket = f"{where} | weight {weight}"
        entry = summary.setdefault(bucket, {"detected": 0, "harmless": 0, "harmful": 0, "harmful_paulis": {}})
        entry[r["verdict"]] += 1
        if r["verdict"] == "harmful":
            entry["harmful_paulis"][r["pauli"]] = entry["harmful_paulis"].get(r["pauli"], 0) + 1
    totals = {v: sum(1 for r in results if r["verdict"] == v) for v in ("detected", "harmless", "harmful")}
    partial = [r for r in results if 1e-9 <= r["p_pass"] < 1 - 1e-9]
    return {"circuit": spec["name"], "checks": info["checks"], "qubits": n, "gates": len(gates), "faults": len(results), "totals": totals,
            "faults_with_partial_detection": len(partial), "by_location": summary}


def main():
    ns = load_main()
    out = {"note": "exact statevector verification of the Iceberg circuits in qollab/main.py", "equivalence": [], "readout": [], "transpile": [], "faults": []}
    specs = [ns["lr_spec"](1), ns["lr_spec"](2), ns["lr_spec"](3), next(x for x in ns["CIRCUITS"] if x["name"] == "B CVaR 0.1 p=3")]
    for spec in specs:
        plain = ns["build_circuit"](spec)
        ideal = np.abs(run(gate_list(plain), plain.num_qubits)) ** 2
        for checks in (1, 2, 3):
            if checks > len(spec["angles"]["gammas"]):
                continue
            ice, info, _ = ns["build_iceberg"](spec, checks)
            n = ice.num_qubits
            prob = np.abs(run(gate_list(ice), n)) ** 2
            p_pass, dist = decode_vector(prob, info, n)
            # the shipped decoder, fed the exact distribution as counts
            counts = {format(i, "0%db" % n): int(round(v * 1e12)) for i, v in enumerate(prob) if v > 1e-14}
            logical, tally = ns["iceberg_decode"](counts, info)
            decoded = np.zeros(1 << info["k"])
            for key, c in logical.items():
                decoded[int(key, 2)] += c
            flips = []
            for q in range(n):   # one flipped readout bit on qubit q in every shot
                flipped = np.zeros_like(prob)
                flipped[np.arange(1 << n) ^ (1 << q)] = prob
                flips.append(decode_vector(flipped, info, n)[0])
            row = {"circuit": spec["name"], "checks": info["checks"], "qubits": n, "two_qubit_gates": ns["resources"](ice)["two_qubit_gates"],
                   "pass_probability": p_pass, "tvd_vs_plain": tvd(dist, ideal), "decoder_tvd_vs_plain": tvd(decoded, ideal),
                   "decoder_kept_share": tally["accepted"] / tally["shots"], "max_pass_after_one_readout_flip": max(flips)}
            out["equivalence"].append(row)
            print("equivalence", row, flush=True)
        ice, info, _ = ns["build_iceberg"](spec, 1)
        compiled = transpile(ice, basis_gates=["rzz", "rxx", "cx", "rz", "rx", "ry", "h", "sx", "x"], optimization_level=1, seed_transpiler=1)
        prob = np.abs(run(gate_list(compiled), compiled.num_qubits)) ** 2
        p_pass, dist = decode_vector(prob, info, compiled.num_qubits)
        row = {"circuit": spec["name"], "two_qubit_gates_before": ns["resources"](ice)["two_qubit_gates"],
               "two_qubit_gates_after": sum(1 for inst in compiled.data if inst.operation.num_qubits == 2),
               "pass_probability": p_pass, "tvd_vs_plain": tvd(dist, ideal)}
        out["transpile"].append(row)
        print("transpile", row, flush=True)
    for layers, checks in ((1, 1), (2, 2)):
        started = time.time()
        result = fault_injection(ns, ns["lr_spec"](layers), checks)
        result["seconds"] = round(time.time() - started, 1)
        out["faults"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "by_location"}), flush=True)
        for bucket, entry in sorted(result["by_location"].items()):
            print("   ", bucket, {k: v for k, v in entry.items() if k != "harmful_paulis" or v}, flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("saved", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
