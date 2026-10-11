"""Iceberg error DETECTION for our 12-qubit mesh QAOA: build, verify, predict, and write qollab/iceberg_circuits.json.

Code: the Iceberg [[k+2, k, 2]] code (Self, Benedetti, Amaro, Nature Physics 20, 219 (2024)). k = 12 logical qubits (the mesh
regions) live on k + 2 = 14 data qubits: the 12 region qubits plus a 'top' qubit t and a 'bottom' qubit b. Stabilisers
S_X = X^(x14) and S_Z = Z^(x14); logical operators X_i = X_t X_i and Z_i = Z_i Z_b. So the QAOA gates stay physical
two-qubit rotations that commute with both stabilisers:
    cost   exp(-i g J_ij Z_i Z_j)    -> rzz(2 g J_ij, i, j)        (identical to the plain circuit)
    field  exp(-i g h_i Z_i)         -> rzz(2 g h_i, i, b)
    mixer  exp(+i b X_i)             -> rxx(-2 b, t, i)
    start  logical |+>^12            -> (|+>^14 + |->^14)/sqrt 2 = H on every data qubit after a GHZ state (H(t), CNOT t -> all)
Syndrome checks (Qollab / IonQ rules: no reset, no gate after a measurement): every check uses FRESH ancillas, and every qubit is
measured once, at the very end (an ancilla that is never touched again can be measured late without changing anything):
    X check: ancilla H, CNOT(ancilla -> each data qubit), H  (reads 0 if S_X = +1)
    Z check: ancilla |0>, CNOT(each data qubit -> ancilla)   (reads 0 if S_Z = +1)
    final Z check: parity of the 14 data bits read out at the end (free, no ancilla).
A shot is ACCEPTED if every check reads 0; its regions are decoded as bit_i XOR bit_b.
What it detects: any single Pauli error on a data qubit between the encoding and the final checks (it anticommutes with S_X or
S_Z, and every logical gate commutes with both, so the anticommutation survives to the final checks). What it does NOT do:
correct errors (distance 2), and the encoding / syndrome circuits are the simple versions, not the flagged fault-tolerant ones of
Self et al., so some single faults there (e.g. mid-fan-out in the GHZ encoding) can pass undetected. This is error detection
with post-selection, not error correction.

Verification (all must pass before anything is written):
  1. noiseless: the decoded Iceberg distribution equals the plain QAOA distribution (max difference < 1e-9) and every shot is
     accepted;
  2. single-fault injection: X, Y or Z inserted on each data qubit after every logical layer (and after the encoding) is caught
     by the checks in 100 % of cases;
  3. approximate local noise model (Aer, NOT IonQ: 0.5 % 2q depolarising, 0.05 % 1q, 0.5 % readout): plain vs Iceberg success.
Logical circuit: the standard QAOA angles tuned at 12 qubits on calibration meshes (results/qaoa_tuning/
tuning_n12_large_calibration.json), mesh case 2, both scenarios, p = 1, 2, 3. Choice for Qollab (rule fixed before any IonQ run):
the scenario and depth with the most near-optimal shots per SUBMITTED Iceberg shot under the local noise model.
usage: python scripts/iceberg_build.py   ->  qollab/iceberg_circuits.json, results/iceberg/local_check.json, figures
"""
import json
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iceberg"
OUT.mkdir(parents=True, exist_ok=True)
SELECTION = json.loads((ROOT / "qollab" / "selected_circuits.json").read_text())
TUNING = json.loads((ROOT / "results" / "qaoa_tuning" / "tuning_n12_large_calibration.json").read_text())
K = 12                                   # logical qubits = mesh regions
TOP, BOTTOM = K, K + 1                   # data qubits 0..11 = regions, 12 = t, 13 = b
DEPTHS = (1, 2, 3)
NOISE = dict(two_qubit_depolarizing=0.005, one_qubit_depolarizing=0.0005, readout_flip=0.005)   # as scripts/ionq_selection_evaluate.py
SHOTS_LOCAL = 2000                       # noisy trajectories per circuit (Aer): enough for acceptance and P(near) to ~1 %
WITHIN = 0.05


def tuned_angles(scenario, p):
    rows = [r for r in TUNING["scenarios"][scenario]["rows"] if r["p"] == p and "warm" not in r["method"]]
    best = max(rows, key=lambda r: r["calibration_p_near"])
    return best["method"], np.array(best["angles"], float)


def instance(scenario):
    inst = SELECTION["instances"][scenario]
    J, h = np.asarray(inst["J"], float), np.asarray(inst["h"], float)
    return inst, np.triu(J, 1) / inst["scale"], h / inst["scale"]


def round_robin_pairs(n):
    """Same pair order as the notebook and qollab/main.py (circle method)."""
    players = list(range(n)) + ([None] if n % 2 else [])
    m, rounds = len(players), []
    for _ in range(m - 1):
        rounds.append([(min(a, b), max(a, b)) for a, b in zip(players[: m // 2], players[::-1][: m // 2]) if a is not None and b is not None])
        players = [players[0], players[-1]] + players[1:-1]
    return rounds


def plain_circuit(J, h, angles, measure=True):
    p = len(angles) // 2
    qc = QuantumCircuit(K, K if measure else 0)
    qc.h(range(K))
    for gamma, beta in zip(angles[:p], angles[p:]):
        for i, j in [pair for rnd in round_robin_pairs(K) for pair in rnd]:
            qc.rzz(float(2 * gamma * J[i, j]), i, j)
        for q in range(K):
            if h[q] != 0:
                qc.rz(float(2 * gamma * h[q]), q)
        for q in range(K):
            qc.rx(float(-2 * beta), q)
    if measure:
        qc.measure(range(K), range(K))
    return qc


def iceberg_circuit(J, h, angles, check_after=(), measure=True, faults=None):
    """Iceberg version of plain_circuit. check_after: logical layers after which a mid-circuit X and Z check is made (fresh
    ancillas). The final X check always comes last. faults: {position: (pauli, qubit)}, position 0 = after encoding,
    k = after logical layer k (used only by the verification)."""
    p = len(angles) // 2
    n_anc = 2 * len(check_after) + 1
    data, n_data = list(range(K + 2)), K + 2
    qc = QuantumCircuit(n_data + n_anc, n_data + n_anc if measure else 0)
    ancillas = iter(range(n_data, n_data + n_anc))

    def x_check():
        a = next(ancillas)
        qc.h(a)
        for q in data:
            qc.cx(a, q)
        qc.h(a)

    def z_check():
        a = next(ancillas)
        for q in data:
            qc.cx(q, a)

    def fault(position):
        if faults and position in faults:
            pauli, qubit = faults[position]
            getattr(qc, pauli.lower())(qubit)

    qc.h(TOP)                                              # encoding: GHZ on the 14 data qubits, then H on all = logical |+>^12
    for q in data:
        if q != TOP:
            qc.cx(TOP, q)
    qc.h(data)
    fault(0)
    for layer, (gamma, beta) in enumerate(zip(angles[:p], angles[p:]), start=1):
        for i, j in [pair for rnd in round_robin_pairs(K) for pair in rnd]:
            qc.rzz(float(2 * gamma * J[i, j]), i, j)
        for q in range(K):
            if h[q] != 0:
                qc.rzz(float(2 * gamma * h[q]), q, BOTTOM)  # logical Z_q = Z_q Z_b
        for q in range(K):
            qc.rxx(float(-2 * beta), TOP, q)                # logical X_q = X_t X_q
        fault(layer)
        if layer in check_after:
            x_check()
            z_check()
    x_check()                                              # final X check; the final Z check is the parity of the data readout
    if measure:
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    return qc


def decode(key, n_data=K + 2):
    """Count key (rightmost = clbit 0) -> (accepted, region bits). Accepted: all ancillas 0 and even parity of the 14 data bits."""
    bits = [int(c) for c in key.replace(" ", "")[::-1]]
    data, anc = bits[:n_data], bits[n_data:]
    accepted = sum(data) % 2 == 0 and not any(anc)
    return accepted, [b ^ data[BOTTOM] for b in data[:K]]


def region_state(bits):
    return int(sum(b << r for r, b in enumerate(bits)))


def cost_table(inst):
    z = 1.0 - 2.0 * ((np.arange(2**K)[:, None] >> np.arange(K)) & 1)
    J, h = np.asarray(inst["J"], float), np.asarray(inst["h"], float)
    return np.einsum("ki,ij,kj->k", z, J, z) + z @ h + inst["const"]


def logical_distribution_from_iceberg(qc):
    """Exact decoded distribution and acceptance of an Iceberg circuit (no noise): all qubits measured at the end
    (vectorised version of decode())."""
    probs = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
    index = np.flatnonzero(probs > 1e-14)
    data = (index[:, None] >> np.arange(K + 2)) & 1
    ok = (data.sum(axis=1) % 2 == 0) & ((index >> (K + 2)) == 0)
    regions = data[:, :K] ^ data[:, BOTTOM:BOTTOM + 1]
    states = regions @ (1 << np.arange(K))
    out = np.bincount(states[ok], weights=probs[index][ok], minlength=2**K)
    return out, float(probs[index][ok].sum())


def noise_model():
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(NOISE["two_qubit_depolarizing"], 2), ["rzz", "rxx", "cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(NOISE["one_qubit_depolarizing"], 1), ["rx", "rz", "h", "x", "y", "z"])
    f = NOISE["readout_flip"]
    model.add_all_qubit_readout_error(ReadoutError([[1 - f, f], [f, 1 - f]]))
    return model


def grade(counts, inst, costs, iceberg):
    optimal, e_min = set(inst["optimal_states"]), inst["e_min"]
    total = sum(counts.values())
    kept = opt = near = 0
    for key, c in counts.items():
        if iceberg:
            ok, regions = decode(key)
            if not ok:
                continue
        else:
            regions = [int(b) for b in key.replace(" ", "")[::-1][:K]]
        state = region_state(regions)
        kept += c
        opt += c * (state in optimal)
        near += c * (costs[state] <= (1 + WITHIN) * e_min + 1e-12)
    return dict(shots=total, accepted=kept, acceptance=kept / total, p_opt=opt / max(kept, 1), p_near=near / max(kept, 1),
                opt_per_submitted_shot=opt / total, near_per_submitted_shot=near / total)


def size(qc):
    ops = qc.count_ops()
    return dict(qubits=qc.num_qubits, two_qubit=int(sum(v for k, v in ops.items() if k in ("rzz", "rxx", "cx"))),
                depth=int(qc.depth()), depth_two_qubit=int(qc.depth(lambda ins: ins.operation.num_qubits == 2)), ops={k: int(v) for k, v in ops.items()})


if __name__ == "__main__":
    t0 = time.time()
    sim = AerSimulator(method="statevector", noise_model=noise_model(), seed_simulator=2026)
    report, candidates = dict(code="Iceberg [[14, 12, 2]] (error detection, post-selection)", noise_model_local=NOISE, checks={}, runs=[]), []
    for scenario in SELECTION["instances"]:
        inst, J, h = instance(scenario)
        costs = cost_table(inst)
        assert abs(costs.min() - inst["e_min"]) < 1e-9, "embedded optimum does not match brute force"
        for p in DEPTHS:
            method, angles = tuned_angles(scenario, p)
            plain = plain_circuit(J, h, angles)
            ideal = Statevector(plain_circuit(J, h, angles, measure=False)).probabilities()
            for checks in ((),):        # final checks only (mid-circuit checks: check_after=..., not run: 17-19 qubits, slow locally)
                ice = iceberg_circuit(J, h, angles, check_after=checks)
                # 1. noiseless equivalence and full acceptance
                dist, acc = logical_distribution_from_iceberg(ice)
                diff = float(np.abs(dist - ideal).max())
                assert diff < 1e-9 and abs(acc - 1) < 1e-9, f"noiseless Iceberg differs from plain QAOA ({diff}, acceptance {acc})"
                # 2. single faults on data qubits after the encoding and after every layer: all detected
                tried = caught = 0
                for position in range(p + 1):
                    for qubit in range(K + 2):
                        for pauli in "XYZ":
                            faulty = iceberg_circuit(J, h, angles, check_after=checks, faults={position: (pauli, qubit)})
                            _, acc_f = logical_distribution_from_iceberg(faulty)
                            tried += 1
                            caught += acc_f < 1e-9
                assert caught == tried, f"{tried - caught} of {tried} injected single faults were not detected"
                key = f"{scenario} p={p} checks={list(checks)}"
                report["checks"][key] = dict(noiseless_max_difference=diff, noiseless_acceptance=acc, single_faults_injected=tried, single_faults_detected=caught)
                # 3. approximate local noise model
                t1 = time.time()
                counts_plain = sim.run(transpile(plain, sim, optimization_level=0), shots=SHOTS_LOCAL).result().get_counts()
                counts_ice = sim.run(transpile(ice, sim, optimization_level=0), shots=SHOTS_LOCAL).result().get_counts()
                row = dict(scenario=scenario, p=p, angles_from=method, angles=angles.tolist(), checks_after_layers=list(checks),
                           noiseless=dict(p_opt=float(ideal[inst["optimal_states"]].sum()),
                                          p_near=float(ideal[costs <= (1 + WITHIN) * inst["e_min"] + 1e-12].sum())),
                           plain=dict(size=size(plain), local_noise=grade(counts_plain, inst, costs, False)),
                           iceberg=dict(size=size(ice), local_noise=grade(counts_ice, inst, costs, True)), seconds=time.time() - t1)
                report["runs"].append(row)
                pl, ic = row["plain"]["local_noise"], row["iceberg"]["local_noise"]
                print(f"[{time.time() - t0:5.0f} s] {key}: noiseless P(opt) {row['noiseless']['p_opt']:.4f} | plain P(opt) {pl['p_opt']:.4f} "
                      f"P(near) {pl['p_near']:.4f} | Iceberg accepted {ic['acceptance']:.3f}, P(opt|acc) {ic['p_opt']:.4f}, P(near|acc) {ic['p_near']:.4f}, "
                      f"near per submitted shot {ic['near_per_submitted_shot']:.4f} | qubits {row['plain']['size']['qubits']} vs {row['iceberg']['size']['qubits']}, "
                      f"2q {row['plain']['size']['two_qubit']} vs {row['iceberg']['size']['two_qubit']}; faults caught {caught}/{tried}", flush=True)
    # choice for Qollab (rule in the docstring): most near-optimal shots per SUBMITTED Iceberg shot under the local noise model
    best = max(report["runs"], key=lambda r: r["iceberg"]["local_noise"]["near_per_submitted_shot"])
    report["chosen"] = dict(scenario=best["scenario"], p=best["p"], rule="max near-optimal shots per submitted Iceberg shot (local noise model)")
    (OUT / "local_check.json").write_text(json.dumps(report, indent=1))
    print(f"saved {OUT / 'local_check.json'}; chosen: {best['scenario']}, p = {best['p']}")
    inst, J, h = instance(best["scenario"])
    tag = "equal" if abs(inst["alpha"] - 0.5) < 1e-9 else "cpu_gpu"
    plain_key, ice_key = f"qaoa_plain_p{best['p']}_{tag}", f"qaoa_iceberg_p{best['p']}_{tag}"
    p_best, angles = best["p"], best["angles"]
    common = dict(p=p_best, qubit_to_region=list(range(K)), fixed_regions={}, gammas=angles[:p_best], betas=angles[p_best:], start="plus",
                  mixer="x", J_scaled=J.tolist(), h_scaled=h.tolist(), zz_pairs=[[list(pair) for pair in rnd] for rnd in round_robin_pairs(K)],
                  optimiser="none here: angles tuned offline on calibration meshes (" + best["angles_from"] + ")")
    circuits = {}
    for key, kind, partner in ((plain_key, "plain", ice_key), (ice_key, "iceberg", plain_key)):
        size_ = best[kind]["size"]
        local = best[kind]["local_noise"]
        spec = dict(common, kind="qaoa", variant="standard QAOA, tuned angles", n_qubits=K) if kind == "plain" else dict(
            common, kind="iceberg", variant="standard QAOA, tuned angles, Iceberg [[14,12,2]] error detection", n_qubits=size_["qubits"],
            n_logical=K, top=TOP, bottom=BOTTOM, check_after=best["checks_after_layers"])
        circuits[key] = dict(
            name=("QAOA p = %d, no error detection" if kind == "plain" else "QAOA p = %d + Iceberg error detection") % p_best,
            label="plain QAOA (error-detection comparison)" if kind == "plain" else "Iceberg-encoded QAOA, post-selected",
            role="error-detection experiment: run both %s and %s" % (plain_key, ice_key), graded_under=[best["scenario"]], partner=partner,
            spec=spec, gates=dict(two_qubit=size_["two_qubit"], depth=size_["depth"], qubits=size_["qubits"]),
            predicted=dict(p_opt=best["noiseless"]["p_opt"], p_near=best["noiseless"]["p_near"],
                           crude_p_opt=best["noiseless"]["p_opt"] * (1 - NOISE["two_qubit_depolarizing"]) ** size_["two_qubit"],
                           aer_p_opt=local["p_opt"], aer_p_near=local["p_near"], acceptance_local=local["acceptance"],
                           near_per_submitted_local=local["near_per_submitted_shot"]))
    (ROOT / "qollab" / "iceberg_circuits.json").write_text(json.dumps(dict(source="scripts/iceberg_build.py", verification=report["checks"],
                                                                            chosen=report["chosen"], circuits=circuits), indent=1))
    print(f"saved qollab/iceberg_circuits.json: {plain_key}, {ice_key}")
