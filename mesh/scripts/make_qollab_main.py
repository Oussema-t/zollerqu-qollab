"""Generate the self-contained Qollab main.py from qollab/selected_circuits.json (circuits, angles and exact optima embedded).
Writes qollab/main.py and the notebook cell "Code for Qollab". The old multi-circuit batch main.py is kept in qollab/legacy/
(Qollab's IonQ simulator accepts ONE circuit per job, so the batch version cannot run there).
usage: python scripts/make_qollab_main.py
"""
import json
import pprint
import shutil
import subprocess
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
selection = json.loads((ROOT / "qollab" / "selected_circuits.json").read_text())


def key_for(block):
    spec, graded = block["spec"], block["graded_under"]
    if spec.get("variant") == "uniform control":
        return "control_p0"
    tag = "equal" if graded != "both scenarios" and abs(selection["instances"][graded]["alpha"] - 0.5) < 1e-9 else "cpu_gpu"
    variant = spec["variant"]
    if spec["kind"] == "varqite":
        short = "zy" if " ZY" in variant else "a_plus" if "A+" in variant else "b" if " B " in variant else "a_edges" if "graph edges" in variant else "a"
        return ("lemma_" if block["role"].startswith("optional extra") else "") + f"varqite_{short}_{tag}"
    family = ("warm" if variant.startswith("warm") else "cvar" if variant.startswith("CVaR") else "symred" if variant.startswith("symmetry")
              else "lr" if variant.startswith("LR") else "qaoa")
    optimiser = spec["optimiser"].split(":")[-1].strip().split()[0].lower().replace("-", "")
    return f"{family}_{optimiser}_p{spec['p']}_{tag}"


blocks = selection["circuits"] + selection.get("extras", [])
keys = {}
for block in blocks:
    key = key_for(block)
    assert key not in keys, f"duplicate key {key}"
    keys[key] = block
fallback_key = next(k for k, b in keys.items() if b["name"] == selection["fallback"]["name"] and b["graded_under"] == selection["fallback"]["graded_under"])
scenarios = {name: dict(alpha=inst["alpha"], J=inst["J"], h=inst["h"], const=inst["const"], work=inst["work"], e_min=inst["e_min"],
                        optimal_states=inst["optimal_states"], near_states=inst["near_states"], uniform_p_opt=inst["uniform_p_opt"],
                        target_share_p_opt=inst["target_share_p_opt"],
                        classical_median_evaluations={m: v["median"] for m, v in inst["classical_samples_to_solution"].items()})
             for name, inst in selection["instances"].items()}
def short_label(block):
    role, spec = block["role"], block["spec"]
    if spec.get("variant") == "uniform control":
        return "control: uniform guessing"
    if role.startswith("optional extra"):
        return "Lemma 1 demo: predicted to fail"
    label = "best #" + role.split("best #")[1][0] + " by crude estimate"
    return label + (" (starts from the classical answer)" if spec.get("seeded_by") else "")


circuits = {key: dict(name=b["name"], label=short_label(b), role=b["role"], graded_under=list(selection["instances"]) if b["graded_under"] == "both scenarios" else [b["graded_under"]],
                      spec=b["spec"], predicted=b["predicted"], gates=b["gates"]) for key, b in keys.items()}
iceberg_path = ROOT / "qollab" / "iceberg_circuits.json"         # written by scripts/iceberg_build.py (verified there)
if iceberg_path.exists():
    circuits.update(json.loads(iceberg_path.read_text())["circuits"])
commit = subprocess.run(["git", "-C", str(Path.home() / "zollerqu-mesh"), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"

TEMPLATE = '''# ======  COPY THIS WHOLE CELL INTO QOLLAB'S main.py  ======
"""ZollerQu: splitting a simulation mesh between two compute devices, 12 qubits (one per mesh region). Qollab x IonQ hackathon.
Error-detection experiment: run qaoa_plain_* and then qaoa_iceberg_* (same logical QAOA; the Iceberg version encodes the 12
logical qubits in 14 data qubits + 1 check ancilla and keeps only the shots that pass every check). Error DETECTION, not correction.

This effort is supported by Qollab & IonQ.

What this file does (press Run on Qollab):
  1. builds the circuit chosen in the Run dialog (`circuit`); all circuits were chosen offline from our 12-qubit benchmark,
     with their trained angles embedded below (no training or tuning happens here);
  2. sends that ONE circuit to the selected backend (Qollab's IonQ simulators accept one circuit per job);
     choose "IonQ Forte 1 (29q) noise model" in Select QPU (never the paid hardware entries);
  3. grades every measured shot against the exact optimum (known from checking all 4,096 splits) and prints the results
     next to the predictions (noiseless, crude estimate, approximate local noise model);
  4. prints the raw counts as JSON at the end, so the run can be saved and analysed again later.
Without a backend (local run, or no backend selected) it samples the exact NOISELESS probabilities instead, clearly labelled.

Problem: mesh case 2 (L-plate with a heat source), 12 regions. Device B gets the work share alpha: 0.5 for two equal devices,
0.8 for a CPU + a GPU assumed 4x faster. Cost = cut border + lambda x (balance error)^2 (Ising form, exact optimum known).
Conventions: qubit q = mesh region qubit_to_region[q]; bit 1 = region on device B (GPU / second CPU); in a Qiskit count key
qubit 0 is the RIGHTMOST character.
Generated by scripts/make_qollab_main.py from qollab/selected_circuits.json (repo commit __COMMIT__).
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

# Qollab draws the figures after main() returns, and its in-browser matplotlib then warns about float figure sizes
# (harmless). This filter stays active for the whole run, so those two warning lines no longer appear.
warnings.filterwarnings("ignore", message=r"The (width|height) parameter as float")

N_REGIONS = __N__
WITHIN = 0.05            # "near-optimal" = cost <= (1 + WITHIN) x optimum
BALANCE_TOLERANCE = 0.05
# the two device scenarios: Ising terms (cost = z J z + h z + const, z = 1 - 2 bit), region work, exact optimum, baselines
SCENARIOS = __SCENARIOS__
# the circuits that can be run (one per job), with their exact parameters and offline predictions
CIRCUITS = __CIRCUITS__
DEFAULT_CIRCUIT = "__DEFAULT__"


def build_iceberg(spec):
    """Iceberg [[k+2, k, 2]] error-detection version of the QAOA circuit (Self, Benedetti, Amaro, Nature Physics 2024).
    Data qubits 0..k-1 = regions, k = top t, k+1 = bottom b. Logical X_i = X_t X_i, logical Z_i = Z_i Z_b; stabilisers X^(k+2)
    and Z^(k+2). Start: GHZ, then H on every data qubit (= logical |+>^k). Cost rzz(2 g J_ij) as in the plain circuit; fields
    rzz(2 g h_i) on (i, b); mixer rxx(-2 b) on (t, i). Checks use FRESH ancillas (no reset) and every qubit is measured once,
    at the end (no gate after a measurement). The final Z check is the parity of the 14 data bits."""
    k, top, bottom = spec["n_logical"], spec["top"], spec["bottom"]
    data = list(range(k + 2))
    qc = QuantumCircuit(spec["n_qubits"])
    ancillas = iter(range(k + 2, spec["n_qubits"]))

    def x_check():                                  # ancilla reads 0 if X^(k+2) = +1
        a = next(ancillas)
        qc.h(a)
        for q in data:
            qc.cx(a, q)
        qc.h(a)

    def z_check():                                  # ancilla reads 0 if Z^(k+2) = +1
        a = next(ancillas)
        for q in data:
            qc.cx(q, a)

    qc.h(top)
    for q in data:
        if q != top:
            qc.cx(top, q)
    qc.h(data)
    J, h = np.asarray(spec["J_scaled"]), np.asarray(spec["h_scaled"])
    pairs = [pair for layer_round in spec["zz_pairs"] for pair in layer_round]
    for layer, (gamma, beta) in enumerate(zip(spec["gammas"], spec["betas"]), start=1):
        for i, j in pairs:
            if J[i][j] != 0:
                qc.rzz(float(2 * gamma * J[i][j]), i, j)
        for q in range(k):
            if h[q] != 0:
                qc.rzz(float(2 * gamma * h[q]), q, bottom)
        for q in range(k):
            qc.rxx(float(-2 * beta), top, q)
        if layer in spec["check_after"]:
            x_check()
            z_check()
    x_check()
    qc.measure_all()
    return qc


def iceberg_decode(key, spec):
    """Count key (rightmost = qubit 0) -> (accepted, logical bits): accepted if every ancilla reads 0 and the data parity is even;
    logical bit i = bit_i XOR bit_b."""
    bits = [int(b) for b in key.replace(" ", "")[::-1]]
    n_data = spec["n_logical"] + 2
    data, ancillas = bits[:n_data], bits[n_data:spec["n_qubits"]]
    return sum(data) % 2 == 0 and not any(ancillas), [data[q] ^ data[spec["bottom"]] for q in range(spec["n_logical"])]


def build_circuit(spec):
    """QAOA: H (or RY(theta) warm start); per layer rzz(2 gamma J_ij) over zz_pairs, rz(2 gamma h_i), then the mixer
    rx(-2 beta) = exp(+i beta X) (warm starts: ry(-s theta) rz(-2 beta) ry(s theta)). varQITE: H, then the trained gates."""
    if spec["kind"] == "iceberg":
        return build_iceberg(spec)
    n = spec["n_qubits"]
    qc = QuantumCircuit(n)
    qc.h(range(n)) if spec.get("start", "plus") == "plus" else None
    if spec["kind"] == "varqite":
        for kind, qubits, index in spec["gates"]:
            value = None if index is None else float(spec["params"][index])
            if kind == "ry":
                qc.ry(value, qubits[0])
            elif kind == "rx":
                qc.rx(value, qubits[0])
            elif kind == "cx":
                qc.cx(qubits[0], qubits[1])
            elif kind == "rzz":
                qc.rzz(value, qubits[0], qubits[1])
            elif kind == "zy":                          # exp(i theta Z_c Y_t): one native two-qubit gate
                control, target = qubits
                qc.sdg(target)
                qc.h(target)
                qc.rzz(-2.0 * value, control, target)
                qc.h(target)
                qc.s(target)
        qc.measure_all()
        return qc
    J, h, thetas = np.asarray(spec["J_scaled"]), np.asarray(spec["h_scaled"]), spec.get("thetas")
    if spec.get("start") == "ry":
        for q in range(n):
            qc.ry(float(thetas[q]), q)
    pairs = [pair for layer_round in spec["zz_pairs"] for pair in layer_round]
    for gamma, beta in zip(spec["gammas"], spec["betas"]):
        for i, j in pairs:
            if J[i][j] != 0:
                qc.rzz(float(2 * gamma * J[i][j]), i, j)
        for q in range(n):
            if h[q] != 0:
                qc.rz(float(2 * gamma * h[q]), q)
        for q in range(n):
            if spec["mixer"] == "x":
                qc.rx(float(-2 * beta), q)
            else:
                sign = -1.0 if spec["mixer"] == "warm B" else 1.0
                qc.ry(float(-sign * thetas[q]), q)
                qc.rz(float(-2 * beta), q)
                qc.ry(float(sign * thetas[q]), q)
    qc.measure_all()
    return qc


def region_bits(key, spec):
    """Count key -> bits of all regions (rightmost character = qubit 0; fixed regions are not measured; Iceberg: decoded)."""
    key = key.replace(" ", "")[-spec["n_qubits"]:]
    qubit_bits = iceberg_decode(key, spec)[1] if spec["kind"] == "iceberg" else [int(b) for b in key[::-1]]
    bits = [0] * N_REGIONS
    for region, value in spec.get("fixed_regions", {}).items():
        bits[int(region)] = int(value)
    for qubit, region in enumerate(spec["qubit_to_region"]):
        bits[region] = qubit_bits[qubit]
    return bits


def cost(bits, scenario):
    z = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    return float(z @ np.asarray(scenario["J"]) @ z + np.asarray(scenario["h"]) @ z + scenario["const"])


def grade(counts, spec, scenario):
    """Plain circuits: every shot counted (nothing filtered). Iceberg: only ACCEPTED shots are graded (post-selection); the
    acceptance and the successes per SUBMITTED shot are reported too. P(optimum), P(within 5 %), P(balanced), mean cost, ..."""
    optimal, e_min, work = set(scenario["optimal_states"]), scenario["e_min"], np.asarray(scenario["work"], dtype=float)
    submitted = sum(counts.values())
    if spec["kind"] == "iceberg":
        counts = {key: c for key, c in counts.items() if iceberg_decode(key, spec)[0]}
    shots = sum(counts.values())
    if shots == 0:
        return dict(shots=0, submitted=submitted, acceptance=0.0, p_opt=0.0, p_near=0.0, p_balanced=0.0, mean_cost=float("nan"),
                    distinct_splits=0, splits={}, opt_per_submitted_shot=0.0, near_per_submitted_shot=0.0)
    splits = {}
    for key, count in counts.items():
        bits = region_bits(key, spec)
        state = int(sum(b << r for r, b in enumerate(bits)))
        if state not in splits:
            splits[state] = dict(bits=bits, count=0, cost=cost(bits, scenario), share_b=float(work @ np.asarray(bits) / work.sum()))
        splits[state]["count"] += count
    p_opt = sum(s["count"] for k, s in splits.items() if k in optimal) / shots
    p_near = sum(s["count"] for s in splits.values() if s["cost"] <= (1 + WITHIN) * e_min + 1e-12) / shots
    p_bal = sum(s["count"] for s in splits.values() if abs(s["share_b"] - scenario["alpha"]) <= BALANCE_TOLERANCE) / shots
    best = min(splits.items(), key=lambda item: item[1]["cost"])
    top = max(splits.items(), key=lambda item: item[1]["count"])
    return dict(shots=shots, submitted=submitted, acceptance=shots / submitted, opt_per_submitted_shot=p_opt * shots / submitted,
                near_per_submitted_shot=p_near * shots / submitted, p_opt=p_opt, p_near=p_near, p_balanced=p_bal,
                mean_cost=sum(s["count"] * s["cost"] for s in splits.values()) / shots, distinct_splits=len(splits),
                best_state=best[0], best_cost=best[1]["cost"], best_gap_percent=max(0.0, 100 * (best[1]["cost"] / e_min - 1)),
                most_frequent_state=top[0], most_frequent_count=top[1]["count"], most_frequent_is_optimum=top[0] in optimal,
                splits=splits)


def split_text(bits):
    return "".join(str(b) for b in bits[::-1]) + "  (regions on B: " + (", ".join(str(r) for r, b in enumerate(bits) if b) or "none") + ")"


def local_counts(qc, shots, seed):
    """Noiseless shots sampled from the exact state (used only when no backend result is available)."""
    state = Statevector(qc.remove_final_measurements(inplace=False))
    state.seed(seed)
    return {str(k): int(v) for k, v in state.sample_counts(shots).items()}


def run_backend(qc, backend, shots, wait_minutes):
    """ONE circuit per job; prints the status every 10 s; returns counts or None."""
    compiled = transpile(qc, backend, optimization_level=1)
    first_try = time.time()
    while True:                                   # IonQ runs one job per account at a time: wait while another job runs
        try:
            job = backend.run(compiled, shots=shots)
            break
        except Exception as err:                  # qiskit-ionq can hide the real error behind an AttributeError
            if "already running" in str(err).lower() and time.time() - first_try < 60 * wait_minutes:
                print("IonQ: another job of this account is still running; waiting 30 s and trying again (%.0f s so far)" % (time.time() - first_try))
                time.sleep(30)
                continue
            print("Backend run failed:", type(err).__name__, err)
            return None
    try:
        print("Job submitted:", job.job_id())
    except Exception:
        print("Job submitted.")
    started = time.time()
    while time.time() - started < 60 * wait_minutes:
        try:
            status = job.status()
        except Exception as err:
            print("status check failed:", type(err).__name__, err)
            break
        name = getattr(status, "name", str(status))
        print("  status after %4.0f s: %s" % (time.time() - started, name))
        if name in ("DONE", "ERROR", "CANCELLED") or "DONE" in str(name).upper():
            break
        time.sleep(10)
    try:
        return job.result().get_counts()
    except Exception as err:
        print("Could not read the result:", type(err).__name__, err)
        return None


def show_results(key, entry, counts, source):
    spec = entry["spec"]
    reports = []
    for scenario_name in entry["graded_under"]:
        scenario = SCENARIOS[scenario_name]
        result = grade(counts, spec, scenario)
        predicted = entry["predicted"] if len(entry["graded_under"]) == 1 else dict(p_opt=scenario["uniform_p_opt"], p_near=None)
        print()
        print("=== %s, scored as '%s' (alpha = %s), %d shots, %s ===" % (entry["name"], scenario_name, scenario["alpha"], result["submitted"], source))
        if spec["kind"] == "iceberg":
            print("  Iceberg error detection: %d of %d shots ACCEPTED (%.1f %%; local approximate noise model predicted %.1f %%)."
                  " The numbers below are per ACCEPTED shot; per SUBMITTED shot: P(optimum) %.4f, P(within 5 %%) %.4f" % (
                      result["shots"], result["submitted"], 100 * result["acceptance"], 100 * entry["predicted"]["acceptance_local"],
                      result["opt_per_submitted_shot"], result["near_per_submitted_shot"]))
            if result["shots"] == 0:
                print("  No shot passed the checks: nothing to grade.")
                reports.append(dict(scenario=scenario_name, **{k: v for k, v in result.items() if k != "splits"}))
                continue
        if spec.get("seeded_by"):
            print("  NOTE: this circuit starts from the %s; with these angles it returns that answer, so it shows how much"
                  " the noise damages a known split - it is not a quantum search." % spec["seeded_by"])
        if entry["label"].startswith("Lemma"):
            print("  NOTE: Lemma 1 of the report: one qubit of this ansatz stays 50/50 for every angle, which caps a unique optimum"
                  " at 1/2; the trained circuit reaches almost 0 at alpha = 0.8 (it works at alpha = 0.5).")
        print("  P(optimum)            measured %.4f | noiseless prediction %.4f | random guessing %.5f" % (
            result["p_opt"], predicted["p_opt"], scenario["uniform_p_opt"]))
        if len(entry["graded_under"]) == 1:
            print("                         crude estimate %.4f (noiseless x 0.995^gates) | approximate local noise model %s%s" % (
                predicted["crude_p_opt"], "%.4f" % predicted["aer_p_opt"] if predicted.get("aer_p_opt") is not None else "-",
                " (per accepted shot)" if spec["kind"] == "iceberg" else ""))
            print("  P(within 5 %% of opt)  measured %.4f | noiseless prediction %.4f" % (result["p_near"], predicted["p_near"]))
        else:
            print("  P(within 5 %% of opt)  measured %.4f" % result["p_near"])
        print("  P(balanced, share within 0.05 of alpha) %.3f; mean cost %.4f (optimum %.4f); %d different splits measured" % (
            result["p_balanced"], result["mean_cost"], scenario["e_min"], result["distinct_splits"]))
        hits = round(result["p_opt"] * result["shots"])
        print("  samples to solution: this run %s shots per optimum | classical median evaluations: simulated annealing %d, local search %d" % (
            ("%.1f" if result["shots"] / hits < 10 else "%.0f") % (result["shots"] / hits) if hits else "> %d" % result["shots"],
            scenario["classical_median_evaluations"]["simulated annealing"], scenario["classical_median_evaluations"]["local search (restarts)"]))
        best = result["splits"][result["best_state"]]
        print("  best measured split:  %s cost %.4f (%s)" % (split_text(best["bits"]), best["cost"],
              "= exact optimum" if result["best_state"] in scenario["optimal_states"] else "+%.1f %% vs optimum" % result["best_gap_percent"]))
        print("  ten most frequent splits:")
        for state, s in sorted(result["splits"].items(), key=lambda item: -item[1]["count"])[:10]:
            tag = "OPTIMUM" if state in scenario["optimal_states"] else ("within 5%" if s["cost"] <= (1 + WITHIN) * scenario["e_min"] + 1e-12 else "")
            print("    x%-4d cost %.4f  share B %.2f  %-9s %s" % (s["count"], s["cost"], s["share_b"], tag, split_text(s["bits"])))
        plot(entry, scenario_name, scenario, result, source)
        reports.append(dict(scenario=scenario_name, **{k: v for k, v in result.items() if k != "splits"}))
    if entry.get("partner"):
        compare_with_partner(key, entry, reports[0])
    return reports


def compare_with_partner(key, entry, report):
    """Plain vs Iceberg side by side, if the partner circuit's saved run is available (results/ionq_<partner>_*.json)."""
    import glob
    partner = entry["partner"]
    files = sorted(glob.glob("results/ionq_%s_*.json" % partner))
    if not files:
        print("\\n(Run %s too, for the plain vs Iceberg comparison; or paste both RAW COUNTS lines to the team.)" % partner)
        return
    with open(files[-1]) as handle:
        other = json.load(handle)["summary"][0]
    rows = {key: report, partner: other}
    print("\\n=== Plain vs Iceberg (same logical QAOA, same scenario; %s) ===" % ("this run + saved " + files[-1]))
    print("  %-30s %8s %9s %10s %12s %12s %7s %6s" % ("circuit", "shots", "accepted", "P(opt)|acc", "P(opt)/shot", "P(near)/shot", "qubits", "2q"))
    for k, r in rows.items():
        e = CIRCUITS[k]
        print("  %-30s %8d %8.1f%% %10.4f %12.4f %12.4f %7d %6d" % (k, r["submitted"], 100 * r["acceptance"], r["p_opt"],
              r["opt_per_submitted_shot"], r["near_per_submitted_shot"], e["spec"]["n_qubits"], e["gates"]["two_qubit"]))


def cost_band(state, split_cost, scenario):
    """Band of a split: the optimum is recognised by its state (not by the cost ratio, which rounding can push a hair below 1)."""
    if state in scenario["optimal_states"]:
        return 0
    ratio = split_cost / scenario["e_min"]
    for index, (_, upper) in enumerate(BANDS[1:], start=1):
        if ratio <= upper + 1e-12:
            return index
    return len(BANDS) - 1


BANDS = [("optimum", 1.0), ("within 5 %", 1.05), ("5-20 %", 1.2), ("20-50 %", 1.5), ("1.5-2x", 2.0), ("2-3x", 3.0),
         ("3-5x", 5.0), ("5-10x", 10.0), ("> 10x", float("inf"))]


def plot(entry, scenario_name, scenario, result, source):
    """Measured shots vs the ideal (exact noiseless) distribution of the same circuit, in cost bands relative to the optimum."""
    import warnings
    try:
        import matplotlib.pyplot as plt
    except Exception as err:
        print("  (matplotlib not available: %s; the text above is the presentation)" % err)
        return
    spec, shots = entry["spec"], result["shots"]
    measured = [0] * len(BANDS)
    for state, s in result["splits"].items():
        measured[cost_band(state, s["cost"], scenario)] += s["count"]
    logical = dict(spec, kind="qaoa", n_qubits=spec["n_logical"]) if spec["kind"] == "iceberg" else spec   # ideal = plain logical circuit
    probabilities = Statevector(build_circuit(logical).remove_final_measurements(inplace=False)).probabilities()
    ideal = [0.0] * len(BANDS)
    for index, probability in enumerate(probabilities):
        if probability > 1e-12:
            bits = region_bits(format(index, "0%db" % logical["n_qubits"]), logical)
            state = int(sum(b << r for r, b in enumerate(bits)))
            ideal[cost_band(state, cost(bits, scenario), scenario)] += probability * shots
    with warnings.catch_warnings():               # Qollab's in-browser matplotlib warns about float figure sizes (harmless)
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = np.arange(len(BANDS))
        ax.bar(x - 0.2, measured, width=0.4, color="#2a78d6", label="measured shots (%s)%s" % (source, ", accepted only" if spec["kind"] == "iceberg" else ""))
        ax.bar(x + 0.2, ideal, width=0.4, color="#eb6834", alpha=0.75, label="ideal: exact noiseless x shots")
        for xi, value in zip(x, measured):
            if value:
                ax.text(xi - 0.2, value, str(value), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _ in BANDS])
        ax.set_xlabel("cost of the measured split compared with the exact optimum")
        ax.set_ylabel("shots")
        ax.set_title("%s: %s (%d shots)" % (entry["name"], scenario_name, shots), loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        plt.show()


def main(circuit: str = "__DEFAULT__", shots: int = 1000, wait_minutes: int = 60, seed: int = 2026):
    """circuit: __KEYS__, or iceberg_compare (= the plain and the Iceberg QAOA as two jobs, one after the other, then the
    comparison). ONE circuit per job on IonQ; shots per job; wait_minutes for the IonQ queue."""
    if circuit.strip() == "iceberg_compare":
        pair = [k for k, e in CIRCUITS.items() if e.get("partner")]
        pair.sort(key=lambda k: CIRCUITS[k]["spec"]["kind"] == "iceberg")     # plain first, then Iceberg
        results = {}
        for k in pair:
            print("\\n" + "=" * 100 + "\\nJob %d of 2: %s\\n" % (len(results) + 1, k) + "=" * 100)
            results[k] = run_one(k, shots, wait_minutes, seed)
        compare_pair(results)
        return
    run_one(circuit, shots, wait_minutes, seed)


def compare_pair(results):
    """Plain vs Iceberg from the two jobs of this Run: table and the figure for the video."""
    rows = [(k, r) for k, r in results.items() if r]
    if len(rows) < 2:
        print("Only %d of the 2 jobs returned results: no comparison." % len(rows))
        return
    print("\\n=== Plain vs Iceberg error detection (same logical QAOA, same scenario, %s) ===" % rows[0][1]["source"])
    print("  %-26s %7s %9s %11s %11s %12s %12s %7s %6s %6s" % ("circuit", "shots", "accepted", "P(opt)|kept", "P(near)|kept",
          "P(opt)/sent", "P(near)/sent", "qubits", "2q", "depth"))
    for k, r in rows:
        s, g = r["summary"][0], CIRCUITS[k]["gates"]
        print("  %-26s %7d %8.1f%% %11.4f %11.4f %12.4f %12.4f %7d %6d %6d" % (k, s["submitted"], 100 * s["acceptance"], s["p_opt"], s["p_near"],
              s["opt_per_submitted_shot"], s["near_per_submitted_shot"], g["qubits"], g["two_qubit"], g["depth"]))
    ice = [r for k, r in rows if CIRCUITS[k]["spec"]["kind"] == "iceberg"][0]
    if ice["source"].startswith("backend") and ice["summary"][0]["acceptance"] >= 0.999:
        print("  WARNING: Iceberg kept every shot. On a noisy simulator some shots must fail the checks, so this backend was most likely")
        print("  NOISELESS. For the noise model choose Select QPU -> 'IonQ Forte 1 (29q) noise model'.")
    try:
        import matplotlib.pyplot as plt
    except Exception as err:
        print("  (matplotlib not available: %s)" % err)
        return
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        names = ["plain" if CIRCUITS[k]["spec"]["kind"] != "iceberg" else "Iceberg" for k, _ in rows]
        colours = ["#2a78d6", "#1baf7a"]
        s = [r["summary"][0] for _, r in rows]
        noiseless = CIRCUITS[rows[0][0]]["predicted"]["p_near"]
        ax = axes[0]
        x = np.arange(2)
        ax.bar(x - 0.2, [100 * v["p_near"] for v in s], 0.4, color=colours, label="per kept shot")
        ax.bar(x + 0.2, [100 * v["near_per_submitted_shot"] for v in s], 0.4, color=colours, alpha=0.45, hatch="//", label="per shot sent")
        ax.axhline(100 * noiseless, color="#555555", ls="--", lw=1, label="noiseless")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("% of shots within 5 % of the optimum")
        ax.set_title("success rate", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        for ax, field, label in ((axes[1], "acceptance", "% of shots accepted"),):
            ax.bar(names, [100 * v[field] for v in s], color=colours)
            ax.set_ylim(0, 105)
            ax.set_title(label, loc="left", fontsize=10)
        for ax, field, label in ((axes[2], "qubits", "qubits"), (axes[3], "depth", "circuit depth (2q gates in labels)")):
            values = [CIRCUITS[k]["gates"][field] for k, _ in rows]
            ax.bar(names, values, color=colours)
            if field == "depth":
                for i, (k, _) in enumerate(rows):
                    ax.text(i, values[i], "%d 2q" % CIRCUITS[k]["gates"]["two_qubit"], ha="center", va="bottom", fontsize=8)
            ax.set_title(label, loc="left", fontsize=10)
        fig.suptitle("Mesh QAOA (12 regions), plain vs Iceberg error detection: %s" % rows[0][1]["source"], fontsize=10)
        fig.tight_layout()
        plt.show()


def run_one(circuit, shots, wait_minutes, seed):
    """One circuit, one job: build, run (or sample noiselessly without a backend), grade, print, save. Returns the payload."""
    print("ZollerQu mesh partitioning on IonQ, 12 qubits. This effort is supported by Qollab & IonQ.")
    key = circuit.strip()
    if key not in CIRCUITS:
        print("Unknown circuit %r. Allowed: %s. Using the default %s." % (circuit, ", ".join(CIRCUITS), DEFAULT_CIRCUIT))
        key = DEFAULT_CIRCUIT
    for scenario in SCENARIOS.values():                # exact optimum re-checked by brute force over all 4,096 splits
        best = min(cost([(k >> r) & 1 for r in range(N_REGIONS)], scenario) for k in range(2 ** N_REGIONS))
        assert abs(best - scenario["e_min"]) < 1e-9, "embedded optimum does not match brute force"
    print("Exact optima re-checked by brute force over all %d splits: passed" % 2 ** N_REGIONS)
    print("Circuits (chosen offline; '>' = the one sent now). Noiseless P(optimum) | crude noisy estimate | 2-qubit gates:")
    for k, entry in CIRCUITS.items():
        p = entry["predicted"]
        print("  %s %-26s %-24s %8.4f | %8.4f | %3d   %s" % (">" if k == key else " ", k, "/".join(s.split(" (")[0] for s in entry["graded_under"]),
              p["p_opt"], p["crude_p_opt"], entry["gates"]["two_qubit"], entry["label"]))
    entry = CIRCUITS[key]
    qc = build_circuit(entry["spec"])
    backend = globals().get("backend")             # Qollab pre-creates `backend` from the Select QPU dialog
    counts, source = None, None
    if backend is not None:
        print("Running on:", getattr(backend, "name", backend), "| one job: %s x %d shots" % (key, shots))
        counts = run_backend(qc, backend, shots, wait_minutes)
        source = "backend %s" % getattr(backend, "name", backend)
    if counts is None:
        print("No backend result: sampling the exact NOISELESS probabilities locally instead (not a hardware result).")
        counts = local_counts(qc, shots, seed)
        source = "local noiseless sampling"
    reports = show_results(key, entry, counts, source)
    payload = dict(source=source, shots=shots, circuit=key, name=entry["name"], counts=counts, summary=reports,
                   time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if source.startswith("backend"):
        try:
            os.makedirs("results", exist_ok=True)
            path = "results/ionq_%s_%s.json" % (key, time.strftime("%Y%m%d-%H%M%S"))
            with open(path, "w") as handle:
                json.dump(payload, handle)
            print("Saved", path, "(if the file does not appear, copy the RAW COUNTS line below)")
        except Exception as err:
            print("Could not save a file here (%s); copy the RAW COUNTS line below instead." % err)
    print("RAW COUNTS (copy this line to save the run):")
    print(json.dumps(payload))
    return payload


if sys.platform != "emscripten":  # Qollab (Pyodide) calls main() itself; locally we call it here
    main()
'''


def literal(value):
    return pprint.pformat(value, width=130, compact=True, sort_dicts=False)


code = (TEMPLATE.replace("__COMMIT__", commit).replace("__N__", str(selection["n"])).replace("__SCENARIOS__", literal(scenarios))
        .replace("__CIRCUITS__", literal(circuits)).replace("__DEFAULT__", fallback_key).replace("__KEYS__", ", ".join(circuits)))
assert "__" not in code.replace("__file__", "").replace("__init__", "").replace("__name__", ""), "unfilled placeholder"
legacy = ROOT / "qollab" / "legacy"
if not (legacy / "main_batch.py").exists() and "COPY THIS WHOLE CELL" not in (ROOT / "qollab" / "main.py").read_text():
    legacy.mkdir(exist_ok=True)
    shutil.move(ROOT / "qollab" / "main.py", legacy / "main_batch.py")
(ROOT / "qollab" / "main.py").write_text(code)
# notebook cell (markdown + code), replaced in place on later runs
nb_path = ROOT / "mesh_idea_ZollerQu.ipynb"
nb = nbformat.read(nb_path, as_version=4)
markdown = ("## Code for Qollab (paste into `main.py`)\n\n"
            "Copy the whole cell below into Qollab's `main.py`, press **Run** and choose **IonQ Forte 1 (29q) noise model** in Select QPU.\n\n"
            f"- **One circuit per job**: pick it with the `circuit` parameter in the Run dialog. The default is `{fallback_key}`. Options: "
            + ", ".join(f"`{k}`" for k in circuits) + ".\n"
            "- **Nothing is trained on Qollab**: the circuits and angles were chosen offline from the 12-qubit results (`scripts/ionq_selection_*.py`, table in `results/ionq_selection/candidates_table.md`).\n"
            "- **Error-detection experiment**: set `circuit` to `iceberg_compare`. One Run sends two jobs, one after the other: the tuned QAOA (p = 3, CPU + GPU) without error detection, then the same logical circuit on the Iceberg [[14, 12, 2]] code (15 qubits: 14 data + 1 check ancilla, no reset, every qubit measured once at the end). Iceberg shots that fail a check are discarded (error DETECTION, not correction); the comparison table and figure follow. Verified offline by `scripts/iceberg_build.py` (noiseless equivalence, every injected single-qubit fault caught; local approximate noise model in `results/iceberg/local_check.json`).\n"
            "- **Grading**: every shot is graded against the exact optimum. The output prints the predictions next to the measured numbers, and ends with a `RAW COUNTS` line to save.\n"
            "- **This notebook cell** runs the same code locally (exact noiseless sampling of the default circuit, labelled as such).")
start = next((i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.startswith("# ======  COPY THIS WHOLE CELL INTO QOLLAB")), None)
if start is None:
    nb.cells += [nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(code.rstrip("\n"))]
else:
    nb.cells[start - 1].source, nb.cells[start].source = markdown, code.rstrip("\n")
nbformat.write(nb, nb_path)
print(f"wrote qollab/main.py ({len(code) / 1e3:.0f} kB, {len(circuits)} circuits, default {fallback_key}) and the notebook cell")
