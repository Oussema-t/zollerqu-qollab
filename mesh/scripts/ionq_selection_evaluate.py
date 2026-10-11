"""Step 2 of the IonQ circuit selection: score every candidate and pick the circuits for ONE IonQ batch.

For each candidate collected by scripts/ionq_selection_extract.py (existing 12-qubit results only, no new training):
  1. noiseless P(optimum) and P(cost <= 1.05 x optimum) per shot (the notebook's own distribution; the circuit is rebuilt from its
     spec with qollab/mesh_ionq.circuit_from_spec, the builder main.py will use, and must reproduce that distribution exactly);
  2. two-qubit gates and depth MEASURED by transpiling to IonQ Forte 1 with qiskit-ionq: the QIS gate set that is submitted
     (rzz / cx count as one native gate each; IonQ maps rzz to one native ZZ) and, as an upper bound, qiskit-ionq's own local
     native compile (GPI / GPI2 / ZZ), which uses 2 ZZ per rzz;
  3. crude estimate = noiseless score x 0.995^(two-qubit gates)  [crude estimate];
  4. Aer density-matrix check (exact, no sampling): 0.5 % two-qubit depolarizing, 0.05 % one-qubit depolarizing, 0.5 % readout
     flips [approximate noise model, not IonQ]. It takes about a minute per circuit at 12 qubits, so it runs on the shortlist that
     decides the choice: the control and the AER_SHORTLIST best eligible circuits per scenario by crude estimate;
  5. samples to solution: quantum 1 / P(optimum) shots; classical median evaluations to first reach the optimum (simulated
     annealing and local search, restarted until they hit it).
Selection rule (fixed here, before looking at IonQ): the p = 0 control (one circuit, graded under both scenarios) plus, per
scenario, the two highest crude-estimate P(optimum) circuits with at most MAX_2Q two-qubit gates; at most one of the two may
start from a classical answer (warm start); 'initial state only' controls are not candidates. Fallback (one circuit per job):
the highest crude-estimate P(optimum) circuit that does not start from a classical answer. Optional extra: the varQITE ZY
circuit trained for CPU + GPU, which Lemma 1 of the report predicts to fail (the demonstration of the main finding).
usage: <python with qiskit-ionq and qiskit-aer> scripts/ionq_selection_evaluate.py
"""
import csv
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qollab"))
import mesh_ionq  # noqa: E402  (the builder main.py uses)
from qiskit import transpile  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402
from qiskit_ionq import IonQProvider  # noqa: E402

OUT = ROOT / "results" / "ionq_selection"
MAX_2Q = 200
CRUDE_ERROR = 0.005
NOISE = dict(two_qubit_depolarizing=0.005, one_qubit_depolarizing=0.0005, readout_flip=0.005)
SHOTS = 1000
AER_SHORTLIST = 6


def full_distribution(probs_qubits, spec, n_regions):
    """Distribution over all region splits (state index k: bit r = region r on device B) from the qubits' distribution."""
    qubit_to_region = spec["qubit_to_region"]
    if qubit_to_region == list(range(n_regions)) and not spec["fixed_regions"]:
        return probs_qubits
    base = sum(int(v) << int(r) for r, v in spec["fixed_regions"].items())
    states = np.arange(len(probs_qubits))
    index = np.full(len(states), base, dtype=np.int64)
    for qubit, region in enumerate(qubit_to_region):
        index |= ((states >> qubit) & 1) << region
    full = np.zeros(2**n_regions)
    np.add.at(full, index, probs_qubits)
    return full


def apply_readout(probs, n, flip):
    """Independent symmetric readout bit flips with probability `flip` on every measured qubit (exact, no sampling)."""
    tensor = probs.reshape([2] * n)
    confusion = np.array([[1 - flip, flip], [flip, 1 - flip]])
    for axis in range(n):
        tensor = np.moveaxis(np.tensordot(confusion, tensor, axes=([1], [axis])), 0, axis)
    return tensor.reshape(-1)


def label(spec):
    if spec["kind"] == "varqite":
        return spec["variant"]
    if spec["variant"] == "uniform control":
        return "p = 0 control (uniform)"
    if spec["optimiser"] in ("fixed schedule",) or spec["optimiser"].startswith("fixed:") or spec["optimiser"] == "initial state only":
        return f"{spec['variant']}, {spec['optimiser']}, p = {spec['p']}"
    return f"{spec['variant']} + {spec['optimiser']}, p = {spec['p']}"


def two_qubit_count(circuit):
    return sum(1 for item in circuit.data if item.operation.num_qubits == 2 and item.operation.name not in ("barrier", "measure"))


if __name__ == "__main__":
    clock = time.time()
    data = pickle.load(open(OUT / "candidates.pkl", "rb"))
    provider = IonQProvider(token="offline-gate-counting-only")     # no API call: only the backend's gate set is used
    forte_qis = provider.get_backend("qpu.forte-1", gateset="qis")
    forte_native = provider.get_backend("qpu.forte-1", gateset="native")
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE["two_qubit_depolarizing"], 2), ["rzz"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE["one_qubit_depolarizing"], 1), ["rx", "ry", "rz"])
    aer = AerSimulator(method="density_matrix", noise_model=noise_model, precision="single")
    rows, table = [], {}
    cache_path = OUT / "aer_cache.json"
    aer_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for scenario, info in data["scenarios"].items():
        n = len(info["h"])
        optimal, near = np.asarray(info["optimal_states"]), np.asarray(info["near_mask"])
        alpha = info["alpha"]
        ones = np.array([bin(s).count("1") for s in optimal])
        share_p_opt = float(np.sum(alpha**ones * (1 - alpha) ** (n - ones)))
        hits = info["classical_first_hit"]
        classical = {m: dict(median=float(np.median(v)), mean=float(np.mean(v)), runs=len(v)) for m, v in hits.items()}
        table[scenario] = dict(uniform_p_opt=len(optimal) / 2**n, target_share_p_opt=share_p_opt, classical=classical, rows=[])
        for candidate in info["candidates"]:
            spec = dict(candidate["spec"], scenario=scenario)
            if spec["kind"] == "qaoa":
                spec["zz_pairs"] = mesh_ionq.round_robin_pairs(spec["n_qubits"])
            spec.setdefault("qubit_to_region", list(range(spec["n_qubits"])))
            spec["qubit_to_region"] = [int(r) for r in spec["qubit_to_region"]]
            circuit = mesh_ionq.circuit_from_spec(spec, measure=False)
            rebuilt = full_distribution(Statevector(circuit).probabilities(), spec, n)
            rebuild_error = float(np.max(np.abs(rebuilt - candidate["reference"])))
            assert rebuild_error < 1e-9, f"{label(spec)}: main.py builder differs from the notebook by {rebuild_error}"
            p_opt, p_near = float(rebuilt[optimal].sum()), float(rebuilt[near].sum())
            qis = transpile(circuit, forte_qis, optimization_level=1)
            g2 = two_qubit_count(qis)
            depth, depth2 = qis.depth(), qis.depth(lambda item: item.operation.num_qubits == 2)
            native = transpile(circuit, forte_native, optimization_level=1)
            native_zz = int(native.count_ops().get("zz", 0))
            fidelity = (1 - CRUDE_ERROR) ** g2
            aer_p_opt = aer_p_near = None                               # filled for the shortlist below
            seeded = spec.get("seeded_by")
            state_prep = "H on every qubit" if spec.get("start", "plus") == "plus" else "RY product state (one RY per qubit; from the classical warm start)"
            row = dict(scenario=scenario, circuit=label(spec), kind=spec["kind"], variant=spec["variant"], optimiser=spec["optimiser"],
                       p=spec["p"], qubits=spec["n_qubits"], training_circuits=spec.get("training_circuits", 0), seeded_by=seeded or "",
                       state_prep=state_prep, generic_state_prep=False,
                       p_opt=p_opt, p_near=p_near, two_qubit_gates=g2, depth=depth, two_qubit_depth=depth2, native_zz_upper_bound=native_zz,
                       crude_fidelity=fidelity, crude_p_opt=p_opt * fidelity, crude_p_near=p_near * fidelity,
                       aer_p_opt=aer_p_opt, aer_p_near=aer_p_near,
                       shots_to_solution_noiseless=(1 / p_opt) if p_opt > 0 else float("inf"),
                       shots_to_solution_crude=(1 / (p_opt * fidelity)) if p_opt > 0 else float("inf"),
                       shots_to_solution_aer=None,
                       rebuild_error=rebuild_error, eligible=bool(g2 <= MAX_2Q and spec["optimiser"] != "initial state only"
                                                                   and spec["variant"] != "uniform control"))
            rows.append(row)
            table[scenario]["rows"].append(dict(row, spec=spec, _circuit=circuit))
        print(f"{scenario}: {len(info['candidates'])} candidates scored ({time.time() - clock:.0f} s)")

    def rank(scenario_rows):
        return sorted(scenario_rows, key=lambda r: (-r["crude_p_opt"], -r["crude_p_near"], r["two_qubit_gates"]))

    # ---------------------------------------------------------------- Aer check (shortlist + every chosen circuit) ----
    def aer_check(row):
        info = data["scenarios"][row["scenario"]]
        n, optimal, near = len(info["h"]), np.asarray(info["optimal_states"]), np.asarray(info["near_mask"])
        cache_key = f"{row['scenario']}|{row['circuit']}|{json.dumps(row['spec'].get('gammas', row['spec'].get('params')))[:200]}"
        if cache_key not in aer_cache:
            started = time.time()
            noisy_circuit = transpile(row["_circuit"], basis_gates=["rzz", "rx", "ry", "rz"], optimization_level=1)
            noisy_circuit.save_probabilities()
            noisy_qubits = np.asarray(aer.run(noisy_circuit, shots=1).result().data(0)["probabilities"], dtype=float)
            noisy = full_distribution(apply_readout(noisy_qubits / noisy_qubits.sum(), row["spec"]["n_qubits"], NOISE["readout_flip"]), row["spec"], n)
            aer_cache[cache_key] = [float(noisy[optimal].sum()), float(noisy[near].sum())]
            cache_path.write_text(json.dumps(aer_cache, indent=1))
            print(f"  Aer [{row['scenario']}] {row['circuit']}: P(opt) {aer_cache[cache_key][0]:.5f} (crude {row['crude_p_opt']:.5f}) "
                  f"in {time.time() - started:.0f} s", flush=True)
        row["aer_p_opt"], row["aer_p_near"] = aer_cache[cache_key]
        row["shots_to_solution_aer"] = 1 / row["aer_p_opt"] if row["aer_p_opt"] > 0 else float("inf")
        flat = next(r for r in rows if r["scenario"] == row["scenario"] and r["circuit"] == row["circuit"])
        flat.update(aer_p_opt=row["aer_p_opt"], aer_p_near=row["aer_p_near"], shots_to_solution_aer=row["shots_to_solution_aer"])

    for scenario, entry in table.items():
        shortlist = [r for r in entry["rows"] if r["variant"] == "uniform control"] + rank([r for r in entry["rows"] if r["eligible"]])[:AER_SHORTLIST]
        for row in shortlist:
            aer_check(row)

    # ---------------------------------------------------------------- selection (rule fixed in the docstring) ----
    selected = []
    control = next(r for r in table[next(iter(table))]["rows"] if r["variant"] == "uniform control")
    selected.append(dict(control, role="p = 0 control (uniform guessing; the same circuit is graded under both scenarios)", graded_under="both scenarios"))
    for scenario, entry in table.items():
        picked, seeded_used = [], False
        for row in rank([r for r in entry["rows"] if r["eligible"]]):
            if row["seeded_by"] and seeded_used:
                continue
            picked.append(row)
            seeded_used |= bool(row["seeded_by"])
            if len(picked) == 2:
                break
        for position, row in enumerate(picked, start=1):
            selected.append(dict(row, role=f"{scenario}: best #{position} by crude estimate (<= {MAX_2Q} two-qubit gates)", graded_under=scenario))
    pure = [r for entry in table.values() for r in entry["rows"] if r["eligible"] and not r["seeded_by"]]
    fallback = max(pure, key=lambda r: (r["crude_p_opt"], r["crude_p_near"]))
    lemma_scenario = next(s for s in table if abs(data["scenarios"][s]["alpha"] - 0.5) > 1e-9)
    lemma = next(r for r in table[lemma_scenario]["rows"] if r["variant"] == "varQITE ZY (paper, all pairs)")
    extras = [dict(lemma, role="optional extra: Lemma 1 demonstration (varQITE ZY trained for CPU + GPU; predicted to fail)",
                   graded_under=lemma_scenario)]
    for row in selected + [fallback] + extras:                  # every circuit that may run gets the Aer check
        aer_check(row)

    # ---------------------------------------------------------------- outputs ----
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0]]
    with open(OUT / "candidates_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def instance_block(scenario):
        info = data["scenarios"][scenario]
        return dict(name=scenario, alpha=info["alpha"], lam=info["lam"], J=np.asarray(info["J"]).tolist(), h=np.asarray(info["h"]).tolist(),
                    const=info["const"], scale=info["scale"], work=np.asarray(info["work"]).tolist(), e_min=info["e_min"],
                    optimal_states=[int(s) for s in info["optimal_states"]], near_states=[int(s) for s in np.flatnonzero(info["near_mask"])],
                    uniform_p_opt=table[scenario]["uniform_p_opt"], target_share_p_opt=table[scenario]["target_share_p_opt"],
                    classical_samples_to_solution=table[scenario]["classical"])

    def circuit_block(row, index):
        spec = row["spec"]
        keep = {k: spec[k] for k in ("kind", "variant", "optimiser", "p", "n_qubits", "qubit_to_region", "fixed_regions") if k in spec}
        if spec["kind"] == "qaoa":
            keep.update({k: spec[k] for k in ("gammas", "betas", "start", "mixer", "J_scaled", "h_scaled", "zz_pairs") if k in spec})
            keep.update({k: spec[k] for k in ("thetas", "warm_start_c", "seeded_by", "note") if k in spec})
        else:
            keep.update(gates=spec["gates"], params=spec["params"])
        return dict(index=index, name=row["circuit"], role=row["role"], graded_under=row["graded_under"], spec=keep,
                    predicted=dict(p_opt=row["p_opt"], p_near=row["p_near"], crude_p_opt=row["crude_p_opt"], crude_p_near=row["crude_p_near"],
                                   aer_p_opt=row["aer_p_opt"], aer_p_near=row["aer_p_near"]),
                    gates=dict(two_qubit=row["two_qubit_gates"], depth=row["depth"], two_qubit_depth=row["two_qubit_depth"],
                               native_zz_upper_bound=row["native_zz_upper_bound"]),
                    training_on_qollab=0, training_done_offline_circuits=row["training_circuits"])

    selection = dict(
        created=time.strftime("%Y-%m-%d %H:%M"), case=data["instance"][0], n=data["instance"][1], seed=data["instance"][2], shots=SHOTS,
        source="existing 12-qubit results (results/task04_checkpoint_small.json.gz, results/varqite_checkpoint_small.json.gz); no new training",
        conventions=dict(qubit_order="qubit q = region qubit_to_region[q] (identity unless stated); Qiskit count keys: the RIGHTMOST character is qubit 0",
                         bit_meaning="bit 1 = region on device B (GPU / second CPU), bit 0 = device A; state index k has bit r = region r",
                         alpha="target work share of device B per scenario (0.5 equal devices, 0.8 CPU + GPU)",
                         qaoa="H (or RY(theta) warm start); per layer rzz(2 gamma J_ij) over zz_pairs, rz(2 gamma h_i), mixer rx(-2 beta) or warm RY-RZ-RY",
                         rebuild="qollab/mesh_ionq.circuit_from_spec(spec); verified to reproduce the notebook distribution to < 1e-9"),
        selection_rule=__doc__.split("Selection rule")[1].split("usage:")[0].strip(),
        noise_labels=dict(crude="crude estimate: noiseless x 0.995^(two-qubit gates)", aer="approximate noise model, not IonQ: " + json.dumps(NOISE)),
        instances={scenario: instance_block(scenario) for scenario in table},
        circuits=[circuit_block(row, index) for index, row in enumerate(selected)],
        extras=[circuit_block(row, len(selected) + index) for index, row in enumerate(extras)],
        fallback=circuit_block(dict(fallback, role="fallback: the single circuit to run if IonQ accepts only one circuit per job",
                                    graded_under=fallback["scenario"]), 0))
    (ROOT / "qollab" / "selected_circuits.json").write_text(json.dumps(selection, indent=1))
    (OUT / "selection_summary.json").write_text(json.dumps(dict(
        selected=[dict(circuit=r["circuit"], scenario=r["graded_under"], role=r["role"]) for r in selected],
        extras=[dict(circuit=r["circuit"], scenario=r["graded_under"], role=r["role"]) for r in extras],
        fallback=dict(circuit=fallback["circuit"], scenario=fallback["scenario"]),
        classical={s: e["classical"] for s, e in table.items()},
        uniform_p_opt={s: e["uniform_p_opt"] for s, e in table.items()}, target_share_p_opt={s: e["target_share_p_opt"] for s, e in table.items()},
        noise=NOISE, max_two_qubit=MAX_2Q), indent=1))
    for entry in table.values():
        for row in entry["rows"]:
            row.pop("_circuit", None)
    print(f"wrote qollab/selected_circuits.json ({len(selected)} circuits + fallback) and results/ionq_selection/ in {time.time() - clock:.0f} s")
    for row in selected:
        print(f"  [{row['graded_under']}] {row['circuit']}: P(opt) {row['p_opt']:.4g}, crude {row['crude_p_opt']:.4g}, Aer {row['aer_p_opt']:.4g}, 2q {row['two_qubit_gates']}")
    print(f"  fallback: [{fallback['scenario']}] {fallback['circuit']}")
