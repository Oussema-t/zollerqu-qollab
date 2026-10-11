"""Do the tuned 12-qubit QAOA settings carry over to larger meshes? (16 and 20 qubits, notebook simulator; 24 and 29 qubits use
scripts/run_29_qubits.py with the settings file written by --choose.)

Leakage-free, like scripts/qaoa_tuning.py:
  - start: for each device scenario and depth p, the setting that scored best on the 12-qubit CALIBRATION meshes
    (results/qaoa_tuning/tuning_n12_large_calibration.json if present, else tuning_n12.json); warm-start rows excluded;
  - 'transferred': those angles applied unchanged at n (no training at n);
  - 're-optimised at n': all 2p angles optimised again on calibration meshes at n (mesh seeds 101 / 102, never scored), started
    from the transferred angles, with the optimiser that won at 12 qubits and a fixed budget of calibration-objective evaluations;
  - TEST: cases '2 main' and '9 priority' on mesh seeds 7, 8, 9 (the instances of the 29-qubit benchmark), scored once.
Calibration meshes: all 9 cases at n = 16 (18 instances); the Task 04 calibration cases at n = 20 (4 instances, for run time).
Metrics per test instance: exact optimum by enumeration, P(optimum), P(within 5 %), hit@1000 for both, the classical baselines
(Task 04 functions and budgets) and random sampling; circuit cost measured on the built Qiskit circuit (two-qubit gates, depth).

usage: python scripts/qaoa_scaling.py <n> <scenario index> <p, comma-separated> [budget]
       python scripts/qaoa_scaling.py --choose      -> scripts/q29_choice_tuned.json (settings for n = 24 and 29, from n <= 20 only)
       python scripts/qaoa_scaling.py --freeze <n>  -> freeze the calibration-mesh regions at n once, before parallel runs
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "qaoa_tuning" / "scaling"
OUT.mkdir(parents=True, exist_ok=True)
TEST_CASES, TEST_SEEDS = ["2 main", "9 priority"], [7, 8, 9]
DEFAULT_BUDGET = {16: 300, 20: 150}
T0 = time.time()

_spec = importlib.util.spec_from_file_location("qaoa_tuning", ROOT / "scripts" / "qaoa_tuning.py")
qt = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, sys.argv[:1]                  # qaoa_tuning reads sys.argv[1] at import time
_spec.loader.exec_module(qt)
sys.argv = _argv


def log(message):
    print(f"[{time.time() - T0:6.0f} s] {message}", flush=True)


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def twelve_qubit_source():
    for name in ("tuning_n12_large_calibration.json", "tuning_n12.json"):
        path = ROOT / "results" / "qaoa_tuning" / name
        if path.exists():
            return name, json.loads(path.read_text())
    raise FileNotFoundError("run scripts/qaoa_tuning.py first")


def start_setting(source, scenario, p):
    """Best non-warm-start row at depth p on the 12-qubit calibration meshes."""
    rows = [r for r in source["scenarios"][scenario]["rows"] if r["p"] == p and "warm" not in r["method"]]
    best = max(rows, key=lambda r: r["calibration_p_near"])
    optimiser = next((o for o in qt.OPTIMISERS if f"({o})" in best["method"]), "L-BFGS-B")
    return best, optimiser


def circuit_cost(ns, inst, angles):
    """Two-qubit gates and depth of the logical circuit (all-to-all, as sent to IonQ), measured on the built circuit."""
    model, p = inst["model"], len(angles) // 2
    circuit = ns["build_qaoa_circuit"](model["J"] / model["scale"], model["h"] / model["scale"], angles[:p], angles[p:])
    ops = circuit.count_ops()
    return dict(two_qubit_gates=int(ops.get("rzz", 0)), depth=int(circuit.depth()),
                depth_two_qubit=int(circuit.depth(lambda instruction: instruction.operation.num_qubits == 2)))


def test_instances(ns, n, scenario):
    out = []
    for case in TEST_CASES:
        for seed in TEST_SEEDS:
            regions = ns["coarsen_mesh"](qt.case_mesh(ns, case, seed), n, seed)
            ns["check_region_case"]((case, n, seed), regions, n)
            out.append(ns["t4_build_instance"](regions, case, n, seed, scenario))
    return out


def classical_references(ns, insts):
    rows = []
    for inst in insts:
        records = ns["run_classical"](inst)
        row = {m: dict(hit_opt=r["hit_opt"], hit_near=r["hit_near"]) for m, r in records.items()}
        row["per-shot random (uniform)"] = dict(p_opt=len(inst["optimal_states"]) / 2 ** inst["n"],
                                                p_near=float(inst["near_mask"].mean()))
        rows.append(row)
    return rows


def calibration_cases(ns, n):
    return None if n <= 16 else sorted({case for case, _ in ns["calibration_regions"]})


def run(n, scenario_index, layers, budget):
    ns = qt.load_namespace()
    scenario = list(ns["TARGET_SHARES"])[scenario_index]
    source_name, source = twelve_qubit_source()
    log(f"notebook loaded; n = {n}, {scenario}, p = {layers}, budget {budget}; 12-qubit settings from {source_name}")
    cal_insts = qt.large_calibration_instances(ns, scenario, n=n, cases=calibration_cases(ns, n))
    test_insts = test_instances(ns, n, scenario)
    cal_views, test_views = qt.views_for(ns, cal_insts), qt.views_for(ns, test_insts)
    refs = classical_references(ns, test_insts)
    log(f"{len(cal_insts)} calibration and {len(test_insts)} test instances built; classical baselines done")
    qt.BUDGET = budget
    for p in layers:
        target = OUT / f"n{n}_{slug(scenario)}_p{p}.json"
        start, optimiser = start_setting(source, scenario, p)
        angles12 = np.array(start["angles"])
        clock = time.time()
        cal_transferred = float(np.mean([qt.metrics(ns, v, angles12)["p_near"] for v in cal_views]))
        angles_n, cal_reopt, used = qt.optimise(ns, cal_views, angles12, optimiser, seed=p)
        seconds = time.time() - clock
        record = dict(n=n, scenario=scenario, p=p, source=source_name, start_method=start["method"],
                      start_calibration_p_near_n12=start["calibration_p_near"], optimiser=optimiser, budget=budget,
                      calibration_instances=[f"{i['case']}, seed {i['seed']}" for i in cal_insts],
                      test_instances=[f"{i['case']}, seed {i['seed']}" for i in test_insts],
                      circuit=circuit_cost(ns, test_insts[0], angles12), classical=refs, methods={})
        for label, angles, cal, evaluations, wall in (("transferred from 12 qubits", angles12, cal_transferred, 0, 0.0),
                                                      (f"re-optimised at n = {n}", angles_n, cal_reopt, used, seconds)):
            rows = [qt.metrics(ns, v, angles) for v in test_views]
            record["methods"][label] = dict(angles=[float(a) for a in angles], calibration_p_near=cal, training_evaluations=evaluations,
                                            training_instances=len(cal_insts) if evaluations else 0, training_seconds=wall,
                                            test_mean={k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, test=rows)
        target.write_text(json.dumps(record, indent=1))
        t, r = record["methods"]["transferred from 12 qubits"], record["methods"][f"re-optimised at n = {n}"]
        log(f"p = {p}: calibration {cal_transferred:.2e} -> {cal_reopt:.2e} ({used} evaluations, {seconds:.0f} s); TEST P(near) "
            f"transferred {t['test_mean']['p_near']:.2e}, re-optimised {r['test_mean']['p_near']:.2e}; hit@1000 "
            f"{t['test_mean']['hit_near']:.3f} / {r['test_mean']['hit_near']:.3f}")


def choose():
    """Settings for n = 24 and 29 from the n <= 20 calibration scores only (rule fixed before any n = 24 / 29 result exists):
    per scenario, the re-optimised n = 20 angles at every depth that was run, plus the single best depth by n = 20 calibration score."""
    choice = {}
    for path in sorted(OUT.glob("n20_*_p*.json")):
        rec = json.loads(path.read_text())
        m = rec["methods"]["re-optimised at n = 20"]
        choice.setdefault(rec["scenario"], {"configs": []})["configs"].append(
            dict(label=f"tuned QAOA p = {rec['p']} (angles re-optimised at n = 20)", role=f"tuned p{rec['p']}", p=rec["p"],
                 angles=m["angles"], calibration_p_near_n20=m["calibration_p_near"]))
    for scenario, c in choice.items():
        c["configs"].sort(key=lambda d: d["p"])
        c["best_p_by_n20_calibration"] = max(c["configs"], key=lambda d: d["calibration_p_near_n20"])["p"]
    path = ROOT / "scripts" / "q29_choice_tuned.json"
    path.write_text(json.dumps(dict(rule=choose.__doc__, source="results/qaoa_tuning/scaling/n20_*.json", choice=choice), indent=1))
    print(f"saved {path}: " + "; ".join(f"{s}: p = {[d['p'] for d in c['configs']]}" for s, c in choice.items()))


if __name__ == "__main__":
    if sys.argv[1] == "--choose":
        choose()
    elif sys.argv[1] == "--freeze":                       # the region assignments do not depend on the scenario
        size, namespace = int(sys.argv[2]), qt.load_namespace()
        qt.large_calibration_instances(namespace, list(namespace["TARGET_SHARES"])[0], n=size, cases=calibration_cases(namespace, size))
        log(f"calibration regions at n = {size} frozen")
    else:
        n = int(sys.argv[1])
        run(n, int(sys.argv[2]), [int(p) for p in sys.argv[3].split(",")],
            int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_BUDGET[n])
