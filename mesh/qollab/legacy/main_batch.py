"""ZollerQu - heterogeneous mesh partitioning on IonQ. This effort is supported by Qollab & IonQ.

One IonQ job: for the headline mesh (L-plate, 12 regions = 12 qubits) and two device scenarios (equal devices, CPU + GPU),
the job contains the p = 0 control, LR-QAOA (p = 1, 2), optimised QAOA (p = 2) and two final trained varQITE circuits.
Every shot is graded against the exact optimum (checked by brute force here). varQITE was trained on a noiseless local
simulator; only its final circuits are sampled on IonQ. Choose 'IonQ Forte 1 (29q) noise model' in Select QPU.
"""
import json
import sys
from pathlib import Path

import mesh_ionq as mi

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")


def show_table(labels, metrics, batch):
    predicted = {(s["name"], c["name"]): c["predicted"] for s in batch["scenarios"] for c in s["circuits"]}
    print(f"  {'scenario':<20}{'circuit':<38}{'P(opt)':>8}{'(noiseless)':>12}{'P(<=1.05 opt)':>14}{'(noiseless)':>12}{'P(balanced)':>12}{'best shot vs opt':>17}")
    for (scenario, name), m in zip(labels, metrics):
        p = predicted[(scenario, name)]
        print(f"  {scenario:<20}{name:<38}{m['p_opt']:>8.4f}{p['p_opt']:>12.4f}{m['p_near']:>14.4f}{p['p_near']:>12.4f}{m['p_balanced']:>12.3f}"
              f"{'+' + format(m['best_gap_percent'], '.1f') + ' %' if m['best_gap_percent'] > 1e-9 else 'optimum':>17}")


def main(shots: int = 1000, show_saved: bool = True, include_varqite: bool = True, print_raw_counts: bool = True):
    batch = json.loads((HERE / "mesh_batch.json").read_text())
    n = batch["n"]
    print("ZollerQu mesh partitioning on IonQ - This effort is supported by Qollab & IonQ")
    print(f"case {batch['case']}, {n} regions = {n} qubits, scenarios: " + ", ".join(f"{s['name']} (alpha = {s['alpha']})" for s in batch["scenarios"]))
    for scenario in batch["scenarios"]:
        mi.check_exact_optimum(scenario, n)
    print("exact optimum re-checked by brute force over all splits: passed")
    if show_saved:
        saved = sorted((HERE / "results_ionq").glob("*.json")) if (HERE / "results_ionq").exists() else []
        if not saved:
            print("no recorded IonQ run is stored yet (results_ionq/ is empty)")
        for path in saved:
            record = json.loads(path.read_text())
            print(f"\nRECORDED on {record['backend']}, {record['date']}, {n} qubits, {record['shots']} shots ({path.name}):")
            show_table([tuple(label) for label in record["labels"]], record["metrics"], batch)
    circuits, labels = mi.build_batch(batch, include_varqite)
    print(f"\nlive run: {len(circuits)} circuits in ONE job, {shots} shots each; two-qubit gates per circuit: "
          + ", ".join(str(sum(1 for d in c.data if d.operation.num_qubits == 2)) for c in circuits))
    backend = globals().get("backend")
    counts = mi.run_batch(circuits, backend, shots)
    source = f"IonQ backend {getattr(backend, 'name', backend)}"
    if counts is None:
        source = "LOCAL NOISELESS SAMPLES (no IonQ backend or the IonQ run failed)"
        counts = mi.local_counts(circuits, shots)
    scenario_of = {s["name"]: s for s in batch["scenarios"]}
    metrics = [mi.grade(c, scenario_of[label[0]], n) for c, label in zip(counts, labels)]
    print(f"\nRESULTS ({source}); '(noiseless)' = local exact prediction for comparison:")
    show_table(labels, metrics, batch)
    if print_raw_counts:
        print("\nRAW_RESULT_JSON " + json.dumps(dict(source=source, shots=shots, labels=labels, metrics=metrics, counts=counts)))


if sys.platform != "emscripten":
    main()
