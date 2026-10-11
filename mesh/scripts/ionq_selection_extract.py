"""Step 1 of the IonQ circuit selection: collect every candidate circuit of the EXISTING 12-qubit results (no new training).

Runs the notebook cells up to Task 04b in-process with IPython (the way Jupyter compiles them, so the saved checkpoints are
reused and nothing is retrained), in a temporary copy of the workspace (nothing in the workspace is written). For the IonQ
instance (case 2, 12 regions, seed 7, both device scenarios) it stores, per candidate: the exact circuit parameters, the
notebook's own reference probability distribution and the stored metrics. It also measures classical "samples to solution":
evaluations simulated annealing and local search (same algorithms and settings as Task 04) need to first reach the optimum.
usage: <python with IPython> scripts/ionq_selection_extract.py   ->  results/ionq_selection/candidates.pkl
"""
import contextlib
import io
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ionq_selection"
INSTANCE = ("2 main", 12, 7)
CLASSICAL_RUNS = 100                      # independent restart sequences per classical method
CLASSICAL_CAP = 1_000_000                 # evaluations; a sequence that has not hit the optimum by then is censored


def load_namespace(work):
    from IPython.core.interactiveshell import InteractiveShell
    nb = nbformat.read(work / "mesh_idea_ZollerQu.ipynb", as_version=4)
    shell = InteractiveShell.instance()
    for cell in nb.cells:
        if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
            continue
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = shell.run_cell(cell.source, store_history=False, silent=True)
        if not result.success:
            raise RuntimeError(f"cell failed: {cell.source[:60]!r}: {result.error_in_exec or result.error_before_exec}")
        text = buffer.getvalue()
        if cell.source.startswith("# TASK_04_IMPLEMENTATION"):
            line = next((l for l in text.splitlines() if "loaded from the checkpoint" in l or "instances computed" in l), "")
            print("  Task 04:", line.strip()[:160])
        if cell.source.startswith("# TASK_04B_IMPLEMENTATION"):
            print("  Task 04b:", next((l for l in text.splitlines() if "RESULT" in l), "").strip()[:160])
            return shell.user_ns
    raise RuntimeError("Task 04b cell not found")


def first_hit_annealing(inst, settings, rng, e_min, budget):
    """Simulated annealing exactly as in Task 04 (single-spin flips, geometric schedule over `budget` evaluations), restarted
    with a fresh random start until the CURRENT state is optimal. Returns the evaluation count at the first hit (or None)."""
    model, n = inst["model"], inst["n"]
    J_sym, h, const = model["J"] + model["J"].T, model["h"], model["const"]
    start_t = settings["start_fraction"] * inst["e_std_formula"]
    temperatures = np.geomspace(start_t, settings["end_ratio"] * start_t, budget - 1)
    used = 0
    while used < CLASSICAL_CAP:
        z = 1.0 - 2.0 * rng.integers(0, 2, size=n)
        local = J_sym @ z + h
        energy = float(z @ model["J"] @ z + h @ z + const)
        used += 1
        if energy <= e_min + 1e-9:
            return used
        for temperature in temperatures:
            i = int(rng.integers(n))
            delta = -2.0 * z[i] * local[i]
            used += 1
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                z[i] = -z[i]
                local += 2.0 * z[i] * J_sym[:, i]
                energy += delta
                if energy <= e_min + 1e-9:
                    return used
    return None


def first_hit_local_search(inst, rng, e_min):
    """Steepest-descent single flips from random starts (Task 04's local search), restarted until the optimum is reached;
    every neighbour evaluation counts, as in the benchmark."""
    model, n = inst["model"], inst["n"]
    J_sym, h, const = model["J"] + model["J"].T, model["h"], model["const"]
    used = 0
    while used < CLASSICAL_CAP:
        z = 1.0 - 2.0 * rng.integers(0, 2, size=n)
        local = J_sym @ z + h
        energy = float(z @ model["J"] @ z + h @ z + const)
        used += 1
        if energy <= e_min + 1e-9:
            return used
        while True:
            deltas = -2.0 * z * local
            used += n
            i = int(np.argmin(deltas))
            if deltas[i] >= -1e-12:
                break
            z[i] = -z[i]
            local += 2.0 * z[i] * J_sym[:, i]
            energy += deltas[i]
            if energy <= e_min + 1e-9:
                return used
    return None


def pick_run(group):
    """The deterministic run: the one started from the given angles (LR-QAOA / recovery angles), never a random start."""
    runs = [run for run in group["runs"] if run.get("init") != "random"]
    assert runs, f"no deterministic run in {group['variant']} {group['optimiser']} p={group['p']}"
    return runs[0]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ionq_selection_"))
    shutil.copy(ROOT / "mesh_idea_ZollerQu.ipynb", work)
    shutil.copytree(ROOT / "data", work / "data")
    (work / "results").mkdir()
    for name in ("task04_checkpoint_small.json.gz", "varqite_checkpoint_small.json.gz"):
        shutil.copy(ROOT / "results" / name, work / "results" / name)
    os.chdir(work)
    print(f"notebook cells running in {work} (checkpoints reused, workspace untouched)")
    ns = load_namespace(work)
    view_probabilities, t4_view, warm_start_setup = ns["view_probabilities"], ns["t4_view"], ns["warm_start_setup"]
    from qiskit.quantum_info import Statevector
    data = dict(instance=INSTANCE, scenarios={}, settings=dict(TASK04={k: v for k, v in ns["TASK04"].items() if not hasattr(v, "shape")},
                T4_SA_SETTINGS=ns["T4_SA_SETTINGS"], T4_LR_DELTAS=ns["T4_LR_DELTAS"], T4_CODE_HASH=ns["T4_CODE_HASH"]))
    for s_index, scenario in enumerate(ns["TARGET_SHARES"]):
        key = INSTANCE + (scenario,)
        stored_inst = ns["task04_ising_cases"][key]                     # big arrays were dropped after the benchmark: rebuild exactly
        inst = ns["t4_build_instance"](stored_inst["regions"], INSTANCE[0], INSTANCE[1], INSTANCE[2], scenario)
        assert inst["fingerprint"] == stored_inst["fingerprint"], "rebuilt instance differs from the benchmarked one"
        model, n = inst["model"], inst["n"]
        _, spectral_state, _ = ns["spectral_split"](inst)
        regions = inst["regions"]
        info = dict(name=scenario, alpha=inst["alpha"], lam=inst["lam"], J=model["J"], h=model["h"], const=float(model["const"]),
                    scale=float(model["scale"]), J_scaled=model["J_scaled"], h_scaled=model["h_scaled"], work=regions["region_work"],
                    energies=inst["energies"], e_min=inst["e_min"], optimal_states=inst["optimal_states"], near_mask=inst["near_mask"],
                    spectral_state=int(spectral_state), candidates=[], excluded=[])
        candidates = info["candidates"]
        for gkey, group in ns["qaoa_results"].items():
            if gkey[:4] != key:
                continue
            variant, optimiser, p = gkey[4], gkey[5], gkey[6]
            if "ablation" in variant:
                info["excluded"].append(f"{variant} p={p}: modified cost layer (analysis only, does not solve the problem)")
                continue
            run = pick_run(group)
            angles = np.asarray(run.get("final_angles", run.get("angles", [])), dtype=float)
            spec = dict(kind="qaoa", variant=variant, optimiser=optimiser, p=int(p), gammas=angles[:p].tolist(), betas=angles[p:].tolist(),
                        start="plus", mixer="x", n_qubits=n, fixed_regions={}, init=run.get("init", "given schedule"),
                        training_circuits=int(run.get("n_evals", 0)))
            if variant.startswith("warm-start QAOA"):
                flavour = variant[len("warm-start QAOA ("):-1]
                thetas, mixer_axis, c_values, note = warm_start_setup(inst, flavour, spectral_state)
                reference = view_probabilities(t4_view(inst, thetas, mixer_axis), angles)
                spec.update(start="ry", thetas=np.asarray(thetas).tolist(), mixer="warm B" if flavour.endswith("mixer B") else "warm A",
                            warm_start_c=np.asarray(c_values).tolist(), note=note, seeded_by="classical spectral split" if "spectral" in flavour else "classical convex relaxation",
                            J_scaled=model["J_scaled"].tolist(), h_scaled=model["h_scaled"].tolist())
            elif variant == "symmetry-reduced QAOA":
                reduced = ns["t4_reduced_view"](inst)
                reference = np.zeros(2**n)
                reference[0::2] = view_probabilities(reduced, angles)          # reduced state r = full state 2r (region 0 on device A)
                spec.update(n_qubits=n - 1, fixed_regions={"0": 0}, qubit_to_region=list(range(1, n)),
                            J_scaled=reduced["J_scaled"].tolist(), h_scaled=reduced["h_scaled"].tolist())
            else:
                reference = view_probabilities(t4_view(inst), angles)
                spec.update(J_scaled=model["J_scaled"].tolist(), h_scaled=model["h_scaled"].tolist())
            stored = dict(p_opt=float(run["p_opt"]), p_near=float(run["p_near"]))
            assert abs(reference[inst["optimal_states"]].sum() - stored["p_opt"]) < 1e-9, f"reference differs from stored result: {gkey}"
            candidates.append(dict(spec=spec, reference=reference, stored=stored))
        variants = ns["VARQITE"]["variants"]
        for vkey, entry in ns["varqite_results"].items():
            if vkey[:4] != key:
                continue
            name = vkey[4]
            gates, _ = ns["vq_gate_list"](inst, variants[name])
            params = np.asarray(entry["params"], dtype=float)
            reference = Statevector(ns["vq_qiskit_circuit"](gates, params, n)).probabilities()
            spec = dict(kind="varqite", variant=name, optimiser="varQITE (noiseless training, exact expectations)", p=None,
                        gates=[[g[0], list(g[1]), g[2]] for g in gates], params=params.tolist(), n_qubits=n, fixed_regions={},
                        training_circuits=int(entry.get("training_circuits", entry.get("circuits", 0))))
            stored = dict(p_opt=float(reference[inst["optimal_states"]].sum()), p_near=float(reference[inst["near_mask"]].sum()))
            if "p_opt" in entry:
                assert abs(entry["p_opt"] - stored["p_opt"]) < 1e-9, f"varQITE reference differs from stored result: {vkey}"
            candidates.append(dict(spec=spec, reference=reference, stored=stored))
        # classical samples to solution (same algorithms and annealing settings as Task 04; restarts until the optimum is hit)
        hits = {}
        for method in ("simulated annealing", "local search (restarts)"):
            counts = []
            for run in range(CLASSICAL_RUNS):
                rng = np.random.default_rng([6006, s_index, run, len(method)])
                if method == "simulated annealing":
                    counts.append(first_hit_annealing(inst, ns["T4_SA_SETTINGS"][scenario], rng, inst["e_min"], ns["TASK04"]["classical_budget"]))
                else:
                    counts.append(first_hit_local_search(inst, rng, inst["e_min"]))
            hits[method] = counts
            done = [c for c in counts if c is not None]
            print(f"  {scenario}, {method}: median {np.median(done) if len(done) == len(counts) else 'censored'} evaluations to the optimum "
                  f"({len(done)} of {len(counts)} runs hit within {CLASSICAL_CAP:,})")
        info["classical_first_hit"] = hits
        data["scenarios"][scenario] = info
        print(f"  {scenario}: {len(candidates)} candidate circuits, {len(info['excluded'])} excluded (ablations)")
    os.chdir(ROOT)
    with open(OUT / "candidates.pkl", "wb") as handle:
        pickle.dump(data, handle)
    shutil.rmtree(work, ignore_errors=True)
    print(f"saved {OUT / 'candidates.pkl'}")
