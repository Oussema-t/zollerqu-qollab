"""Which QAOA settings really improve QAOA? 12 qubits (larger sizes: scripts/qaoa_scaling.py).

Leakage-free protocol: every parameter is chosen on CALIBRATION meshes only (mesh seeds 101 / 102, never scored), then frozen and
applied unchanged to the TEST instances (the 18 scored instances at 12 qubits: 9 cases x 2 device scenarios, mesh seed 7).
Calibration set 'small' = the Task 04 calibration meshes (cases '2 main' and '6 three sources': 4 instances per scenario);
'large' = all 9 cases on both calibration seeds (18 instances per scenario), to reduce over-fitting of the 2p free angles.
The Hamiltonian, the balance weights (lambda 10 / 3.98), the target shares and the energy normalisation are the notebook's, unchanged.

Settings compared, per scenario and depth p in {1, 2, 3, 4, 5, 6, 8}:
  - LR-QAOA, re-calibrated for each p: the (dgamma, dbeta) grid of Task 04, chosen by mean energy ('energy') or by mean
    P(cost <= 1.05 optimum) ('success');
  - all 2p angles optimised jointly on the calibration set (objective: mean P(within 5 %)), with COBYLA, Nelder-Mead, Powell,
    L-BFGS-B and SPSA under the SAME budget of calibration-objective evaluations, all started from the success-calibrated LR angles;
  - warm-start QAOA (Egger et al. 2021, rounded spectral start, mixer A), LR success-calibrated and fully optimised (COBYLA).
Quiver: not installed in either Python environment of this project (checked: no 'quiver' package), so it is not compared.
References from the existing benchmark (no recomputation): per-instance QAOA + COBYLA p = 2 and the classical baselines.
usage: python scripts/qaoa_tuning.py [small|large] [scenario index] [depths, comma-separated]
       -> results/qaoa_tuning/tuning_n12[_large_calibration].json
       with a scenario index (and depths) only that part runs (file suffix .s<index>[.p<depths>]); 'python scripts/qaoa_tuning.py large
       merge' joins the part files (scenarios and depths are independent: same meshes, optimiser seed = p, so the merged file equals
       a single run).
"""
import contextlib
import gzip
import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "qaoa_tuning"
OUT.mkdir(parents=True, exist_ok=True)
LAYERS = [1, 2, 3, 4, 5, 6, 8]
OPTIMISERS = ["COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B", "SPSA"]
BUDGET = 1500                                   # calibration-objective evaluations per optimiser run (each = all calibration instances)
# 'small': the Task 04 calibration set (cases 2 and 6, mesh seeds 101 / 102: 4 instances per scenario)
# 'large': every benchmark case on mesh seeds 101 / 102 (18 instances per scenario), to test whether 4 instances over-fit
CALIBRATION = sys.argv[1] if len(sys.argv) > 1 else "small"
CALIBRATION_SEEDS = (101, 102)
ONLY = sys.argv[2] if len(sys.argv) > 2 else None        # a scenario index, or 'merge'
if len(sys.argv) > 3:
    LAYERS = [int(p) for p in sys.argv[3].split(",")]
SHOTS = 1000
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:6.0f} s] {message}", flush=True)


def load_namespace():
    nb = nbformat.read(ROOT / "mesh_idea_ZollerQu.ipynb", as_version=4)
    ns = {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
            continue
        source, last = cell.source, cell.source.startswith("# TASK_04_IMPLEMENTATION")
        if last:
            source = source.split("# --------------------------------------------------------------- benchmark ----")[0]
        if cell.source.startswith("# TASK_03B_IMPLEMENTATION"):
            source = source.split("# ---- save the pool and the plans as data")[0]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f"cell{index}", "exec"), ns)
        if last:
            return ns
    raise RuntimeError("Task 04 cell not found")


class Budget(Exception):
    pass


def views_for(ns, instances, warm=False):
    views = []
    for inst in instances:
        if warm:
            _, spectral_state, _ = ns["spectral_split"](inst)
            thetas, axis, _, _ = ns["warm_start_setup"](inst, "rounded spectral, mixer A", spectral_state)
            views.append(ns["t4_view"](inst, thetas, axis))
        else:
            views.append(ns["t4_view"](inst))
    return views


def metrics(ns, view, angles):
    probs = ns["view_probabilities"](view, angles)
    m = ns["distribution_metrics"](probs, view)
    return dict(p_opt=m["p_opt"], p_near=m["p_near"], mean_ratio=m["mean_ratio"], hit_opt=1 - (1 - m["p_opt"]) ** SHOTS,
                hit_near=1 - (1 - m["p_near"]) ** SHOTS)


def lr_calibration(ns, views, p):
    """Task 04 grid; returns the best (dgamma, dbeta) by mean ratio ('energy') and by mean P(near) ('success')."""
    best = {}
    for dg in ns["TASK04"]["lr_grid_gamma"]:
        for db in ns["TASK04"]["lr_grid_beta"]:
            angles = ns["lr_angles"](p, dg, db)
            scores = [metrics(ns, v, angles) for v in views]
            ratio, near = float(np.mean([s["mean_ratio"] for s in scores])), float(np.mean([s["p_near"] for s in scores]))
            for key, value in (("energy", ratio), ("success", near)):
                if key not in best or value > best[key]["score"] + 1e-12:
                    best[key] = dict(dgamma=float(dg), dbeta=float(db), score=value, angles=angles.tolist())
    return best


def optimise(ns, views, start, method, seed):
    """Maximise the mean P(within 5 %) over the calibration views with a hard budget of BUDGET evaluations."""
    state = dict(n=0, best=-np.inf, best_x=np.array(start, float))

    def objective(x):
        if state["n"] >= BUDGET:
            raise Budget
        state["n"] += 1
        value = float(np.mean([metrics(ns, v, x)["p_near"] for v in views]))
        if value > state["best"]:
            state["best"], state["best_x"] = value, np.array(x, float)
        return -value
    try:
        if method == "SPSA":
            rng, x = np.random.default_rng([seed, 7]), np.array(start, float)
            a, c, alpha, gamma, big_a = 0.05, 0.05, 0.602, 0.101, 0.1 * (BUDGET // 2)
            for k in range(BUDGET // 2):
                delta = rng.choice([-1.0, 1.0], size=len(x))
                ck = c / (k + 1) ** gamma
                gradient = (objective(x + ck * delta) - objective(x - ck * delta)) / (2 * ck) * delta
                x = x - a / (k + 1 + big_a) ** alpha * gradient
        else:
            options = {"COBYLA": dict(maxiter=BUDGET, rhobeg=0.1, tol=1e-6), "Nelder-Mead": dict(maxfev=BUDGET, xatol=1e-5, fatol=1e-9),
                       "Powell": dict(maxfev=BUDGET, xtol=1e-5, ftol=1e-9), "L-BFGS-B": dict(maxfun=BUDGET, ftol=1e-12, gtol=1e-8)}[method]
            minimize(objective, np.array(start, float), method=method, options=options)
    except Budget:
        pass
    return state["best_x"], state["best"], state["n"]


def case_mesh(ns, case, seed):
    """The notebook's mesh of a benchmark case for a mesh seed (seed 7 = the benchmark meshes already built by the notebook)."""
    if seed == 7:
        return ns["mesh_cases"][case]
    spec = ns["MESH_CASES"][case]
    source = spec.get("mesh_of", case)
    points, triangles = ns["build_case_mesh"](ns["MESH_CASES"][source], seed)
    return ns["add_case_weights"](spec, ns["check_case_mesh"](source, ns["MESH_CASES"][source], points, triangles))


def large_calibration_instances(ns, scenario, n=12, cases=None):
    """Benchmark cases (default: all 9) on the calibration mesh seeds (never scored) at n regions. Their region assignments are
    frozen in files of their own (results/qaoa_tuning/frozen_regions_calibration[_n<n>].npz, one per size so runs at different sizes
    never write the same file), so data/frozen_regions.npz is never modified."""
    frozen = ns["frozen_regions"]
    extra_path = OUT / ("frozen_regions_calibration.npz" if n == 12 else f"frozen_regions_calibration_n{n}.npz")
    known, saved = set(frozen), {}
    if extra_path.exists():
        saved = dict(np.load(extra_path))
        known -= set(saved)
        frozen.update(saved)
    instances = []
    for case in cases or ns["BENCHMARK_CASES"]:
        for seed in CALIBRATION_SEEDS:
            regions = ns["coarsen_mesh"](case_mesh(ns, case, seed), n, seed)
            ns["check_region_case"]((case, n, seed), regions, n)
            instances.append(ns["t4_build_instance"](regions, case, n, seed, scenario))
    if set(frozen) - known - set(saved):                   # write only when new assignments were created
        np.savez_compressed(extra_path, **{k: v for k, v in frozen.items() if k not in known})
    return instances


def test_scores(ns, test_views, angles):
    rows = [metrics(ns, v, angles) for v in test_views]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | dict(per_instance=[r["p_near"] for r in rows])


def merge():
    base = "tuning_n12.json" if CALIBRATION == "small" else "tuning_n12_large_calibration.json"
    parts = [json.loads(f.read_text()) for f in sorted(OUT.glob(base.replace(".json", ".s*.json")))]
    merged = dict(parts[0], scenarios={})
    for part in parts:
        for scenario, block in part["scenarios"].items():
            slot = merged["scenarios"].setdefault(scenario, dict(block, rows=[]))
            slot["rows"] += block["rows"]
    for block in merged["scenarios"].values():
        block["rows"].sort(key=lambda r: r["p"])               # stable: keeps the method order within a depth
    merged["layers"] = sorted({r["p"] for block in merged["scenarios"].values() for r in block["rows"]})
    (OUT / base).write_text(json.dumps(merged, indent=1))
    print(f"merged {len(parts)} files -> {OUT / base}: {list(merged['scenarios'])}")


if __name__ == "__main__":
    if ONLY == "merge":
        merge()
        sys.exit(0)
    ns = load_namespace()
    log("notebook loaded")
    scenarios = list(ns["TARGET_SHARES"])
    if ONLY is not None:
        scenarios = [scenarios[int(ONLY)]]
    checkpoint = json.load(gzip.open(ROOT / "results" / "task04_checkpoint_small.json.gz", "rt"))["instances"]
    results = dict(protocol=__doc__.split("usage:")[0].strip(), budget=BUDGET, layers=LAYERS, calibration=CALIBRATION,
                   quiver="not installed (no 'quiver' package in either environment); not compared", scenarios={})
    name = "tuning_n12.json" if CALIBRATION == "small" else "tuning_n12_large_calibration.json"
    if ONLY is not None:
        name = name.replace(".json", f".s{ONLY}.json" if len(sys.argv) <= 3 else f".s{ONLY}.p{'_'.join(map(str, LAYERS))}.json")
    for scenario in scenarios:
        cal_insts = ns["calibration_instances"][scenario] if CALIBRATION == "small" else large_calibration_instances(ns, scenario)
        test_keys = [(case, 12, 7, scenario) for case in ns["BENCHMARK_CASES"]]
        test_insts = [ns["t4_build_instance"](ns["region_cases"][k[:3]], k[0], 12, 7, scenario) for k in test_keys]
        cal_views, test_views = views_for(ns, cal_insts), views_for(ns, test_insts)
        cal_warm, test_warm = views_for(ns, cal_insts, warm=True), views_for(ns, test_insts, warm=True)
        # references from the existing benchmark (same 18 instances, not recomputed)
        refs = {}
        for key in test_keys:
            rec = checkpoint["|".join(map(str, key))]
            for method, c in rec["classical"].items():
                refs.setdefault(method, []).append(c["hit_near"])
            group = rec["qaoa"].get("QAOA|COBYLA|2")
            if group:
                run = next(r for r in group["runs"] if r.get("init") != "random")
                refs.setdefault("QAOA + COBYLA p=2, trained per instance (existing)", []).append(1 - (1 - run["p_near"]) ** SHOTS)
                refs.setdefault("per-shot QAOA + COBYLA p=2 (existing)", []).append(run["p_near"])
        rows = []
        for p in LAYERS:
            lr = lr_calibration(ns, cal_views, p)
            for kind in ("energy", "success"):
                rows.append(dict(method=f"LR-QAOA ({kind}-calibrated)", p=p, angles=lr[kind]["angles"], dgamma=lr[kind]["dgamma"], dbeta=lr[kind]["dbeta"],
                                 calibration_p_near=float(np.mean([metrics(ns, v, lr[kind]["angles"])["p_near"] for v in cal_views])),
                                 calibration_evaluations=len(ns["TASK04"]["lr_grid_gamma"]) * len(ns["TASK04"]["lr_grid_beta"]),
                                 test=test_scores(ns, test_views, lr[kind]["angles"])))
            start = np.array(lr["success"]["angles"])
            for method in OPTIMISERS:
                x, score, used = optimise(ns, cal_views, start, method, seed=p)
                rows.append(dict(method=f"QAOA, all angles optimised ({method})", p=p, angles=x.tolist(), calibration_p_near=score,
                                 calibration_evaluations=used, test=test_scores(ns, test_views, x)))
            wlr = lr_calibration(ns, cal_warm, p)["success"]
            rows.append(dict(method="warm start, LR (success-calibrated)", p=p, angles=wlr["angles"], calibration_p_near=wlr["score"],
                             calibration_evaluations=len(ns["TASK04"]["lr_grid_gamma"]) * len(ns["TASK04"]["lr_grid_beta"]),
                             test=test_scores(ns, test_warm, wlr["angles"]), seeded_by="classical spectral split"))
            x, score, used = optimise(ns, cal_warm, np.array(wlr["angles"]), "COBYLA", seed=p)
            rows.append(dict(method="warm start, all angles optimised (COBYLA)", p=p, angles=x.tolist(), calibration_p_near=score,
                             calibration_evaluations=used, test=test_scores(ns, test_warm, x), seeded_by="classical spectral split"))
            best = max((r for r in rows if r["p"] == p and "warm" not in r["method"]), key=lambda r: r["calibration_p_near"])
            log(f"{scenario}, p = {p}: best on calibration = {best['method']} ({best['calibration_p_near']:.4f}); "
                f"its TEST mean P(near) {best['test']['p_near']:.4f}, hit@1000 {best['test']['hit_near']:.3f}")
        results["scenarios"][scenario] = dict(rows=rows, references={k: float(np.mean(v)) for k, v in refs.items()},
                                              test_instances=[list(k) for k in test_keys])
        (OUT / name).write_text(json.dumps(results, indent=1))
    log(f"saved {OUT / name}")
