"""29-asset QAOA improvement experiment: exact noiseless statevector runs, one factor at a time (offline).

    python -m stocks.exp29 run jobs.json out.jsonl          # simulation jobs (needs numba; a many-core server)
    python -m stocks.exp29 classical 29 buy_or_skip out.json  # classical baselines at several budgets (any machine)

jobs.json is a list of blocks {"problem": {"n", "mode", "kind"}, "jobs": [...]}; every job becomes one JSON line.
The optimum is used ONLY to grade a circuit (p_optimum, hit@B). Angles are selected by quantities a device can
estimate from shots: the expected energy, CVaR(alpha) or a finite-shot mean. Job types:
  eval  - fixed angle sets (linear ramps or explicit angles)
  grid  - linear-ramp grid (dgamma x dbeta) at one depth; reports the point with the lowest expected energy
  opt   - one optimiser run over all 2p angles from a named start, with a hard cap on circuit evaluations
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

from stocks import large_sim, qaoa, qaoa_tuning as tuning, task3_report as t3

BUDGETS = (100, 1000, 10000, 100000)
SCIPY_OPTIONS = {  # as at 12 assets (qaoa_tuning.SCIPY_OPTIONS); L-BFGS-B gets a finite-difference step that a
    "COBYLA": lambda cap: {"maxiter": cap, "rhobeg": 0.2},              # single-precision simulation can resolve
    "Nelder-Mead": lambda cap: {"maxfev": cap, "maxiter": cap},
    "Powell": lambda cap: {"maxfev": cap},
    "L-BFGS-B": lambda cap: {"maxfun": cap, "eps": 0.01},
}


class CapReached(Exception):
    pass


def prepare(n, mode, kind="dense", eta=1.0):
    """Simulator for one problem. kind 'dense' = the full objective; 'thinned' = sparse cost layer, graded on dense."""
    from stocks import fast_sim
    ctx = t3.load_context()
    dense = t3.build_model(t3.size_setup(ctx, n, 1.0), mode, eta, False)
    e_dense = large_sim.energies_all_states(dense)
    if kind == "dense":
        return fast_sim.FastQaoa(dense, e_dense), dense
    thin = t3.build_model(t3.size_setup(ctx, n), mode, eta, False)
    return fast_sim.FastQaoa(thin, large_sim.energies_all_states(thin), e_dense), thin


def measure(sim, gammas, betas, pairs, budgets=BUDGETS):
    started = time.time()
    distribution = sim.evaluate(gammas, betas)
    row = distribution.summary(budgets)
    row.update({"p": int(len(gammas)), "gammas": [float(v) for v in gammas], "betas": [float(v) for v in betas],
                "two_qubit_gates": int(pairs * len(gammas)), "seconds": round(time.time() - started, 2)})
    return distribution, row


def angles_of(point):
    if "gammas" in point:
        return np.array(point["gammas"], dtype=float), np.array(point["betas"], dtype=float)
    return qaoa.lr_schedule(int(point["p"]), float(point["dgamma"]), float(point["dbeta"]))


def run_eval(sim, job, pairs):
    results = []
    for point in job["points"]:
        _, row = measure(sim, *angles_of(point), pairs)
        results.append({**{k: v for k, v in point.items() if k not in ("gammas", "betas")}, **row})
        print(f"    {job['id']} p={row['p']}: quality {row['quality']:.4f}  P(opt) {row['p_optimum']:.3e}  hit@1000 {row['hit@1000']:.4f}  ({row['seconds']} s)", flush=True)
    return {"results": results, "evaluations": len(results)}


def run_grid(sim, job, pairs):
    p, compact, results = int(job["p"]), bool(job.get("compact")), []
    for dgamma in job["dgammas"]:
        for dbeta in job["dbetas"]:
            _, row = measure(sim, *qaoa.lr_schedule(p, dgamma, dbeta), pairs, budgets=(1000,) if compact else BUDGETS)
            if compact:
                row = {k: row[k] for k in ("expectation", "quality", "p_optimum", "cvar_0.1", "hit@1000")}
            results.append({"dgamma": dgamma, "dbeta": dbeta, **row})
    pick = lambda key, sign: (min if sign > 0 else max)(results, key=lambda r: r[key])
    selected = {"by_expectation": pick("expectation", 1), "by_cvar_0.1": pick("cvar_0.1", 1),
                "oracle_best_p_optimum (uses the known optimum; for reference only)": pick("p_optimum", -1)}
    best = selected["by_expectation"]
    edge = best["dgamma"] in (job["dgammas"][0], job["dgammas"][-1]) or best["dbeta"] in (job["dbetas"][0], job["dbetas"][-1])
    print(f"    {job['id']} p={p}: lowest expectation at ({best['dgamma']}, {best['dbeta']}){' [grid edge]' if edge else ''}: quality {best['quality']:.4f}  "
          f"P(opt) {best['p_optimum']:.3e}  hit@1000 {best['hit@1000']:.4f}", flush=True)
    return {"results": results, "selected": selected, "best_on_grid_edge": bool(edge), "evaluations": len(results)}


def run_opt(sim, job, pairs):
    p, cap, objective, start = int(job["p"]), int(job["cap"]), job["objective"], job["start"]
    seed = int(job.get("seed", tuning.START_SEED))
    shot_rng = np.random.default_rng(seed + 1)
    if "random_seed" in start:
        x0 = tuning.random_start("standard", p, np.random.default_rng(int(start["random_seed"])))
    else:
        x0 = np.concatenate(angles_of({**start, "p": p}))
    trajectory = []

    class Target:
        def __init__(self):
            self.cap = cap

        def __call__(self, x):
            if len(trajectory) >= cap:
                raise CapReached
            x = np.asarray(x, dtype=float)
            distribution, row = measure(sim, x[:p], x[p:], pairs)
            if objective == "expectation":
                value = distribution.expectation
            elif objective.startswith("cvar_"):
                value = distribution.cvar(float(objective[5:]))
            elif objective.startswith("shots_"):
                value = distribution.sample_mean(int(objective[6:]), shot_rng)
            else:
                raise ValueError(f"unknown objective {objective!r}")
            row["objective_value"] = value
            trajectory.append(row)
            return value / sim.scale          # same units as the 12-asset tuning

    target, note, error = Target(), "", None
    try:
        if job["optimizer"] == "SPSA":
            tuning.spsa(target, x0, np.random.default_rng(seed))
            note = "finished its steps"
        else:
            result = optimize.minimize(target, x0, method=job["optimizer"], options=SCIPY_OPTIONS[job["optimizer"]](cap))
            note = str(result.message)
    except CapReached:
        note = "stopped at the evaluation cap"
    except Exception as err:   # keep failed runs in the record
        error = f"{type(err).__name__}: {err}"
    selected = min(trajectory, key=lambda r: r["objective_value"]) if trajectory else None
    if selected:
        print(f"    {job['id']}: {len(trajectory)} evaluations, start hit@1000 {trajectory[0]['hit@1000']:.4f} -> selected {selected['hit@1000']:.4f} "
              f"(quality {trajectory[0]['quality']:.4f} -> {selected['quality']:.4f}); {note}", flush=True)
    return {"start_angles": [float(v) for v in x0], "trajectory": trajectory, "selected": selected, "last": trajectory[-1] if trajectory else None,
            "evaluations": len(trajectory), "note": note, "error": error}


HANDLERS = {"eval": run_eval, "grid": run_grid, "opt": run_opt}


def run(jobs_path, out_path):
    import numba
    blocks = json.loads(Path(jobs_path).read_text())
    out_path = Path(out_path)
    done = {json.loads(line).get("id") for line in out_path.read_text().splitlines()} if out_path.exists() else set()
    with out_path.open("a") as out:
        for block in blocks:
            jobs = [job for job in block["jobs"] if job["id"] not in done]
            if not jobs:
                continue
            started = time.time()
            sim, model = prepare(**block["problem"])
            pairs = len(model.pair_terms())
            print(f"problem {block['problem']}: {pairs} pair terms, e_min {sim.e_min:+.6f}, {len(sim.optimal)} optimal state(s), "
                  f"{numba.get_num_threads()} threads, set up in {time.time() - started:.0f} s", flush=True)
            for job in jobs:
                t0 = time.time()
                result = HANDLERS[job["type"]](sim, job, pairs)
                result.update({"id": job["id"], "job": job, "problem": block["problem"], "pairs": pairs, "scale": sim.scale,
                               "e_min": sim.e_min, "e_max": sim.e_max, "n_optimal": int(len(sim.optimal)), "threads": numba.get_num_threads(),
                               "job_seconds": round(time.time() - t0, 1)})
                out.write(json.dumps(result) + "\n")
                out.flush()
            del sim
    print("ALL JOBS DONE", flush=True)


def classical(n, mode, out_path, budgets=BUDGETS, repetitions=200):
    """Greedy, annealing and random sampling at each budget (same code and seeds as large_sim), graded on the exact optimum."""
    exact = json.loads((large_sim.ROOT / "results" / f"stocks_qaoa_n{n}.json").read_text())["modes"][mode]["exact_dense"]
    e_min, e_max = exact["e_min"], exact["e_max"]
    model = t3.build_model(t3.size_setup(t3.load_context(), n, 1.0), mode, 1.0, False)
    out = {"n": n, "mode": mode, "repetitions": repetitions, "e_min": e_min, "e_max": e_max, "budgets": {}}
    for budget in budgets:
        started = time.time()
        out["budgets"][str(budget)] = large_sim.classical_at_budget(model, e_min, e_max, 1e-6 * (e_max - e_min), budget, repetitions)
        out["budgets"][str(budget)]["seconds"] = round(time.time() - started, 1)
        print(budget, out["budgets"][str(budget)], flush=True)
        Path(out_path).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "classical":
        classical(int(sys.argv[2]), sys.argv[3], sys.argv[4])
