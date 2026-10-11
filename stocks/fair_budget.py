"""Same-budget comparison: every method gets exactly B candidate evaluations (default 1,000).

One "candidate evaluation" = computing the energy of one portfolio: one measured shot (quantum), one random
sample, one single-flip neighbour examined by greedy descent, or one Metropolis proposal in simulated annealing.
Reported per method: hit@B = probability that the optimal portfolio is found within the budget, and the quality
of the best portfolio found. The circuits' OFFLINE tuning cost (simulated circuits) is reported separately.
Run from the repo root:  python -m stocks.fair_budget
"""
import json
import sys

import numpy as np

from stocks import classical_solvers as solvers
from stocks import ising, kaggle_data, qaoa_sim, task3_report as t3, task4_report as t4
from stocks import qaoa_tuning as tuning

ROOT = kaggle_data.DATA_DIR.parent
RESULTS_PATH = ROOT / "results" / "stocks_fair_budget_n12.json"
BUDGET = 1000
SEEDS = 200          # independent repetitions per classical method and per sampled quantum circuit
SA_T_END_FACTOR = 1e-3


def greedy_within_budget(model, budget, rng):
    """Restarted steepest-descent (single flips) until `budget` neighbour evaluations are used.

    Each step evaluates all n single-flip neighbours (n evaluations) and moves to the best improving one.
    Returns (best energy seen, evaluations used, restarts).
    """
    n, couplings, fields = model.n, model.couplings, model.fields
    if budget < n:
        raise ValueError(f"budget {budget} is smaller than one greedy step ({n} neighbours)")
    best, used, restarts = np.inf, 0, 0
    while used + n <= budget:
        s = (1 - 2 * rng.integers(0, 2, size=n)).astype(float)
        energy = float(model.energy(s))
        best = min(best, energy)
        restarts += 1
        while used + n <= budget:
            gains = -2.0 * s * (couplings @ s + fields)   # energy change of every single flip
            used += n
            k = int(np.argmin(gains))
            if gains[k] >= 0:
                break
            s[k] = -s[k]
            energy += gains[k]
            best = min(best, energy)
    return best, used, restarts


def annealing_within_budget(model, budget, rng, t_end_factor=SA_T_END_FACTOR):
    """One Metropolis run of exactly `budget` single-flip proposals, geometric temperature schedule.

    t_start = largest possible single-flip energy change (most moves accepted), t_end = t_start * t_end_factor.
    Returns (best energy seen, evaluations used).
    """
    n, couplings, fields = model.n, model.couplings, model.fields
    s = (1 - 2 * rng.integers(0, 2, size=n)).astype(float)
    local = couplings @ s + fields
    energy = float(model.energy(s))
    best = energy
    t_start = 2.0 * float(np.max(np.abs(couplings).sum(axis=1) + np.abs(fields)))
    temperatures = np.geomspace(t_start, t_start * t_end_factor, budget)
    for step, temperature in enumerate(temperatures):
        k = int(rng.integers(n))
        delta = -2.0 * s[k] * local[k]
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            s[k] = -s[k]
            local += 2.0 * s[k] * couplings[k]
            energy += delta
            best = min(best, energy)
    return best, budget


def random_within_budget(model, budget, rng):
    states = solvers.random_spins(budget, model, rng)
    return float(model.energy(states).min()), budget


def quantum_within_budget(probabilities, energies, budget, rng):
    """`budget` shots drawn from the exact distribution of a circuit."""
    index = qaoa_sim.sample(probabilities, budget, rng)
    return float(energies[index].min()), budget


def summarize(best_energies, e_min, e_max, tolerance):
    best = np.asarray(best_energies)
    return {"hit_rate": float(np.mean(best <= e_min + tolerance)),
            "mean_best_quality": float(np.mean([(e_max - e) / (e_max - e_min) for e in best])),
            "repetitions": int(len(best))}


def run(budget=BUDGET, seeds=SEEDS, eta=1.0):
    results = t4.load()
    parameters = json.loads(t4.PARAMETERS_PATH.read_text())
    ctx = t3.load_context()
    setup = t3.size_setup(ctx, 12)
    out = {"budget": budget, "repetitions": seeds, "eta": eta, "n": 12,
           "definition": "one candidate evaluation = the energy of one portfolio: a measured shot, a random sample, "
                         "one greedy neighbour, or one annealing proposal; every method gets exactly `budget` of them",
           "modes": {}}
    for mode in ising.MODES:
        model = t3.build_model(setup, mode, eta, False)
        sim = qaoa_sim.QaoaSimulator(model)
        energies = sim.energies
        e_min, e_max = float(energies.min()), float(energies.max())
        tolerance = solvers.TIE_RTOL * (e_max - e_min)
        rows = {}
        rng = np.random.default_rng(2026)
        rows["random sampling"] = {**summarize([random_within_budget(model, budget, rng)[0] for _ in range(seeds)], e_min, e_max, tolerance),
                                   "offline_tuning_circuits": 0}
        rng = np.random.default_rng(2027)
        greedy = [greedy_within_budget(model, budget, rng) for _ in range(seeds)]
        rows["greedy descent (restarted)"] = {**summarize([g[0] for g in greedy], e_min, e_max, tolerance),
                                              "mean_restarts": float(np.mean([g[2] for g in greedy])), "offline_tuning_circuits": 0}
        rng = np.random.default_rng(2028)
        rows["simulated annealing (one run)"] = {**summarize([annealing_within_budget(model, budget, rng)[0] for _ in range(seeds)], e_min, e_max, tolerance),
                                                 "offline_tuning_circuits": 0}
        key = t4.key_of(mode, False)
        model_result = results["models"][key]
        circuits = {"p = 0 control (uniform)": (sim.probabilities([], []), 0)}
        lr = parameters["modes"][mode]["linear_ramp"]
        best_p = max(lr, key=lambda p: lr[p]["p_optimum_per_shot"])
        grid_points = len(tuning.LR_GRID["dgamma"]) * len(tuning.LR_GRID["dbeta"])
        circuits[f"LR-QAOA p={best_p}"] = (sim.probabilities(lr[best_p]["gammas"], lr[best_p]["betas"]), grid_points)
        standard = model_result["standard"]["best_by_p"]
        p_std = max(standard, key=lambda p: standard[p]["metrics"]["p_optimum_per_shot"])
        total_std = sum(r["evaluations"] for r in model_result["standard_runs"] if str(r["p"]) == str(p_std))
        circuits[f"free-angle QAOA p={p_std}"] = (tuning.probabilities_for(sim, standard[p_std]["angles"]), total_std)
        cvar = results["extras"][key]["cvar_0.1"]
        p_cvar = max(cvar["summary"]["best_by_p"], key=lambda p: cvar["summary"]["best_by_p"][p]["metrics"]["p_optimum_per_shot"])
        total_cvar = sum(r["evaluations"] for r in cvar["runs"] if str(r["p"]) == str(p_cvar))
        circuits[f"CVaR-QAOA (0.1) p={p_cvar}"] = (tuning.probabilities_for(sim, cvar["summary"]["best_by_p"][p_cvar]["angles"]), total_cvar)
        for name, (probabilities, tuning_cost) in circuits.items():
            rng = np.random.default_rng(2029)
            per_shot = float(probabilities[energies <= e_min + tolerance].sum())
            rows[name] = {**summarize([quantum_within_budget(probabilities, energies, budget, rng)[0] for _ in range(seeds)], e_min, e_max, tolerance),
                          "hit_rate_exact": float(1.0 - (1.0 - per_shot) ** budget), "p_optimum_per_shot": per_shot,
                          "offline_tuning_circuits": int(tuning_cost)}
        out["modes"][mode] = {"e_min": e_min, "e_max": e_max, "rows": rows}
    return out


def main():
    out = run()
    RESULTS_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[FAIR BUDGET] {out['budget']} candidate evaluations per method, {out['repetitions']} repetitions, 12 assets, eta {out['eta']}")
    for mode, block in out["modes"].items():
        print(f"- {mode}")
        for name, r in block["rows"].items():
            extra = f", exact {r['hit_rate_exact']:.3f}" if "hit_rate_exact" in r else ""
            restarts = f", {r['mean_restarts']:.1f} restarts" if "mean_restarts" in r else ""
            print(f"    {name:<32} hit@{out['budget']} {r['hit_rate']:.3f}{extra} | mean best quality {r['mean_best_quality']:.4f}"
                  f" | offline tuning circuits {r['offline_tuning_circuits']}{restarts}")
    print("saved:", RESULTS_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
