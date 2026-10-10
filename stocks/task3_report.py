"""Task 3: run the classical grid (mode x eta x balance) for each size, evaluate on held-out weeks, report.

Run from the repo root:  python -m stocks.task3_report
Classical only (no circuits). Writes results/stocks_task3_classical.json.
Nothing here is chosen with test data: masks, models and solver settings use training rows only,
and every configuration of the grid is reported.
"""
import json
import subprocess
import sys

import numpy as np

from stocks import classical_evaluation as evaluation
from stocks import classical_solvers as solvers
from stocks import graph, ising, kaggle_data, returns
from stocks.task2_report import git_checks, run_tests

ROOT = kaggle_data.DATA_DIR.parent
RESULTS_PATH = ROOT / "results" / "stocks_task3_classical.json"
SIZES = (12, 29)
BALANCE_GRID = (False, True)
SETTINGS = {
    "seeds": (2026, 2027, 2028),      # the same seeds for every configuration
    "random_samples": 1000,           # uniform random states per seed
    "greedy_restarts": 30,            # random starts per seed
    "sa_restarts": 30,                # annealing restarts per seed
    "sa_sweeps": 200,                 # one sweep = n proposed moves per restart
    "random_portfolios": 1000,        # held-out reference distribution
    "portfolio_seed": 12345,
    "penalty_max_n": 12,              # balance by penalty up to this size, hard constraint above
}


def load_context(train_fraction=returns.DEFAULT_TRAIN_FRACTION):
    weekly = kaggle_data.load_csv()
    log_r = returns.log_returns(weekly.prices)
    simple_r = evaluation.simple_returns(weekly.prices)
    if not np.allclose(np.log1p(simple_r), log_r, rtol=0, atol=1e-12):
        raise RuntimeError("simple and log returns are not aligned row by row")
    split = returns.chronological_split(len(log_r), train_fraction)
    return {"weekly": weekly, "log_r": log_r, "simple_r": simple_r, "dates": weekly.dates[1:], "split": split,
            "benchmark_col": weekly.tickers.index(kaggle_data.BENCHMARK[0])}


def size_setup(ctx, n, keep_fraction=None):
    """Training statistics, correlation mask and held-out context for one size."""
    keep_fraction = ising.keep_fraction_for(n) if keep_fraction is None else keep_fraction
    tickers = kaggle_data.select_assets(n)
    columns = [ctx["weekly"].tickers.index(t) for t in tickers]
    split = ctx["split"]
    stats = graph.training_stats(ctx["log_r"][split.train][:, columns], tickers)
    mask, mask_report = ising.correlation_mask(stats.corr, keep_fraction)
    test_assets = ctx["simple_r"][split.test][:, columns]
    test_benchmark = ctx["simple_r"][split.test][:, ctx["benchmark_col"]]
    equal_weight = evaluation.series_metrics(test_assets.mean(axis=1))
    return {"n": n, "tickers": tickers, "groups": [kaggle_data.GROUP_OF[t] for t in tickers], "stats": stats,
            "mask": mask, "mask_report": mask_report, "keep_fraction": keep_fraction,
            "test_assets": test_assets, "test_benchmark": test_benchmark,
            "benchmark_metrics": evaluation.series_metrics(test_benchmark), "equal_weight_metrics": equal_weight}


def build_model(setup, mode, eta, balance, target_buys=None, dense=False, settings=SETTINGS):
    n, stats = setup["n"], setup["stats"]
    method = None
    if balance:
        method = ising.BALANCE_PENALTY if n <= settings["penalty_max_n"] else ising.BALANCE_HARD
    mask = None if dense else setup["mask"]
    return ising.build_ising(stats.mu, stats.cov, mode, eta, keep_mask=mask, balance=balance, balance_method=method,
                             target_buys=target_buys if (balance and mode == ising.MODE_BUY_OR_SKIP) else None)


def _run_solvers(model, settings):
    """Each stochastic solver once per seed. Returns {solver: [SolverResult per seed]}."""
    runs = {"random": [], "greedy": [], "annealing": []}
    for seed in settings["seeds"]:
        runs["random"].append(solvers.random_search(model, settings["random_samples"], np.random.default_rng(seed)))
        runs["greedy"].append(solvers.greedy_restarts(model, settings["greedy_restarts"], np.random.default_rng(seed)))
        runs["annealing"].append(solvers.simulated_annealing(model, settings["sa_restarts"], settings["sa_sweeps"],
                                                              np.random.default_rng(seed)))
    return runs


def run_configuration(setup, mode, eta, balance, target_buys=None, settings=SETTINGS):
    """One grid point: model, exact or best-found solution, baselines, dense re-score, held-out metrics."""
    n, stats, tickers = setup["n"], setup["stats"], setup["tickers"]
    model = build_model(setup, mode, eta, balance, target_buys, settings=settings)
    runs = _run_solvers(model, settings)
    best_run = min((r for rs in runs.values() for r in rs), key=lambda r: r.best_energy)

    exact = solvers.exact_solve(model) if n <= ising.MAX_ENUMERATION_N else None
    if exact is not None:
        reference, tolerance, chosen = exact.e_min, exact.tolerance, exact.optimal_states[0]
        label = "exact optimum"
    else:
        reference, chosen, label = best_run.best_energy, best_run.best_spins, "best found (not proven optimal)"
        tolerance = solvers.TIE_RTOL * max(abs(reference), 1e-300)

    solver_stats = {}
    for name, results in runs.items():
        hits = [float(np.mean(r.energies <= reference + tolerance)) for r in results]
        best = min(r.best_energy for r in results)
        solver_stats[name] = {
            "best_energy": best,
            "hit_rate_mean": float(np.mean(hits)),      # share of samples/restarts reaching the reference energy
            "hit_rate_std": float(np.std(hits)),
            "seeds_reaching_reference": int(sum(r.best_energy <= reference + tolerance for r in results)),
            "quality": solvers.normalized_quality(best, exact.e_min, exact.e_max) if exact is not None else None,
        }

    dense_score = ising.dense_scores(chosen, stats.mu, stats.cov, mode, eta)
    thinned = model.kept_edges < n * (n - 1) // 2
    dense_best = None
    if thinned:  # how much does the thinned surrogate cost on the dense objective?
        dense_model = build_model(setup, mode, eta, balance, target_buys, dense=True, settings=settings)
        dense_runs = _run_solvers(dense_model, settings)["annealing"]
        best_dense_run = min(dense_runs, key=lambda r: r.best_energy)
        dense_best = ising.dense_scores(best_dense_run.best_spins, stats.mu, stats.cov, mode, eta)

    expected, variance = ising.score_parts(chosen, stats.mu, stats.cov, mode)
    held_out = evaluation.heldout_metrics(chosen, mode, setup["test_assets"], setup["test_benchmark"])
    random_metrics = evaluation.random_portfolio_metrics(model, setup["test_assets"], settings["random_portfolios"],
                                                         np.random.default_rng(settings["portfolio_seed"]))
    assignment = ising.describe(chosen, tickers, mode)
    plus_label = "long" if mode == ising.MODE_LONG_SHORT else "buy"
    return {
        "n": n, "mode": mode, "eta": float(eta), "balance": bool(balance), "balance_method": model.balance_method,
        "target_buys": model.target_buys, "target_sum": model.target_sum, "penalty_lambda": float(model.penalty_lambda),
        "keep_fraction": float(setup["keep_fraction"]), "kept_edges": model.kept_edges,
        "total_edges": n * (n - 1) // 2, "pair_terms_in_model": len(model.pair_terms()),
        "const": float(model.const), "field_min": float(model.fields.min()), "field_max": float(model.fields.max()),
        "reference": label, "model_energy": float(model.energy(chosen)), "dense_score": float(dense_score),
        "dense_best_found": None if dense_best is None else float(dense_best),
        "n_states": None if exact is None else exact.n_states,
        "search_space": None if exact is None else exact.search_space,
        "n_optimal": None if exact is None else exact.n_optimal,
        "e_max": None if exact is None else exact.e_max,
        "penalty_check": solvers.verify_penalty(exact) if (exact is not None and model.balance_method == ising.BALANCE_PENALTY) else None,
        "feasible": bool(model.is_feasible(chosen)),
        "solvers": solver_stats,
        "spins": [int(v) for v in chosen], "bitstring": ising.spins_to_bitstring(chosen),
        "assignment": assignment, "n_plus": len(assignment[plus_label]),
        "train_expected_return": expected, "train_variance": variance,
        "test": held_out,
        "test_percentile_mean": evaluation.percentile_rank(held_out["mean"], random_metrics["mean"]),
        "test_percentile_volatility": evaluation.percentile_rank(held_out["volatility"], random_metrics["volatility"]),
        "random_mean_median": float(np.median(random_metrics["mean"])),
        "random_volatility_median": float(np.median(random_metrics["volatility"])),
    }


def run_grid(ctx, n, keep_fraction=None, etas=ising.ETA_GRID, modes=ising.MODES, balances=BALANCE_GRID,
             target_buys=None, settings=SETTINGS):
    """All configurations for one size. Returns (setup, [result per configuration])."""
    setup = size_setup(ctx, n, keep_fraction)
    results = [run_configuration(setup, mode, eta, balance, target_buys, settings)
               for mode in modes for balance in balances for eta in etas]
    return setup, results


def save_results(setups, results, path=RESULTS_PATH, settings=SETTINGS):
    path.parent.mkdir(exist_ok=True)
    payload = {
        "task": "Task 3: Ising encoding, exact and classical baselines (classical only, no circuits)",
        "generated_by": "python -m stocks.task3_report",
        "numpy": np.__version__,
        "settings": {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()},
        "eta_grid": list(ising.ETA_GRID), "modes": list(ising.MODES),
        "sizes": {str(n): {
            "tickers": s["tickers"], "keep_fraction": s["keep_fraction"],
            "mask": {k: v for k, v in s["mask_report"].items() if k != "dropped_pairs"},
            "train_rows": s["stats"].n_obs, "test_rows": int(len(s["test_assets"])),
            "benchmark_test": s["benchmark_metrics"], "equal_weight_test": s["equal_weight_metrics"],
        } for n, s in setups.items()},
        "results": [r for n in setups for r in results[n]],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def _line(r):
    kind = "L/S" if r["mode"] == ising.MODE_LONG_SHORT else "buy"
    plus = f"{r['n_plus']}{'L' if kind == 'L/S' else 'B'}/{r['n'] - r['n_plus']}{'S' if kind == 'L/S' else '-'}"
    t = r["test"]
    return (f"    {r['mode']:<11} eta {r['eta']:<4} bal {'on ' if r['balance'] else 'off'} | E_model {r['model_energy']:+.5f} "
            f"dense {r['dense_score']:+.5f} | {plus:>8} | hit rnd/grd/SA {r['solvers']['random']['hit_rate_mean']:.3f}/"
            f"{r['solvers']['greedy']['hit_rate_mean']:.2f}/{r['solvers']['annealing']['hit_rate_mean']:.2f} | test mean "
            f"{100 * t['mean']:+.3f}% vol {100 * t['volatility']:.2f}% worst {100 * t['worst_week']:+.1f}% | pct mean/vol "
            f"{r['test_percentile_mean']:.0f}/{r['test_percentile_volatility']:.0f}")


def main(train_fraction=returns.DEFAULT_TRAIN_FRACTION, sizes=SIZES, keep_fraction=None, target_buys=None, with_tests=True):
    ctx = load_context(train_fraction)
    split, dates = ctx["split"], ctx["dates"]
    setups, results = {}, {}
    for n in sizes:
        fraction = None if keep_fraction is None else ising.keep_fraction_for(n, keep_fraction)
        setups[n], results[n] = run_grid(ctx, n, fraction, target_buys=target_buys)

    head = subprocess.run(["git", "log", "--oneline", "-1"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("[TASK 3] Ising encoding, exact and classical baselines (classical only)")
    print(f"- Branch {branch}, HEAD before this commit: {head}")
    print(f"- Split: train {split.split_index} weeks ({dates[split.train][0]} to {dates[split.train][-1]}), test "
          f"{len(dates) - split.split_index} weeks ({dates[split.test][0]} to {dates[split.test][-1]}); nothing tuned on test")
    print(f"- Grid per size: modes {list(ising.MODES)} x eta {list(ising.ETA_GRID)} x balance [off, on] = "
          f"{len(ising.MODES) * len(ising.ETA_GRID) * 2} configurations; seeds {list(SETTINGS['seeds'])}")
    ok = True
    for n in sizes:
        s, rep = setups[n], setups[n]["mask_report"]
        isolated = ", ".join(s["tickers"][k] for k in rep["isolated_nodes"]) or "none"
        print(f"- n={n}: keep fraction {rep['keep_fraction']}, kept {rep['kept_edges']} of {rep['total_edges']} pairs by |rho|, "
              f"kept negative links {rep['kept_negative_edges']} of {rep['negative_edges_total']}, isolated: {isolated}")
        exact = [r for r in results[n] if r["n_optimal"] is not None]
        if exact:
            checks = [r["penalty_check"] for r in exact if r["penalty_check"]]
            ok &= all(c["ok"] for c in checks)
            print(f"    exact enumeration of {exact[0]['n_states']} states per configuration; tied optima per configuration: "
                  f"{sorted({r['n_optimal'] for r in exact})}; penalty check (no unbalanced state beats the best balanced): "
                  f"{sum(c['ok'] for c in checks)}/{len(checks)} pass, smallest margin {min(c['margin'] for c in checks):.4f}")
        else:
            gaps = [r["dense_score"] - r["dense_best_found"] for r in results[n] if r["dense_best_found"] is not None]
            print(f"    no enumeration at this size: results are BEST FOUND, not proven optimal; balance by hard constraint; "
                  f"dense-score cost of thinning (surrogate best minus dense best found): max {max(gaps):+.5f}, "
                  f"mean {np.mean(gaps):+.5f}")
        ok &= all(r["feasible"] for r in results[n])
        b, e = s["benchmark_metrics"], s["equal_weight_metrics"]
        print(f"    held-out context: SPY mean {100 * b['mean']:+.3f}%/week, vol {100 * b['volatility']:.2f}%, worst "
              f"{100 * b['worst_week']:+.1f}%; equal-weight long-only mean {100 * e['mean']:+.3f}%, vol "
              f"{100 * e['volatility']:.2f}%, worst {100 * e['worst_week']:+.1f}%")
        for r in results[n]:
            print(_line(r))
    path = save_results(setups, results)
    print(f"- Results saved: {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print("- Columns: E_model = model energy (sparse surrogate when thinned, incl. penalty); dense = original dense score; "
          "L/S = longs/shorts, B/- = bought/skipped; hit = share of samples or restarts reaching the reference energy; "
          "pct = percentile of the held-out mean and volatility among 1,000 random portfolios of the same mode and balance rule")
    git = git_checks()
    if git is not None:
        changed = subprocess.run(["git", "status", "--porcelain", "common", "mesh"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        print(f"- Raw archive: ignored {git['ignored']}, tracked {git['tracked']}, staged {git['staged']}; "
              f"common/ or mesh changes: {'none' if not changed else changed}")
        ok &= not (git["tracked"] or git["staged"]) and not changed
    if with_tests:
        passed, failed, skipped, log = run_tests()
        print(f"- Tests (python -m unittest discover -s tests -t .): {passed} passed, {failed} failed, {skipped} skipped")
        if failed:
            print(log)
        ok &= failed == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
