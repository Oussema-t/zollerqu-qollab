"""Task 4: QAOA circuits, offline tuning and benchmarks on the exact statevector. No backend, no IonQ job.

Run from the repo root:  python -m stocks.task4_report
Writes results/stocks_qaoa_tuning_n12.json (everything that was tried, good or bad) and
results/stocks_qaoa_parameters_n12.json (the fixed angles selected for later runs).
All tuning uses the training-derived Ising objective only; test-period returns are never read here.
Every number is a classical statevector calculation: no quantum hardware and no speed-up is claimed.
"""
import datetime
import json
import platform
import sys
import time

import numpy as np
import qiskit
import scipy
from qiskit.transpiler import CouplingMap

from stocks import classical_solvers as solvers
from stocks import ising, kaggle_data, qaoa, qaoa_sim, task3_report as t3
from stocks import qaoa_tuning as tuning
from stocks import varqite
from stocks.task2_report import run_tests

ROOT = kaggle_data.DATA_DIR.parent
TUNING_PATH = ROOT / "results" / "stocks_qaoa_tuning_n12.json"
PARAMETERS_PATH = ROOT / "results" / "stocks_qaoa_parameters_n12.json"

SETTINGS = {
    "n": 12, "eta": 1.0, "shots": 1000,
    "standard_p": {"balance_off": (1, 2, 3), "balance_on": (1, 2)},   # penalty models are diagnostics: p <= 2
    "optimizers": tuning.OPTIMIZERS,
    "extra_p": (1, 2, 3), "extra_optimizers": ("COBYLA", "L-BFGS-B"), "cvar_alphas": (0.1, 0.25),
    "xy_p": (1, 2),
    "varqite_kinds": ("A", "B"), "varqite_lrs": varqite.LR_CANDIDATES, "varqite_steps": 60,
    "sample_seeds": (2026, 2027, 2028),
    "boltzmann_taus": (0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0),
    "routing_seeds": (2026, 2027, 2028, 2029, 2030),
    "circuit_p": (1, 2, 3, 4, 5), "large_n": 29, "large_p": (1, 2, 3),
    "transfer_tuned_size": 16, "transfer_only_size": 20, "transfer_p": (1, 2, 3),
}
MODE_NAMES = {ising.MODE_LONG_SHORT: "A long/short", ising.MODE_BUY_OR_SKIP: "B buy or skip"}


def key_of(mode, balance):
    return f"{mode}|balance_{'on' if balance else 'off'}"


class Context:
    """One n-asset model with its simulator, its answer key and the rule for scoring a distribution."""

    def __init__(self, setup, mode, balance, eta, keep_dense=False):
        self.setup, self.mode, self.balance = setup, mode, balance
        self.model = t3.build_model(setup, mode, eta, balance, dense=keep_dense)
        self.sim = qaoa_sim.QaoaSimulator(self.model)
        states = ising.all_states(self.model.n)
        self.feasible = self.model.is_feasible(states)
        # objective = the score without any penalty; the penalty is 0 on balanced states
        self.objective = t3.build_model(setup, mode, eta, False, dense=keep_dense).energy(states) if balance else self.sim.energies
        space = self.objective[self.feasible]
        self.e_min, self.e_max = float(space.min()), float(space.max())
        self.tolerance = solvers.TIE_RTOL * (self.e_max - self.e_min)
        self.optimal = self.feasible & (self.objective <= self.e_min + self.tolerance)

    def evaluate(self, probabilities, shots=SETTINGS["shots"]):
        """Metrics of a distribution. With balance on, quality is measured on the balanced shots only."""
        share, kept = qaoa_sim.condition_on(probabilities, self.feasible)
        out = {"feasible_share": share, "model_expectation": float(probabilities @ self.sim.energies)}
        if kept is None:
            out.update({"quality": None, "p_optimum": None, "p_optimum_per_shot": 0.0, "p_optimum_in_shots": 0.0,
                        "note": "no balanced outcome has any probability"})
            return out
        out.update(qaoa_sim.distribution_metrics(kept, self.objective, self.e_min, self.e_max, self.tolerance, shots))
        per_shot = share * out["p_optimum"]            # chance that ONE shot is the best (balanced) portfolio
        out["p_optimum_per_shot"] = per_shot
        out["p_optimum_in_shots"] = float(1.0 - (1.0 - min(per_shot, 1.0)) ** shots)
        return out

    def finite_shots(self, probabilities, shots, seeds):
        """What a backend would return: `shots` samples per seed. Reports hits of the optimum and the best sample."""
        rows = []
        for seed in seeds:
            index = qaoa_sim.sample(probabilities, shots, np.random.default_rng(seed))
            ok = self.feasible[index]
            if not ok.any():
                rows.append({"seed": seed, "balanced_shots": 0, "hit_optimum": False, "best_quality": None})
                continue
            best = float(self.objective[index[ok]].min())
            rows.append({"seed": seed, "balanced_shots": int(ok.sum()), "hit_optimum": bool(best <= self.e_min + self.tolerance),
                         "best_quality": solvers.normalized_quality(best, self.e_min, self.e_max)})
        return rows

    def state_label(self, index):
        spins = ising.all_states(self.model.n)[index]
        return {"bitstring": ising.spins_to_bitstring(spins), "assignment": ising.describe(spins, self.setup["tickers"], self.mode),
                "objective": float(self.objective[index])}


def baselines(c, settings):
    """p = 0 control (uniform), the exact answer, and the Task 3 classical solvers on the same model."""
    runs = t3._run_solvers(c.model, t3.SETTINGS)
    reference = float(c.sim.energies[c.optimal].min())
    tolerance = solvers.TIE_RTOL * max(float(np.ptp(c.sim.energies[c.feasible])), 1e-300)
    classical = {}
    for name, results in runs.items():
        hits = [float(np.mean(r.energies <= reference + tolerance)) for r in results]
        tries = int(len(results[0].energies))
        rate = float(np.mean(hits))
        classical[name] = {"hit_rate_per_try": rate, "tries_per_seed": tries,
                           "p_hit_at_least_once": float(1.0 - (1.0 - rate) ** tries),
                           "seeds_reaching_optimum": int(sum(r.best_energy <= reference + tolerance for r in results)),
                           "seeds": len(results)}
    best_state = int(np.flatnonzero(c.optimal)[0])
    return {"p0_control_uniform": c.evaluate(c.sim.probabilities([], [])), "classical": classical,
            "exact": {"e_min": c.e_min, "e_max": c.e_max, "n_optimal": int(c.optimal.sum()), "states": int(c.sim.dim),
                      "balanced_states": int(c.feasible.sum()), **c.state_label(best_state)}}


def run_optimizers(c, kind, p_values, optimizers, lr, objective="expectation", sim=None):
    """Every optimiser from every named start, same cap. Returns the list of run records."""
    sim = c.sim if sim is None else sim
    runs = []
    for p in p_values:
        starts = tuning.starts_for(kind, p, (lr[p]["best_dgamma"], lr[p]["best_dbeta"]))
        for optimizer in optimizers:
            for k, (name, x0) in enumerate(starts.items()):
                record = tuning.optimize_angles(sim, kind, p, optimizer, x0, objective, seed=tuning.START_SEED + k)
                record["start"] = name
                record["metrics"] = c.evaluate(tuning.probabilities_for(sim, record["angles"])) if record["angles"] else None
                runs.append(record)
    return runs


def summarize(runs):
    """Per (p, optimiser): spread over the starts. Per p: the run with the lowest tuning objective."""
    table, best = {}, {}
    for p in sorted({r["p"] for r in runs}):
        table[p] = {}
        for optimizer in dict.fromkeys(r["optimizer"] for r in runs):
            group = [r for r in runs if r["p"] == p and r["optimizer"] == optimizer]
            good = [r for r in group if r["metrics"] and r["metrics"]["quality"] is not None]
            if not group:
                continue
            table[p][optimizer] = {
                "runs": len(group), "failed": sum(r["error"] is not None for r in group),
                "mean_evaluations": float(np.mean([r["evaluations"] for r in group])),
                "quality_mean": float(np.mean([r["metrics"]["quality"] for r in good])) if good else None,
                "quality_best": float(np.max([r["metrics"]["quality"] for r in good])) if good else None,
                "quality_worst": float(np.min([r["metrics"]["quality"] for r in good])) if good else None,
                "p_optimum_per_shot_mean": float(np.mean([r["metrics"]["p_optimum_per_shot"] for r in good])) if good else None,
                "p_optimum_per_shot_best": float(np.max([r["metrics"]["p_optimum_per_shot"] for r in good])) if good else None,
            }
        candidates = [r for r in runs if r["p"] == p and r["objective_value"] is not None]
        if candidates:
            best[p] = min(candidates, key=lambda r: r["objective_value"])
    return {"by_optimizer": table, "best_by_p": best}


def tune_model(c, settings, log):
    """Core: linear-ramp grid for p = 1..5 and free-angle QAOA with every optimiser."""
    started = time.time()
    lr = tuning.lr_grid_search(c.sim, c.optimal, c.evaluate)
    p_values = settings["standard_p"]["balance_on" if c.balance else "balance_off"]
    runs = run_optimizers(c, "standard", p_values, settings["optimizers"], lr)
    summary = summarize(runs)
    for p, entry in lr.items():
        entry["finite_shots"] = c.finite_shots(c.sim.probabilities(entry["gammas"], entry["betas"]), settings["shots"], settings["sample_seeds"])
    for p, record in summary["best_by_p"].items():
        record["finite_shots"] = c.finite_shots(tuning.probabilities_for(c.sim, record["angles"]), settings["shots"], settings["sample_seeds"])
    log(f"  {key_of(c.mode, c.balance)}: grid + {len(runs)} optimiser runs in {time.time() - started:.0f} s")
    return {"mode": c.mode, "balance": c.balance, "balance_method": c.model.balance_method,
            "penalty_lambda": float(c.model.penalty_lambda), "scale": c.sim.scale,
            "max_abs_coupling": float(np.abs(c.model.couplings).max()), "max_abs_field": float(np.abs(c.model.fields).max()),
            "baselines": baselines(c, settings), "lr": lr, "standard_runs": runs, "standard": summary}


def balance_diagnostics(contexts, models, settings):
    """Balance on: penalty in the circuit, against post-selecting balanced shots of the unconstrained circuit."""
    out = {}
    for mode in ising.MODES:
        free, pen = contexts[key_of(mode, False)], contexts[key_of(mode, True)]
        free_result, pen_result = models[key_of(mode, False)], models[key_of(mode, True)]
        objective_max = max(free_result["max_abs_coupling"], free_result["max_abs_field"])
        rows = {"penalty_coupling": 2.0 * pen.model.penalty_lambda, "largest_objective_term": objective_max,
                "penalty_to_objective_ratio": 2.0 * pen.model.penalty_lambda / objective_max,
                "postselect_uniform_p0": pen.evaluate(free.sim.probabilities([], [])), "by_p": {}}
        for p in settings["standard_p"]["balance_on"]:
            lr_free, lr_pen = free_result["lr"][p], pen_result["lr"][p]
            entry = {
                "penalty_lr": pen.evaluate(pen.sim.probabilities(lr_pen["gammas"], lr_pen["betas"])),
                "postselect_lr": pen.evaluate(free.sim.probabilities(lr_free["gammas"], lr_free["betas"])),
            }
            if p in pen_result["standard"]["best_by_p"]:
                entry["penalty_standard"] = pen.evaluate(tuning.probabilities_for(pen.sim, pen_result["standard"]["best_by_p"][p]["angles"]))
            if p in free_result["standard"]["best_by_p"]:
                entry["postselect_standard"] = pen.evaluate(tuning.probabilities_for(free.sim, free_result["standard"]["best_by_p"][p]["angles"]))
            rows["by_p"][p] = entry
        out[mode] = rows
    return out


# ---------------------------------------------------------------------------------------------- extras


def attempt(name, function, log):
    """Run one optional extra; a failure is recorded instead of stopping the task."""
    started = time.time()
    try:
        result = function()
        status = "completed"
    except Exception as err:
        result, status = {"error": f"{type(err).__name__}: {err}"}, "failed"
    log(f"  extra '{name}': {status} in {time.time() - started:.0f} s")
    return {"status": status, "seconds": round(time.time() - started, 1), **result}


def compare_with_standard(c, runs, standard_runs):
    """Same optimisers, starts, p and cap as the standard runs they are compared with."""
    used = {r["optimizer"] for r in runs}
    reference = [r for r in standard_runs if r["optimizer"] in used and r["p"] in {x["p"] for x in runs}]
    return {"runs": runs, "summary": summarize(runs), "standard_same_budget": summarize(reference)}


def extra_tasks(c, model_result, settings):
    """The optional extras for one unconstrained model, as named callables (run sequentially or in parallel)."""
    lr, standard_runs = model_result["lr"], model_result["standard_runs"]
    p_values, optimizers = settings["extra_p"], settings["extra_optimizers"]
    tasks = {}
    for alpha in settings["cvar_alphas"]:
        tasks[f"cvar_{alpha}"] = lambda a=alpha: compare_with_standard(
            c, run_optimizers(c, "standard", p_values, optimizers, lr, ("cvar", a)), standard_runs)
    tasks["multi_angle"] = lambda: compare_with_standard(c, run_optimizers(c, "multi_angle", p_values, optimizers, lr), standard_runs)
    tasks["dc"] = lambda: compare_with_standard(c, run_optimizers(c, "dc", p_values, optimizers, lr), standard_runs)
    tasks["varqite"] = lambda: {
        "runs": (runs := run_optimizers(c, "standard", p_values, ("VarQITE",), lr)), "summary": summarize(runs),
        "standard_all_optimizers": summarize([r for r in standard_runs if r["p"] in p_values])}
    tasks["varqite_pauli"] = lambda: varqite_extra(c, settings)
    tasks["boltzmann_reference"] = lambda: {
        "note": "exact imaginary-time evolution of the uniform state for a diagonal cost = classical Boltzmann sampling",
        "curve": [{"tau": tau, **c.evaluate(qaoa_sim.boltzmann(c.sim.energies, tau))} for tau in settings["boltzmann_taus"]]}
    return tasks


EXTRA_ORDER = ("cvar_0.1", "cvar_0.25", "multi_angle", "dc", "varqite", "varqite_pauli", "boltzmann_reference")
DEFERRED = {
    "warm_start": "not attempted: multi-angle was the one low-risk option chosen from extra 2",
    "variational_real_time_evolution": "not attempted: highest risk for the time left; future work",
    "recursive_qaoa_and_backend_loops": "excluded: not compatible with one job per run",
}


def extras_for(c, model_result, settings, log):
    name = key_of(c.mode, c.balance)
    return {key: attempt(f"{key} {name}", task, log) for key, task in extra_tasks(c, model_result, settings).items()}


def varqite_extra(c, settings, runs=None):
    """varQITE with the Pauli-term update rule (arXiv 2404.16135), trained offline; the final circuit is sampled.

    `runs`: precomputed training records (from varqite.train_one) to merge instead of training here.
    """
    if runs is None:
        result = varqite.tune(c.model, settings["varqite_kinds"], settings["varqite_lrs"], evaluate=c.evaluate,
                              steps=settings["varqite_steps"])
    else:
        result = varqite.select(runs)
    best = result["best"]
    ansatz = varqite.Ansatz(c.model, best["kind"])
    theta = np.array(best["theta"])
    circuit = ansatz.to_qiskit(theta)
    cross_check = float(np.abs(ansatz.probabilities(theta) - qaoa.statevector_probabilities(circuit)).max())
    result.update({"citation": varqite.CITATION, "circuit_metrics": qaoa.circuit_metrics(circuit),
                   "numpy_vs_qiskit_max_difference": cross_check,
                   "finite_shots": c.finite_shots(ansatz.probabilities(theta), settings["shots"], settings["sample_seeds"]),
                   "note": "trained on a noiseless local statevector; only the final circuit would be sampled on a backend"})
    return result


def ring_pairs(n):
    """Neighbour pairs on a ring, even pairs first, then odd pairs (two rounds of disjoint gates for even n)."""
    pairs = [(q, (q + 1) % n) for q in range(n)]
    return pairs[0::2] + pairs[1::2]


def dicke_state(n, ones):
    """Equal superposition of all basis states with exactly `ones` bits set to 1."""
    index = np.arange(1 << n)
    weight = np.array([bin(k).count("1") for k in index]) == ones
    return weight / np.sqrt(weight.sum())


class XYSimulator(qaoa_sim.QaoaSimulator):
    """QAOA that never leaves the balanced states: Dicke start, ring of XX+YY rotations as the mixer."""

    def __init__(self, model, ones):
        super().__init__(model)
        self.pairs, self.ones = ring_pairs(model.n), ones
        self.start = dicke_state(model.n, ones).astype(complex)

    def amplitudes(self, gammas, betas, **_):
        return super().amplitudes(gammas, betas, initial=self.start, xy_pairs=self.pairs)


def xy_extra(contexts, models, settings):
    """Constraint-preserving mixer, compared with the penalty and with post-selection (same p, same caps)."""
    out = {}
    for mode in ising.MODES:
        free, pen = contexts[key_of(mode, False)], contexts[key_of(mode, True)]
        ones = (free.model.n - pen.model.target_sum) // 2
        sim = XYSimulator(free.model, ones)   # objective without penalty
        lr = models[key_of(mode, False)]["lr"]
        runs = run_optimizers(pen, "standard", settings["xy_p"], settings["extra_optimizers"], lr, sim=sim)
        cross_check = float(np.abs(sim.probabilities([0.4], [0.3]) - qaoa.statevector_probabilities(
            build_xy_circuit(free.model, [0.4], [0.3], sim.pairs, sim.start))).max())
        mixer_gates = len(sim.pairs)
        out[mode] = {"runs": runs, "summary": summarize(runs), "ones": int(ones), "mixer_pairs": sim.pairs,
                     "xx_plus_yy_gates_per_layer": mixer_gates, "numpy_vs_qiskit_max_difference": cross_check,
                     "note": "Dicke start is prepared with Qiskit's generic StatePreparation here (correct, not gate-efficient); "
                             "an efficient Dicke circuit is not implemented, so no total gate count is claimed"}
    return out


def build_xy_circuit(model, gammas, betas, pairs, start, scale=None):
    """Qiskit version of XYSimulator's circuit, for the cross-check only."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, XXPlusYYGate
    scale = qaoa_sim.cost_scale(model) if scale is None else scale
    circuit = QuantumCircuit(model.n)
    circuit.append(StatePreparation(np.asarray(start, dtype=complex)), range(model.n))
    for gamma, beta in zip(gammas, betas):
        for i, j, coupling in qaoa.layered_pair_terms(model.pair_terms()):
            circuit.rzz(2.0 * gamma * coupling / scale, i, j)
        for i in range(model.n):
            if model.fields[i] != 0.0:
                circuit.rz(2.0 * gamma * model.fields[i] / scale, i)
        for a, b in pairs:
            circuit.append(XXPlusYYGate(2.0 * beta), [a, b])
    return circuit


# ------------------------------------------------------------------------------- circuits and transfer


def routing_summary(circuit, heavy_hex_distance, seeds):
    flat = qaoa.routed_metrics(circuit)
    routed = [qaoa.routed_metrics(circuit, CouplingMap.from_heavy_hex(heavy_hex_distance), seed) for seed in seeds]
    counts = sorted(r["two_qubit_gates"] for r in routed)
    return {"basis": qaoa.ROUTING_BASIS, "optimization_level": 1, "all_to_all_two_qubit_gates": flat["two_qubit_gates"],
            "all_to_all_two_qubit_depth": flat["two_qubit_depth"],
            "heavy_hex_qubits": CouplingMap.from_heavy_hex(heavy_hex_distance).size(), "heavy_hex_seeds": list(seeds),
            "heavy_hex_two_qubit_gates": {"min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]},
            "heavy_hex_two_qubit_depth_min": min(r["two_qubit_depth"] for r in routed)}


def circuit_report(ctx, contexts, models, settings):
    """Gate counts for n = 12 and, construction only, for n = 29 (never simulated)."""
    out = {"n12": {}, "large": {}}
    for mode in ising.MODES:
        c = contexts[key_of(mode, False)]
        lr = models[key_of(mode, False)]["lr"]
        per_p = {p: qaoa.circuit_metrics(qaoa.build_circuit(c.model, lr[p]["gammas"], lr[p]["betas"])) for p in settings["circuit_p"]}
        out["n12"][mode] = {"by_p": per_p, "routing_p1": routing_summary(
            qaoa.build_circuit(c.model, lr[1]["gammas"], lr[1]["betas"]), 3, settings["routing_seeds"])}
    large_setup = t3.size_setup(ctx, settings["large_n"])
    for mode in ising.MODES:
        model = t3.build_model(large_setup, mode, settings["eta"], False)
        lr = models[key_of(mode, False)]["lr"]
        circuits = {p: qaoa.build_circuit(model, *qaoa.lr_schedule(p, lr[p]["best_dgamma"], lr[p]["best_dbeta"])) for p in settings["large_p"]}
        out["large"][mode] = {
            "n": settings["large_n"], "pair_terms": len(model.pair_terms()), "scale": qaoa_sim.cost_scale(model),
            "keep_fraction": large_setup["keep_fraction"], "simulated": False,
            "angles_from": f"linear ramp tuned at n = {settings['n']} (same mode), NOT tuned for n = {settings['large_n']}",
            "by_p": {p: {**qaoa.circuit_metrics(circuit), "dgamma": lr[p]["best_dgamma"], "dbeta": lr[p]["best_dbeta"]}
                     for p, circuit in circuits.items()},
            "routing_p1": routing_summary(circuits[settings["large_p"][0]], 5, settings["routing_seeds"]),
        }
    return out


def transfer_report(ctx, models, settings, log):
    """Does the ramp tuned at n = 12 still work at a larger size? Measured where the statevector is still feasible."""
    out = {"rule": f"for a larger size use the linear ramp (dgamma, dbeta) tuned at n = {settings['n']} for the same mode and p, "
                   f"with the larger model's own scale; it is not re-tuned"}
    for label, n, own_grid, keep in (("tuned_size", settings["transfer_tuned_size"], True, None),
                                     ("larger_size_dense", settings["transfer_only_size"], False, None),
                                     ("larger_size_thinned", settings["transfer_only_size"], False, ising.DEFAULT_KEEP_FRACTION[29])):
        started = time.time()
        setup = t3.size_setup(ctx, n, keep)
        entry = {"n": n, "keep_fraction": setup["keep_fraction"], "pair_terms": setup["mask_report"]["kept_edges"], "modes": {}}
        for mode in ising.MODES:
            c = Context(setup, mode, False, settings["eta"])
            lr12 = models[key_of(mode, False)]["lr"]
            row = {"uniform_p0": c.evaluate(c.sim.probabilities([], [])), "transferred": {}}
            thinned = setup["mask_report"]["dropped_edges"] > 0
            if thinned:  # score the same distributions on the DENSE objective as well
                dense = Context(setup, mode, False, settings["eta"], keep_dense=True)
                row["uniform_p0_dense"] = dense.evaluate(c.sim.probabilities([], []))
                row["transferred_dense"] = {}
            for p in settings["transfer_p"]:
                probabilities = c.sim.probabilities(*qaoa.lr_schedule(p, lr12[p]["best_dgamma"], lr12[p]["best_dbeta"]))
                row["transferred"][p] = c.evaluate(probabilities)
                if thinned:
                    row["transferred_dense"][p] = dense.evaluate(probabilities)
            if own_grid:
                grid = tuning.lr_grid_search(c.sim, c.optimal, c.evaluate, settings["transfer_p"])
                row["own_grid"] = {p: {"best_dgamma": g["best_dgamma"], "best_dbeta": g["best_dbeta"], **g["best"]} for p, g in grid.items()}
            entry["modes"][mode] = row
        entry["seconds"] = round(time.time() - started, 1)
        out[label] = entry
        log(f"  transfer check n = {n} ({label}): {entry['seconds']} s")
    return out


# --------------------------------------------------------------------------------------------- run all


def meta_block(setup, split, dates, settings):
    return {
        "task": "Task 4: QAOA circuits, offline tuning and benchmarks (statevector only; no backend run)",
        "generated_by": "python -m stocks.task4_report",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "qiskit": qiskit.__version__},
        "note": "all tuning is offline on a classical statevector and uses the training-derived objective only; "
                "no test-period data, no quantum hardware, no speed-up claim",
        "training_rows": int(split.split_index), "training_first": str(dates[split.train][0]), "training_last": str(dates[split.train][-1]),
        "tickers": setup["tickers"], "keep_fraction": setup["keep_fraction"], "kept_links": setup["mask_report"]["kept_edges"],
        "scale_policy": qaoa_sim.SCALE_POLICY,
        "sign_note": "mixer is rx(+2 beta): lowering the energy needs gamma and beta of opposite sign; the ramp 0.6 / 0.3 "
                     "as written (same sign) is kept as a measured control",
        "settings": json.loads(json.dumps(settings)),
        "tuning": {"lr_grid": tuning.LR_GRID, "lr_p": list(tuning.LR_P_VALUES), "optimizers": list(tuning.OPTIMIZERS),
                   "evaluation_cap_per_layer": tuning.EVALUATIONS_PER_LAYER, "start_seed": tuning.START_SEED,
                   "random_starts": tuning.N_RANDOM_STARTS, "random_start_box": tuning.RANDOM_START_BOX,
                   "scipy_options_at_cap_100": {k: v(100) for k, v in tuning.SCIPY_OPTIONS.items()},
                   "spsa": tuning.SPSA_SETTINGS, "varqite": tuning.VARQITE_SETTINGS,
                   "selection_rule": "lowest expected model energy (the tuning objective); never P(optimum), never test data"},
        "classical_baseline_settings": {k: list(v) if isinstance(v, tuple) else v for k, v in t3.SETTINGS.items()},
    }


def run_all(settings=SETTINGS, log=print, with_extras=True, with_transfer=True):
    started = time.time()
    ctx = t3.load_context()
    setup = t3.size_setup(ctx, settings["n"])
    split, dates = ctx["split"], ctx["dates"]
    out = {"meta": meta_block(setup, split, dates, settings)}
    log("Task 4 run: core tuning")
    contexts = {key_of(mode, balance): Context(setup, mode, balance, settings["eta"]) for mode in ising.MODES for balance in (False, True)}
    out["models"] = {key: tune_model(c, settings, log) for key, c in contexts.items()}
    out["balance"] = balance_diagnostics(contexts, out["models"], settings)
    out["circuits"] = circuit_report(ctx, contexts, out["models"], settings)
    out["extras"] = {}
    if with_extras:
        log("Task 4 run: extras (each is optional; failures are recorded)")
        for mode in ising.MODES:
            key = key_of(mode, False)
            out["extras"][key] = extras_for(contexts[key], out["models"][key], settings, log)
        out["extras"]["xy_mixer"] = attempt("XY mixer with Dicke start", lambda: {"modes": xy_extra(contexts, out["models"], settings)}, log)
    out["extras"]["deferred"] = dict(DEFERRED)
    out["transfer"] = attempt("parameter transfer check", lambda: transfer_report(ctx, out["models"], settings, log), log) \
        if with_transfer else {"status": "skipped"}
    out["meta"]["seconds"] = round(time.time() - started, 1)
    return out


def selected_parameters(results):
    """The fixed angles a later task can run without any further tuning."""
    out = {"meta": {k: results["meta"][k] for k in ("task", "timestamp_utc", "versions", "note", "training_rows", "training_first",
                                                    "training_last", "tickers", "keep_fraction", "kept_links", "scale_policy", "sign_note")},
           "circuit_convention": "H on all qubits; per layer rzz(2*gamma*J_ij/scale), rz(2*gamma*h_i/scale), rx(2*beta); "
                                 "spin +1 = |0>; count keys have qubit 0 as the rightmost character",
           "selection_rule": results["meta"]["tuning"]["selection_rule"], "modes": {}}
    for mode in ising.MODES:
        result = results["models"][key_of(mode, False)]
        lr = {str(p): {k: entry[k] for k in ("best_dgamma", "best_dbeta", "gammas", "betas", "best_on_grid_edge")} |
              {"quality": entry["best"]["quality"], "p_optimum_per_shot": entry["best"]["p_optimum_per_shot"]} for p, entry in result["lr"].items()}
        standard = {str(p): {"gammas": r["angles"]["gammas"], "betas": r["angles"]["betas"], "optimizer": r["optimizer"], "start": r["start"],
                             "evaluations": r["evaluations"], "quality": r["metrics"]["quality"],
                             "p_optimum_per_shot": r["metrics"]["p_optimum_per_shot"]} for p, r in result["standard"]["best_by_p"].items()}
        out["modes"][mode] = {
            "eta": results["meta"]["settings"]["eta"], "balance": False, "scale": result["scale"],
            "exact_optimum": result["baselines"]["exact"], "p0_control": {"gammas": [], "betas": []},
            "linear_ramp": lr, "free_angles_best": standard,
        }
    out["transfer_rule_larger_sizes"] = results["transfer"].get("rule", "use the n = 12 linear ramp for the same mode and p; not re-tuned")
    return out


def save(results):
    TUNING_PATH.parent.mkdir(exist_ok=True)
    TUNING_PATH.write_text(json.dumps(results, indent=1, default=_plain) + "\n")
    PARAMETERS_PATH.write_text(json.dumps(selected_parameters(results), indent=1, default=_plain) + "\n")
    return TUNING_PATH, PARAMETERS_PATH


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot store {type(value).__name__}")


def load():
    return json.loads(TUNING_PATH.read_text())


def print_report(results, with_tests=True):
    shots = results["meta"]["settings"]["shots"]
    print("[TASK 4] QAOA circuits, offline tuning and benchmarks (classical statevector only, no backend run)")
    print(f"- Generated {results['meta']['timestamp_utc']} in {results['meta']['seconds']} s; versions {results['meta']['versions']}")
    print(f"- Training-only objective, n = {results['meta']['settings']['n']} assets, eta = {results['meta']['settings']['eta']}; "
          f"selection rule: {results['meta']['tuning']['selection_rule']}")
    print(f"- {results['meta']['sign_note']}")
    for key, r in results["models"].items():
        base = r["baselines"]
        u = base["p0_control_uniform"]
        print(f"- {key}: scale {r['scale']:.4g}; exact optimum {base['exact']['e_min']:+.5f} ({base['exact']['n_optimal']} state)")
        print(f"    p=0 control (uniform): quality {u['quality']:.3f}, P(optimum) per shot {u['p_optimum_per_shot']:.5f}, "
              f"in {shots} shots {u['p_optimum_in_shots']:.3f}")
        for p, e in r["lr"].items():
            b, w = e["best"], e["recipe_as_written"]
            print(f"    LR p={p}: best ramp ({e['best_dgamma']}, {e['best_dbeta']}){' [grid edge]' if e['best_on_grid_edge'] else ''} "
                  f"quality {b['quality']:.3f}, P(opt)/shot {b['p_optimum_per_shot']:.4f}, in {shots} shots {b['p_optimum_in_shots']:.3f}"
                  f" | recipe as written (0.6, 0.3): quality {w['quality']:.3f}, P(opt)/shot {w['p_optimum_per_shot']:.5f}")
        for p, best in r["standard"]["best_by_p"].items():
            m = best["metrics"]
            print(f"    free angles p={p}: best {best['optimizer']} from {best['start']} ({best['evaluations']} evaluations): "
                  f"quality {m['quality']:.3f}, P(opt)/shot {m['p_optimum_per_shot']:.4f}, in {shots} shots {m['p_optimum_in_shots']:.3f}")
            row = ", ".join(f"{name} {v['quality_mean']:.3f} ({v['mean_evaluations']:.0f} ev)" for name, v in r["standard"]["by_optimizer"][p].items()
                            if v["quality_mean"] is not None)
            print(f"      mean quality over the starts: {row}")
        c = base["classical"]
        print("    classical on the same model: " + "; ".join(
            f"{name} {v['hit_rate_per_try']:.4f} per try, {v['p_hit_at_least_once']:.3f} in {v['tries_per_seed']} tries" for name, v in c.items()))
    for mode, b in results["balance"].items():
        print(f"- Balance, {mode}: penalty coupling is {b['penalty_to_objective_ratio']:.0f}x the largest objective term")
        for p, e in b["by_p"].items():
            print("    p=%s: " % p + "; ".join(f"{name}: balanced share {v['feasible_share']:.3f}, P(best balanced)/shot {v['p_optimum_per_shot']:.5f}"
                                             for name, v in e.items()))
        u = b["postselect_uniform_p0"]
        print(f"    p=0 control: balanced share {u['feasible_share']:.3f}, P(best balanced)/shot {u['p_optimum_per_shot']:.5f}")
    for key, extras in results["extras"].items():
        if key in ("deferred", "xy_mixer"):
            continue
        for name, e in extras.items():
            if e["status"] != "completed" or "summary" not in e:
                print(f"- Extra {name} ({key}): {e['status']}" + (f" ({e.get('error')})" if e["status"] == "failed" else ""))
                continue
            line = []
            for p, best in e["summary"]["best_by_p"].items():
                ref = e.get("standard_same_budget", e.get("standard_all_optimizers", {})).get("best_by_p", {}).get(p)
                line.append(f"p={p}: quality {best['metrics']['quality']:.3f}, P(opt)/shot {best['metrics']['p_optimum_per_shot']:.4f}"
                            + (f" (standard {ref['metrics']['quality']:.3f}, {ref['metrics']['p_optimum_per_shot']:.4f})" if ref else ""))
            print(f"- Extra {name} ({key}): " + "; ".join(line))
    for key in (key_of(mode, False) for mode in ising.MODES):
        v = results["extras"].get(key, {}).get("varqite_pauli")
        if not v:
            continue
        if v["status"] != "completed":
            print(f"- Extra varQITE Pauli-update ({key}): {v['status']} ({v.get('error')})")
            continue
        b = v["best"]
        print(f"- Extra varQITE Pauli-update ({key}): best ansatz {b['kind']} lr {b['lr']}, {b['steps_run']} steps, {b['evaluations']} evaluations: "
              f"energy {b['energy_per_step'][0]:+.5f} -> {b['best_energy']:+.5f}; quality {b['metrics']['quality']:.3f}, "
              f"P(opt)/shot {b['metrics']['p_optimum_per_shot']:.4f}, within 1 % {b['metrics']['p_within_1pct']:.4f}; "
              f"{v['circuit_metrics']['two_qubit_gates']} two-qubit gates; all runs: " +
              ", ".join(f"{r['kind']}/{r['lr']}: {r['metrics']['quality']:.3f}" for r in v["runs"]))
    xy = results["extras"].get("xy_mixer")
    if xy:
        if xy["status"] == "completed":
            for mode, e in xy["modes"].items():
                print(f"- Extra XY mixer ({mode}): " + "; ".join(
                    f"p={p}: balanced share {b['metrics']['feasible_share']:.3f}, P(best balanced)/shot {b['metrics']['p_optimum_per_shot']:.4f}"
                    for p, b in e["summary"]["best_by_p"].items()) + f"; NumPy vs Qiskit {e['numpy_vs_qiskit_max_difference']:.1e}")
        else:
            print(f"- Extra XY mixer: {xy['status']} ({xy.get('error')})")
    print(f"- Deferred: {results['extras']['deferred']}")
    for mode, c in results["circuits"]["n12"].items():
        one, r = c["by_p"][1] if 1 in c["by_p"] else c["by_p"]["1"], c["routing_p1"]
        print(f"- Circuit n=12, {mode}, per layer: {one['two_qubit_gates']} two-qubit gates, two-qubit depth {one['two_qubit_depth']}, "
              f"depth {one['depth']}; routed on heavy-hex ({r['heavy_hex_qubits']} qubits): {r['heavy_hex_two_qubit_gates']} two-qubit gates "
              f"against {r['all_to_all_two_qubit_gates']} all-to-all")
    for mode, c in results["circuits"]["large"].items():
        one, r = c["by_p"][1] if 1 in c["by_p"] else c["by_p"]["1"], c["routing_p1"]
        print(f"- Circuit n={c['n']}, {mode} (construction only, never simulated): {one['two_qubit_gates']} two-qubit gates per layer, "
              f"two-qubit depth {one['two_qubit_depth']}; heavy-hex {r['heavy_hex_two_qubit_gates']} against {r['all_to_all_two_qubit_gates']}; "
              f"angles: {c['angles_from']}")
    t = results["transfer"]
    if t.get("status") == "completed":
        for label in ("tuned_size", "larger_size_dense", "larger_size_thinned"):
            e = t[label]
            for mode, row in e["modes"].items():
                parts = [f"p={p}: quality {m['quality']:.3f}, P(opt)/shot {m['p_optimum_per_shot']:.5f}" for p, m in row["transferred"].items()]
                print(f"- Transfer to n={e['n']} ({label}, keep {e['keep_fraction']}), {mode}: uniform P(opt)/shot "
                      f"{row['uniform_p0']['p_optimum_per_shot']:.6f}; " + "; ".join(parts))
    else:
        print(f"- Transfer check: {t.get('status')} {t.get('error', '')}")
    ok = True
    if with_tests:
        passed, failed, skipped, log = run_tests()
        print(f"- Tests (python -m unittest discover -s tests -t .): {passed} passed, {failed} failed, {skipped} skipped")
        if failed:
            print(log)
        ok = failed == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    results = run_all()
    paths = save(results)
    print("saved:", ", ".join(str(p.relative_to(ROOT)) for p in paths))
    return print_report(json.loads(TUNING_PATH.read_text()))


if __name__ == "__main__":
    sys.exit(main())
