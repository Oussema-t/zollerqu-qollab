"""Pick the circuits for the single Qollab/IonQ batch from the EXISTING 12-qubit results (no training, no tuning).

For every candidate circuit: noiseless P(optimum) per shot and share of shots within 5 % of the optimum; two-qubit
gate count and depth after transpiling to an IonQ-like basis (rzz / cx count as one two-qubit gate each); a crude
noisy estimate = noiseless score x 0.995^(two-qubit gates); samples-to-solution = 1 / P(optimum).
Classical reference: median number of candidate evaluations until greedy descent / simulated annealing first hit the
optimum. Writes qollab/circuit_candidates.json (all rows) and qollab/selected_circuits.json (the choice).
Run from the repo root:  python -m stocks.select_circuits
"""
import json
import sys

import numpy as np
from qiskit import transpile

from stocks import classical_solvers as solvers
from stocks import fair_budget, ising, kaggle_data, qaoa, qaoa_sim, task3_report as t3, task4_report as t4, varqite
from stocks import qaoa_tuning as tuning

ROOT = kaggle_data.DATA_DIR.parent
OUT_DIR = ROOT / "qollab"
CANDIDATES_PATH = OUT_DIR / "circuit_candidates.json"
SELECTED_PATH = OUT_DIR / "selected_circuits.json"
BASIS = ["rzz", "cx", "rz", "rx", "ry", "h", "sx", "x"]   # rzz native on IonQ Forte; cx = one MS/ZZ gate + single-qubit gates
TWO_QUBIT_ERROR = 0.995                                      # crude: survival factor per two-qubit gate
MAX_TWO_QUBIT_GATES = 200
BATCH_SIZE = 5
# Hand-picked batch (all under the gate cap): the control, the best circuit overall, the best of mode A, and the main
# method (LR-QAOA) at two depths in the mode where its hits are countable in 1,000 shots. The automatic rule is the fallback.
PREFERRED_BATCH = ["control p=0 (H only)", "B CVaR 0.1 p=3", "A counterdiabatic p=3", "B LR-QAOA p=1", "B LR-QAOA p=3"]
WITHIN = 0.05
ETA = 1.0
SEEDS_TO_SOLUTION = 50


def transpiled_metrics(circuit):
    compiled = transpile(circuit, basis_gates=BASIS, optimization_level=1, seed_transpiler=2026)
    ops = {k: int(v) for k, v in compiled.count_ops().items()}
    two_qubit = int(sum(1 for inst in compiled.data if inst.operation.num_qubits == 2))
    return {"two_qubit_gates": two_qubit, "depth": int(compiled.depth(lambda inst: inst.operation.name != "measure")),
            "two_qubit_depth": int(compiled.depth(lambda inst: inst.operation.num_qubits == 2)), "ops": ops,
            "generic_state_preparation": any(name.lower().startswith(("state_preparation", "initialize", "disentangler")) for name in ops)
            or any(inst.operation.name.lower().startswith(("state_preparation", "initialize")) for inst in circuit.data)}


def scores(probabilities, objective, feasible, e_min, e_max, tolerance):
    """P(optimum) per shot and P(within 5 % of the optimum) per shot, on the balanced states when feasible is a mask."""
    p = np.asarray(probabilities, dtype=float)
    optimal = feasible & (objective <= e_min + tolerance)
    near = feasible & (objective <= e_min + WITHIN * (e_max - e_min))
    return float(p[optimal].sum()), float(p[near].sum())


def evaluations_to_solution(model, e_min, tolerance, seeds=SEEDS_TO_SOLUTION, cap=50_000):
    """Median candidate evaluations until the first hit of the optimum (greedy with restarts; annealing proposals)."""
    n, couplings, fields = model.n, model.couplings, model.fields
    greedy, annealing = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(10_000 + seed)
        used, hit = 0, False
        while used < cap and not hit:
            s = (1 - 2 * rng.integers(0, 2, size=n)).astype(float)
            energy = float(model.energy(s))
            used += 1
            hit = energy <= e_min + tolerance
            while not hit:
                gains = -2.0 * s * (couplings @ s + fields)
                used += n
                k = int(np.argmin(gains))
                if gains[k] >= 0:
                    break
                s[k] = -s[k]
                energy += gains[k]
                hit = energy <= e_min + tolerance
        greedy.append(used if hit else np.inf)
        rng = np.random.default_rng(20_000 + seed)
        s = (1 - 2 * rng.integers(0, 2, size=n)).astype(float)
        local = couplings @ s + fields
        energy = float(model.energy(s))
        t_start = 2.0 * float(np.max(np.abs(couplings).sum(axis=1) + np.abs(fields)))
        temperatures = np.geomspace(t_start, t_start * 1e-3, cap)
        found = np.inf
        for step, temperature in enumerate(temperatures, start=1):
            if energy <= e_min + tolerance:
                found = step
                break
            k = int(rng.integers(n))
            delta = -2.0 * s[k] * local[k]
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                s[k] = -s[k]
                local += 2.0 * s[k] * couplings[k]
                energy += delta
        annealing.append(found)
    return {"greedy_median": float(np.median(greedy)), "annealing_median": float(np.median(annealing)),
            "greedy_unsolved": int(sum(np.isinf(greedy))), "annealing_unsolved": int(sum(np.isinf(annealing))), "runs": seeds}


def ising_terms(model):
    return {"pair_terms": [[i, j, float(c)] for i, j, c in model.pair_terms()], "fields": [float(h) for h in model.fields],
            "const": float(model.const), "scale": qaoa_sim.cost_scale(model)}


def candidates():
    results, parameters = t4.load(), json.loads(t4.PARAMETERS_PATH.read_text())
    ctx = t3.load_context()
    setup = t3.size_setup(ctx, 12)
    rows, circuits, built = [], {}, {}

    def add(name, mode, family, circuit, probabilities, c, model, angles, extra=None):
        p_opt, p_near = scores(probabilities, c.objective, c.feasible, c.e_min, c.e_max, c.tolerance)
        metrics = transpiled_metrics(circuit)
        survival = TWO_QUBIT_ERROR ** metrics["two_qubit_gates"]
        row = {"name": name, "mode": mode, "family": family, "balance": bool(c.balance),
               "p_optimum_noiseless": p_opt, "p_within_5pct_noiseless": p_near,
               "two_qubit_gates": metrics["two_qubit_gates"], "depth": metrics["depth"], "two_qubit_depth": metrics["two_qubit_depth"],
               "ops": metrics["ops"], "generic_state_preparation": metrics["generic_state_preparation"],
               "crude_noisy_p_optimum": p_opt * survival, "crude_noisy_p_within_5pct": p_near * survival,
               "shots_to_solution_noiseless": (1.0 / p_opt) if p_opt > 0 else None,
               "shots_to_solution_crude_noisy": (1.0 / (p_opt * survival)) if p_opt > 0 else None,
               "angles": angles, **(extra or {})}
        rows.append(row)
        circuits[name] = {"model": ising_terms(model), **({} if extra is None else extra)}
        built[name] = (circuit, c)
        return row

    for mode in ising.MODES:
        key, key_bal = t4.key_of(mode, False), t4.key_of(mode, True)
        c = t4.Context(setup, mode, False, ETA)
        c_bal = t4.Context(setup, mode, True, ETA)
        model, sim = c.model, c.sim
        tag = "A" if mode == ising.MODE_LONG_SHORT else "B"
        if mode == ising.MODE_LONG_SHORT:
            add("control p=0 (H only)", "both", "control", qaoa.build_circuit(model, [], []), sim.probabilities([], []), c, model, {"gammas": [], "betas": []})
        lr = parameters["modes"][mode]["linear_ramp"]
        for p, e in lr.items():
            add(f"{tag} LR-QAOA p={p}", mode, "LR-QAOA", qaoa.build_circuit(model, e["gammas"], e["betas"]), sim.probabilities(e["gammas"], e["betas"]), c, model,
                {"gammas": e["gammas"], "betas": e["betas"]}, {"dgamma": e["best_dgamma"], "dbeta": e["best_dbeta"]})
        for p, r in results["models"][key]["standard"]["best_by_p"].items():
            a = r["angles"]
            add(f"{tag} free-angle QAOA p={p}", mode, "standard", qaoa.build_circuit(model, a["gammas"], a["betas"]), tuning.probabilities_for(sim, a), c, model, a,
                {"optimizer": r["optimizer"], "start": r["start"]})
        for extra_name, family in (("cvar_0.1", "CVaR 0.1"), ("cvar_0.25", "CVaR 0.25"), ("multi_angle", "multi-angle"), ("dc", "counterdiabatic"), ("varqite", "VarQITE angle training")):
            e = results["extras"][key].get(extra_name, {})
            if e.get("status") != "completed":
                continue
            for p, r in e["summary"]["best_by_p"].items():
                a = r["angles"]
                circuit = qaoa.build_circuit(model, a["gammas"], a["betas"], field_gammas=a.get("field_gammas"), alphas=a.get("alphas"))
                add(f"{tag} {family} p={p}", mode, family, circuit, tuning.probabilities_for(sim, a), c, model, a, {"optimizer": r["optimizer"], "start": r["start"]})
        v = results["extras"][key]["varqite_pauli"]
        for run in v["runs"]:
            if run["lr"] != max(x["lr"] for x in v["runs"] if x["kind"] == run["kind"]):
                continue
            ansatz = varqite.Ansatz(model, run["kind"])
            theta = np.array(run["theta"])
            add(f"{tag} varQITE Pauli-update ansatz {run['kind']} (lr {run['lr']})", mode, "varQITE", ansatz.to_qiskit(theta), ansatz.probabilities(theta), c, model,
                {"theta": run["theta"]}, {"ansatz": run["kind"], "lr": run["lr"]})
        xy = results["extras"]["xy_mixer"]
        if xy.get("status") == "completed":
            ones = (model.n - c_bal.model.target_sum) // 2
            xsim = t4.XYSimulator(model, ones)
            for p, r in xy["modes"][mode]["summary"]["best_by_p"].items():
                a = r["angles"]
                circuit = t4.build_xy_circuit(model, a["gammas"], a["betas"], xsim.pairs, xsim.start)
                circuit.measure_all()
                add(f"{tag} XY mixer p={p} (balance on, Dicke start)", mode, "XY mixer", circuit, xsim.probabilities(a["gammas"], a["betas"]), c_bal, model, a,
                    {"xy_pairs": xsim.pairs, "dicke_ones": int(ones), "note": "generic Dicke state preparation; objective = best balanced portfolio"})
        lr_bal = results["models"][key_bal]["lr"]
        for p in ("1", "2"):
            e = lr_bal[p]
            add(f"{tag} LR-QAOA p={p} on the penalty model (balance on)", mode, "LR-QAOA penalty", qaoa.build_circuit(c_bal.model, e["gammas"], e["betas"]),
                c_bal.sim.probabilities(e["gammas"], e["betas"]), c_bal, c_bal.model, {"gammas": e["gammas"], "betas": e["betas"]},
                {"penalty_lambda": float(c_bal.model.penalty_lambda), "note": "objective = best balanced portfolio"})
    classical = {mode: evaluations_to_solution(t4.Context(setup, mode, False, ETA).model, t4.Context(setup, mode, False, ETA).e_min, t4.Context(setup, mode, False, ETA).tolerance)
                 for mode in ising.MODES}
    return rows, circuits, classical, setup, built


def choose(rows):
    """PREFERRED_BATCH when every name exists and is under the gate cap; otherwise p=0 control + the best by crude
    noisy P(optimum) under the cap, both modes represented, at most BATCH_SIZE. Fallback = best eligible circuit."""
    control = next(r for r in rows if r["family"] == "control")
    eligible = [r for r in rows if r["family"] != "control" and r["two_qubit_gates"] <= MAX_TWO_QUBIT_GATES and not r["generic_state_preparation"]]
    ranked = sorted(eligible, key=lambda r: -r["crude_noisy_p_optimum"])
    by_name = {r["name"]: r for r in rows}
    if all(name in by_name and by_name[name]["two_qubit_gates"] <= MAX_TWO_QUBIT_GATES for name in PREFERRED_BATCH):
        return [by_name[name] for name in PREFERRED_BATCH], ranked[0]
    chosen, families = [control], set()
    for mode in ising.MODES:   # the best of each mode first
        best = next(r for r in ranked if r["mode"] == mode)
        chosen.append(best)
        families.add((best["mode"], best["family"]))
    for r in ranked:           # then the next best of a different (mode, family)
        if len(chosen) >= BATCH_SIZE:
            break
        if r not in chosen and (r["mode"], r["family"]) not in families:
            chosen.append(r)
            families.add((r["mode"], r["family"]))
    fallback = ranked[0]
    return chosen, fallback


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows, circuits, classical, setup, _ = candidates()
    chosen, fallback = choose(rows)
    CANDIDATES_PATH.write_text(json.dumps({"basis_gates": BASIS, "two_qubit_error_assumed": TWO_QUBIT_ERROR, "within": WITHIN, "eta": ETA,
                                           "classical_evaluations_to_solution": classical, "rows": rows}, indent=1, default=float) + "\n")
    selected = {
        "note": "chosen from existing 12-qubit results only; no training or tuning; one backend.run with all circuits; shots per circuit 1000",
        "n": 12, "eta": ETA, "qubit_order": {"qubit i": "asset tickers[i]", "tickers": setup["tickers"]},
        "bit_convention": "spin +1 (long / buy) <-> |0>; in a count key qubit 0 is the RIGHTMOST character",
        "circuit_recipe": "H on all qubits; per layer: rzz(2*gamma*J_ij/scale) per pair term, rz(2*gamma*h_i/scale) per field, rx(2*beta) per qubit "
                          "(field_gammas replace gamma for the rz terms; alphas add ry(2*alpha) after the mixer); measure_all",
        "batch": [{"name": r["name"], "mode": r["mode"], "family": r["family"], "balance": r["balance"], "angles": r["angles"],
                   "two_qubit_gates": r["two_qubit_gates"], "p_optimum_noiseless": r["p_optimum_noiseless"],
                   "crude_noisy_p_optimum": r["crude_noisy_p_optimum"], **circuits[r["name"]]} for r in chosen],
        "fallback": {"name": fallback["name"], "mode": fallback["mode"], "family": fallback["family"], "angles": fallback["angles"],
                     "two_qubit_gates": fallback["two_qubit_gates"], **circuits[fallback["name"]]},
    }
    SELECTED_PATH.write_text(json.dumps(selected, indent=1, default=float) + "\n")
    print(f"{'circuit':<58} {'P(opt)':>8} {'<5%':>7} {'2q':>4} {'depth':>5} {'crude':>8} {'shots/sol':>10}")
    for r in sorted(rows, key=lambda r: (r["mode"] != "both", r["mode"], -r["crude_noisy_p_optimum"])):
        flag = " [generic prep]" if r["generic_state_preparation"] else ""
        sts = "inf" if r["shots_to_solution_crude_noisy"] is None else f"{r['shots_to_solution_crude_noisy']:.0f}"
        print(f"{r['name'] + flag:<58} {r['p_optimum_noiseless']:8.4f} {r['p_within_5pct_noiseless']:7.3f} {r['two_qubit_gates']:4d} {r['depth']:5d} "
              f"{r['crude_noisy_p_optimum']:8.4f} {sts:>10}")
    print("classical evaluations to first hit of the optimum (median):", classical)
    print("chosen batch:", [r["name"] for r in chosen], "| fallback:", fallback["name"])
    print("saved:", CANDIDATES_PATH.relative_to(ROOT), SELECTED_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
