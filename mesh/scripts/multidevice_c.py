"""Option C - many devices with different speed, memory and power, end to end:
  A. device choice: one qubit per device of the synthetic CPU/GPU pool (Task 03b). QAOA picks which devices to switch on for
     each goal (fastest / overnight / lowest energy). The cost of every choice is the exact Task 03b model (memory-capped
     work shares, busy and idle power), so the cost layer is a general diagonal (its gate cost is reported) and the exact
     optimum is known. Parameters are calibrated on OTHER random pools (seeds 101-110) and frozen before the test pools.
  B. mesh split among the chosen devices: the mesh (case 2, 24 regions) is split by recursive two-way splits. Each split is the
     notebook's Ising model (build_partition_ising on the region subset) with the target share of its device group, so its
     exact optimum is known. QAOA (LR schedule calibrated on the 12-region calibration meshes, never on the test mesh) is
     compared with local search, simulated annealing, spectral bisection and target-share random sampling (1,000 evaluations
     each), every method recursing on its own split. The final multi-device split is scored with the Task 03b/04 model:
     time = slowest device + communication, electricity with busy and idle power, memory overload.
usage: python scripts/multidevice_c.py   ->  results/multidevice/*.json, results/multidevice/*.png
Noiseless simulation; modelled devices (assumptions, not measurements).
"""
import contextlib
import io
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "multidevice"
OUT.mkdir(parents=True, exist_ok=True)
POOL_CAL_SEEDS = list(range(101, 111))         # calibration pools (parameters are chosen here)
POOL_TEST_SEEDS = [7, 8, 9, 10, 11]            # test pools; 7 is the notebook's pool
GOALS = ["fastest", "overnight", "lowest energy"]
INFEASIBLE_LEVEL = 1.5                         # scaled cost of a device choice that cannot hold the job (feasible costs span 0..1)
A_LAYERS = [1, 2, 3, 4, 5]
SHOTS = 1000
MESH_CASE, MESH_SEED, MESH_REGIONS = "2 main", 7, 24
B_LAYERS = [1, 2, 3]
LR_GRID_GAMMA, LR_GRID_BETA = np.linspace(0.05, 6.0, 60), np.linspace(0.05, 1.5, 30)
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:6.0f} s] {message}", flush=True)


def load_namespace():
    """Notebook cells up to the end of the Task 04 calibration (plain exec: no checkpoint is touched)."""
    nb = nbformat.read(ROOT / "mesh_idea_ZollerQu.ipynb", as_version=4)
    ns = {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
            continue
        source, last = cell.source, cell.source.startswith("# TASK_04_IMPLEMENTATION")
        if last:
            source = source.split("# --------------------------------------------------------------- benchmark ----")[0]
        if cell.source.startswith("# TASK_03B_IMPLEMENTATION"):
            source = source.split("# ---- save the pool and the plans as data")[0]          # do not rewrite the saved JSON
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f"cell{index}", "exec"), ns)
        if last:
            return ns
    raise RuntimeError("Task 04 cell not found")


# ---------------------------------------------------------------- noiseless QAOA (standard: H start, X mixer) ----
def qaoa_probs(scaled, gammas, betas, n):
    psi = np.full(2**n, 2.0 ** (-n / 2), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi = psi * np.exp(-1j * gamma * scaled)
        c, s = np.cos(beta), 1j * np.sin(beta)
        for q in range(n):
            view = psi.reshape(-1, 2, 2**q)
            a0, a1 = view[:, 0, :].copy(), view[:, 1, :]
            view[:, 0, :] = c * a0 + s * a1
            view[:, 1, :] = s * a0 + c * a1
    return np.abs(psi) ** 2


def lr_angles(p, dg, db):
    k = np.arange(1, p + 1)
    return k / p * dg, (1 - (k - 1) / p) * db


def calibrate_lr(instances, layers):
    """Best (p, dgamma, dbeta) by mean P(optimum set) over calibration instances [(scaled, optimal_mask, n)]."""
    best = None
    for p in layers:
        for dg in LR_GRID_GAMMA:
            for db in LR_GRID_BETA:
                g, b = lr_angles(p, dg, db)
                score = float(np.mean([qaoa_probs(s, g, b, n)[mask].sum() for s, mask, n in instances]))
                if best is None or score > best["score"] + 1e-12:
                    best = dict(p=p, dgamma=float(dg), dbeta=float(db), score=score)
    return best


def walsh_terms(values, n):
    """Coefficients of the diagonal in the Z-string basis (fast Walsh-Hadamard), counted by weight (= qubits per term)."""
    coeff = np.asarray(values, dtype=float).copy()
    h = 1
    while h < len(coeff):
        for start in range(0, len(coeff), 2 * h):
            a, b = coeff[start:start + h].copy(), coeff[start + h:start + 2 * h].copy()
            coeff[start:start + h], coeff[start + h:start + 2 * h] = a + b, a - b
        h *= 2
    coeff /= len(coeff)
    weights = np.array([bin(k).count("1") for k in range(len(coeff))])
    return coeff, weights


# ================================================================ A. device choice ================================================================
def pool_problem(ns, seed):
    pool = ns["generate_hpc_pool"](seed)
    speed = np.array([d["speed"] for d in pool])
    cap = np.minimum(1.0, np.array([d["memory_gb"] for d in pool]) / ns["JOB_MEMORY_GB"])
    active, idle = np.array([d["active_w"] for d in pool]), np.array([d["idle_w"] for d in pool])
    n = len(pool)
    time_h, energy = np.full(2**n, np.inf), np.full(2**n, np.inf)
    shares = {}
    for mask in range(1, 2**n):
        chosen = np.flatnonzero((mask >> np.arange(n)) & 1)
        s = ns["fastest_shares"](speed[chosen], cap[chosen])
        if s is None:
            continue
        time_h[mask], energy[mask] = ns["plan_time_energy"](s, speed[chosen], active[chosen], idle[chosen])
        shares[mask] = s
    costs = {"fastest": time_h, "lowest energy": energy,
             "overnight": np.where(time_h <= ns["OVERNIGHT_HOURS"] + 1e-9, energy, np.inf)}
    problems = {}
    for goal, cost in costs.items():
        feasible = np.isfinite(cost)
        if not feasible.any():
            continue
        lo, hi = cost[feasible].min(), cost[feasible].max()
        scaled = np.where(feasible, (cost - lo) / max(hi - lo, 1e-12), INFEASIBLE_LEVEL)
        optimal = feasible & (cost <= lo + 1e-9)
        problems[goal] = dict(scaled=scaled, optimal=optimal, n=n, feasible=int(feasible.sum()), best=int(np.flatnonzero(optimal)[0]),
                              best_time=float(time_h[np.flatnonzero(optimal)[0]]), best_energy=float(energy[np.flatnonzero(optimal)[0]]))
    return dict(seed=seed, pool=pool, n=n, speed=speed, cap=cap, active=active, idle=idle, shares=shares, problems=problems)


def stage_a(ns):
    from scipy.optimize import minimize
    calibration = [pool_problem(ns, s) for s in POOL_CAL_SEEDS]
    tests = [pool_problem(ns, s) for s in POOL_TEST_SEEDS]
    log(f"A: {len(calibration)} calibration pools ({min(c['n'] for c in calibration)}-{max(c['n'] for c in calibration)} devices), "
        f"{len(tests)} test pools ({[t['n'] for t in tests]} devices)")
    out = dict(goals={}, test_pools=[dict(seed=t["seed"], devices=[d["id"] for d in t["pool"]]) for t in tests])
    for goal in GOALS:
        cal = [(c["problems"][goal]["scaled"], c["problems"][goal]["optimal"], c["n"]) for c in calibration if goal in c["problems"]]
        rows = []
        for p in A_LAYERS:
            lr = calibrate_lr(cal, [p])
            def objective(x, p=p):
                return -float(np.mean([qaoa_probs(s, x[:p], x[p:], n)[m].sum() for s, m, n in cal]))
            start = np.concatenate(lr_angles(p, lr["dgamma"], lr["dbeta"]))
            opt = minimize(objective, start, method="Nelder-Mead", options=dict(maxfev=400 * p, xatol=1e-4, fatol=1e-6))
            optimised = dict(gammas=opt.x[:p].tolist(), betas=opt.x[p:].tolist(), score=-float(opt.fun), evaluations=int(opt.nfev))
            for label, gammas, betas, cal_score, cost in (
                    ("LR-QAOA", *lr_angles(p, lr["dgamma"], lr["dbeta"]), lr["score"], len(LR_GRID_GAMMA) * len(LR_GRID_BETA)),
                    ("QAOA, angles optimised on calibration pools", optimised["gammas"], optimised["betas"], optimised["score"], optimised["evaluations"])):
                per_pool = []
                for t in tests:
                    if goal not in t["problems"]:
                        continue
                    prob = t["problems"][goal]
                    probs = qaoa_probs(prob["scaled"], gammas, betas, t["n"])
                    p_opt = float(probs[prob["optimal"]].sum())
                    per_pool.append(dict(seed=t["seed"], n=t["n"], p_opt=p_opt, hit_1000=1 - (1 - p_opt) ** SHOTS,
                                         random_p_opt=float(prob["optimal"].sum() / 2 ** t["n"]),
                                         feasible_random_p_opt=float(prob["optimal"].sum() / prob["feasible"]),
                                         most_likely_is_optimal=bool(prob["optimal"][int(np.argmax(probs))])))
                rows.append(dict(method=label, p=p, gammas=[float(g) for g in gammas], betas=[float(b) for b in betas],
                                 calibration_mean_p_opt=cal_score, calibration_evaluations=cost, test=per_pool,
                                 test_mean_p_opt=float(np.mean([r["p_opt"] for r in per_pool])),
                                 test_mean_random=float(np.mean([r["random_p_opt"] for r in per_pool])),
                                 test_most_likely_optimal=int(sum(r["most_likely_is_optimal"] for r in per_pool)), test_pools=len(per_pool)))
            log(f"A {goal}, p = {p}: LR calibration {lr['score']:.3f}, optimised {optimised['score']:.3f}; "
                f"test mean P(opt) LR {rows[-2]['test_mean_p_opt']:.3f}, optimised {rows[-1]['test_mean_p_opt']:.3f} "
                f"(random {rows[-1]['test_mean_random']:.4f})")
        chosen = max(rows, key=lambda r: r["calibration_mean_p_opt"])           # chosen on calibration pools only
        main = next(t for t in tests if t["seed"] == 7)
        prob = main["problems"][goal]
        coeff, weights = walsh_terms(prob["scaled"], main["n"])
        nonzero = np.abs(coeff) > 1e-12
        cnots = int(sum(2 * (w - 1) for w, nz in zip(weights, nonzero) if nz and w >= 2))
        quadratic = np.where(weights <= 2, coeff, 0.0)
        approx = np.array([sum(quadratic[t] * (-1) ** bin(t & k).count("1") for t in np.flatnonzero(quadratic)) for k in range(2 ** main["n"])])
        out["goals"][goal] = dict(
            rows=rows, chosen=dict(method=chosen["method"], p=chosen["p"]),
            pool7=dict(best_devices=[main["pool"][i]["id"] for i in range(main["n"]) if (prob["best"] >> i) & 1],
                       best_time_h=prob["best_time"], best_energy_kwh=prob["best_energy"], feasible_choices=prob["feasible"],
                       diagonal_terms_by_weight={int(w): int(np.sum(nonzero & (weights == w))) for w in range(1, main["n"] + 1)},
                       cnot_per_cost_layer_general_diagonal=cnots,
                       quadratic_approximation_keeps_optimum=bool(prob["optimal"][int(np.argmin(approx))]),
                       quadratic_approximation_rank_of_true_optimum=int(np.sum(approx < approx[prob["best"]] - 1e-12)) + 1))
    return out, {t["seed"]: t for t in tests}


# ================================================================ B. recursive mesh split ================================================================
def device_groups(shares):
    """Two device groups with the most equal total share (exhaustive; group 1 holds device 0)."""
    k = len(shares)
    best = None
    for mask in range(0, 2 ** (k - 1)):
        group_b = [i for i in range(1, k) if (mask >> (i - 1)) & 1]
        if not group_b:
            continue
        group_a = [i for i in range(k) if i not in group_b]
        gap = abs(shares[group_a].sum() - shares[group_b].sum())
        if best is None or gap < best[0] - 1e-12:
            best = (gap, group_a, group_b)
    return best[1], best[2]


def split_problem(ns, regions, subset, alpha):
    """The notebook's Ising model on a region subset; lambda = smallest grid value whose exact optimum is balanced."""
    lam_grid, eps = ns["LAMBDA_GRID"], ns["ISING_EPSILON"]
    base = ns["build_partition_ising"](regions["region_work"], regions["region_pairs"], regions["region_boundary_cost"], alpha, lam=1.0, regions=subset)
    cut, share_b, _ = ns["enumerate_components"](base)
    balanced = np.abs(share_b - alpha) <= eps
    lam = None
    for value in lam_grid:
        energies = cut + value * 4.0 * (alpha - share_b) ** 2
        if balanced[int(np.argmin(energies))]:
            lam = float(value)
            break
    lam = float(lam_grid[-1]) if lam is None else lam
    model = ns["build_partition_ising"](regions["region_work"], regions["region_pairs"], regions["region_boundary_cost"], alpha, lam=lam, regions=subset)
    energies = cut + lam * 4.0 * (alpha - share_b) ** 2
    n = len(subset)
    check = np.random.default_rng(n).integers(0, 2**n, 30)
    for k in check:
        z = 1 - 2 * ((k >> np.arange(n)) & 1)
        assert abs(energies[k] - ns["ising_energy"](z, model["J"], model["h"], model["const"])) < 1e-9, "enumeration differs from the Ising energy"
    e_min = energies.min()
    return dict(model=model, n=n, lam=lam, alpha=alpha, energies=energies, e_min=float(e_min), optimal=energies <= e_min + 1e-9,
                near=energies <= 1.05 * e_min + 1e-9, share_b=share_b, scaled=(energies - model["const"]) / model["scale"])


def calibration_set(ns, alpha):
    """The 12-region calibration meshes (seeds 101 / 102, never scored) with the same target share."""
    instances = []
    for regions in ns["calibration_regions"].values():
        prob = split_problem(ns, regions, np.arange(len(regions["region_work"])), alpha)
        instances.append((prob["scaled"], prob["near"], prob["n"]))
    return instances


def classical_split(ns, prob, subset, regions, method, seed):
    """Best of 1,000 evaluations of a Task 04 classical method on this split (as a state index over the subset)."""
    inst = dict(n=prob["n"], model=prob["model"], energies=prob["energies"], e_std_formula=float(np.sqrt(np.sum(prob["model"]["J"] ** 2) + np.sum(prob["model"]["h"] ** 2))),
                scenario="2 equal devices" if abs(prob["alpha"] - 0.5) < 1e-9 else "CPU + GPU (GPU 4x)", alpha=prob["alpha"],
                regions=dict(region_pairs=prob["model"]["pairs"], region_boundary_cost=prob["model"]["c_tilde"]))
    rng = np.random.default_rng(seed)
    if method == "local search":
        return ns["local_search"](inst, rng, SHOTS)[1]
    if method == "simulated annealing":
        return ns["simulated_annealing"](inst, rng, SHOTS)[1]
    if method == "spectral bisection":
        return ns["spectral_split"](inst)[1]
    candidates = (rng.random((SHOTS, prob["n"])) < prob["alpha"]).astype(np.int64) @ (1 << np.arange(prob["n"], dtype=np.int64))
    return int(candidates[np.argmin(prob["energies"][candidates])])


def recursive_split(ns, regions, subset, devices, shares, method, calibration_cache, log_rows, depth=0, seed=0):
    """Assign the regions in `subset` to `devices` (target shares `shares`, summing to the subset's share)."""
    if len(devices) == 1:
        return {int(r): devices[0] for r in subset}
    if len(subset) == 1:                                   # one region left for several devices: it cannot be split further
        return {int(subset[0]): devices[int(np.argmax(shares))]}
    group_a, group_b = device_groups(shares)
    alpha = float(shares[group_b].sum() / shares.sum())
    prob = split_problem(ns, regions, subset, alpha)
    if method == "QAOA":
        key = round(alpha, 4)
        if key not in calibration_cache:
            calibration_cache[key] = calibrate_lr(calibration_set(ns, alpha), B_LAYERS)
        cal = calibration_cache[key]
        gammas, betas = lr_angles(cal["p"], cal["dgamma"], cal["dbeta"])
        probs = qaoa_probs(prob["scaled"], gammas, betas, prob["n"])
        shots = np.random.default_rng([seed, depth, prob["n"]]).choice(2 ** prob["n"], size=SHOTS, p=probs / probs.sum())
        state = int(shots[np.argmin(prob["energies"][shots])])
        extra = dict(p_opt=float(probs[prob["optimal"]].sum()), p_near=float(probs[prob["near"]].sum()), p=cal["p"],
                     dgamma=cal["dgamma"], dbeta=cal["dbeta"], calibration_p_near=cal["score"], two_qubit_gates=cal["p"] * prob["n"] * (prob["n"] - 1) // 2)
    elif method == "exact":
        state, extra = int(np.flatnonzero(prob["optimal"])[0]), {}
    else:
        state, extra = classical_split(ns, prob, subset, regions, method, [seed, depth, prob["n"]]), {}
    log_rows.append(dict(method=method, depth=depth, regions=int(prob["n"]), devices=[int(d) for d in devices], alpha=alpha, lam=prob["lam"],
                         gap_percent=float(100 * (prob["energies"][state] / prob["e_min"] - 1)), optimal=bool(prob["optimal"][state]),
                         random_p_opt=float(prob["optimal"].mean()), **extra))
    bits = (state >> np.arange(prob["n"])) & 1
    part_a, part_b = subset[bits == 0], subset[bits == 1]
    result = {}
    for part, group in ((part_a, group_a), (part_b, group_b)):
        if len(part) == 0:
            continue
        group_devices = [devices[i] for i in group]
        result.update(recursive_split(ns, regions, part, group_devices, shares[group], method, calibration_cache, log_rows, depth + 1, seed))
    return result


def score_assignment(ns, regions, assignment, pool, devices, targets):
    work = np.asarray(regions["region_work"], dtype=float)
    pairs, cost = np.asarray(regions["region_pairs"]), np.asarray(regions["region_boundary_cost"], dtype=float)
    owner = np.array([assignment[r] for r in range(len(work))])
    share = np.array([work[owner == d].sum() / work.sum() for d in devices])
    cut_fraction = float(cost[owner[pairs[:, 0]] != owner[pairs[:, 1]]].sum() / cost.sum())
    speed, cap = pool["speed"][devices], pool["cap"][devices]
    busy = ns["JOB_REFERENCE_HOURS"] * share / speed
    comm = ns["TASK04"]["comm_fraction_if_all_borders_cut"] * ns["JOB_REFERENCE_HOURS"] * cut_fraction
    total = busy.max() + comm
    energy = float(np.sum(pool["active"][devices] * busy + pool["idle"][devices] * (total - busy)) / 1000.0)
    return dict(shares=share.tolist(), targets=targets.tolist(), max_share_error=float(np.abs(share - targets).max()),
                memory_overload=float(np.maximum(0, share - cap).sum()), cut_fraction=cut_fraction, compute_hours=float(busy.max()),
                comm_hours=float(comm), modelled_hours=float(total), modelled_kwh=energy, unused_devices=int(np.sum(share == 0)))


def stage_b(ns, pools):
    pool = pools[7]
    mesh = ns["mesh_cases"][MESH_CASE]
    regions = ns["coarsen_mesh"](mesh, MESH_REGIONS, MESH_SEED)
    ns["check_region_case"]((MESH_CASE, MESH_REGIONS, MESH_SEED), regions, MESH_REGIONS)
    out, cache = {}, {}
    for goal in GOALS:
        prob = pool["problems"][goal]
        devices = [i for i in range(pool["n"]) if (prob["best"] >> i) & 1]
        targets = pool["shares"][prob["best"]]
        order = np.argsort(-targets, kind="stable")                       # largest share first
        devices, targets = [devices[i] for i in order], targets[order]
        plan_busy = ns["JOB_REFERENCE_HOURS"] * targets / pool["speed"][devices]
        out[goal] = dict(devices=[pool["pool"][d]["id"] for d in devices], target_shares=targets.tolist(),
                         plan_hours=float(plan_busy.max()), plan_kwh=prob["best_energy"], methods={})
        for method in ("exact", "QAOA", "local search", "simulated annealing", "spectral bisection", "random, target share"):
            rows = []
            assignment = recursive_split(ns, regions, np.arange(MESH_REGIONS), devices, targets, method, cache, rows, seed=11)
            out[goal]["methods"][method] = dict(score_assignment(ns, regions, assignment, pool, devices, targets), splits=rows,
                                                splits_optimal=int(sum(r["optimal"] for r in rows)), n_splits=len(rows),
                                                assignment=[int(assignment[r]) for r in range(MESH_REGIONS)])
            m = out[goal]["methods"][method]
            log(f"B {goal} ({len(devices)} devices), {method}: {m['splits_optimal']}/{m['n_splits']} splits optimal, share error "
                f"{m['max_share_error']:.3f}, cut {m['cut_fraction']:.3f}, {m['modelled_hours']:.2f} h, {m['modelled_kwh']:.2f} kWh")
    out["mesh"] = dict(case=MESH_CASE, seed=MESH_SEED, regions=MESH_REGIONS, region_work=[float(w) for w in regions["region_work"]])
    out["qaoa_calibration"] = {str(k): v for k, v in cache.items()}
    return out


if __name__ == "__main__":
    ns = load_namespace()
    log("notebook loaded (cells up to the Task 04 calibration)")
    if (OUT / "stage_a_device_choice.json").exists():               # stage A is already saved: rebuild only the test pools
        pools = {seed: pool_problem(ns, seed) for seed in POOL_TEST_SEEDS}
        log("stage A results found; rebuilding the test pools only")
    else:
        a_results, pools = stage_a(ns)
        (OUT / "stage_a_device_choice.json").write_text(json.dumps(a_results, indent=1))
    b_results = stage_b(ns, pools)
    (OUT / "stage_b_recursive_split.json").write_text(json.dumps(b_results, indent=1))
    log(f"saved {OUT}")
