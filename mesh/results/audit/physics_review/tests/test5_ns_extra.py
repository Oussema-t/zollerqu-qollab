"""Extra physics checks on the real instances: adiabatic direction, relaxation degeneracy, periodicity, normalisation, recovery stationarity."""
import numpy as np
from nsload import load_defs
ns = load_defs("small")
T = ns["TASK04"]
rng = np.random.default_rng(5)
n0, seed0 = ns["RUN"]["region_counts"][0], ns["RUN"]["seeds"][0]
insts = {(case, scen): ns["t4_build_instance"](ns["region_cases"][(case, n0, seed0)], case, n0, seed0, scen)
         for case in ns["BENCHMARK_CASES"] for scen in ns["TARGET_SHARES"]}
cal = {"2 equal devices": (0.26, 0.2), "CPU + GPU (GPU 4x)": (0.96, 0.55)}      # author's calibrated values (summary json)

print("1. many-layer LR-QAOA on real instances (direction of the implemented evolution)")
for (case, scen), inst in insts.items():
    if case not in ("1 control", "2 main", "6 three sources"):
        continue
    view = ns["t4_view"](inst); E = inst["energies"]
    top = np.flatnonzero(E >= E.max() - 1e-9)
    line = []
    for p in (1, 2, 5, 20, 100, 400):
        ang = ns["lr_angles"](p, 0.3, 0.3)
        pr = ns["view_probabilities"](view, ang)
        prw = ns["view_probabilities"](view, np.concatenate([ang[:p], -ang[p:]]))
        line.append(f"p={p}: P(opt)={pr[inst['optimal_states']].sum():.4f} <E>={pr @ E:.3f} | flipped beta: P(max)={prw[top].sum():.4f} <E>={prw @ E:.3f}")
    print(f"   {case} / {scen}: E_min={inst['e_min']:.4f} E_mean={inst['e_mean']:.4f} E_max={inst['e_max']:.3f}\n      " + "\n      ".join(line))

print("2. relaxation warm start degeneracy")
for (case, scen), inst in insts.items():
    if case not in ("1 control", "2 main"):
        continue
    n = inst["n"]
    _, spectral_state, _ = ns["spectral_split"](inst)
    thetas, axis, c, note = ns["warm_start_setup"](inst, "relaxation, mixer A", spectral_state)
    wview, view = ns["t4_view"](inst, thetas, axis), ns["t4_view"](inst)
    worst = 0
    for _ in range(20):
        p = int(rng.integers(1, 4)); ang = np.concatenate([rng.uniform(0, 1, p), rng.uniform(0, np.pi / 2, p)])
        worst = max(worst, np.abs(ns["view_probabilities"](wview, ang) - ns["view_probabilities"](view, ang)).max())
    start = ns["view_probabilities"](wview, np.array([]))
    share = np.ones(1)
    for _ in range(n):
        share = np.kron(share, np.array([1 - inst["alpha"], inst["alpha"]]))
    print(f"   {case} / {scen}: c = {np.unique(np.round(c, 6)).tolist()}; max |P_ws(relax, mixer A) - P_standard QAOA| over 20 random angle sets = {worst:.2e}; "
          f"max |P_start - P('random with target share')| = {np.abs(start - share).max():.2e}; max |P_start - uniform| = {np.abs(start - 2.0 ** -n).max():.2e}")

print("3. beta periodicity (pi/2) for the Z2-symmetric scenario, and what the 'success-calibrated' schedule is")
inst = insts[("2 main", "2 equal devices")]; view = ns["t4_view"](inst)
for p in (1, 2):
    ang = np.concatenate([rng.uniform(0, 2, p), rng.uniform(0, 1.5, p)])
    shifted = ang.copy(); shifted[p:] -= np.pi / 2
    print(f"   equal, p={p}: max |P(gamma, beta) - P(gamma, beta - pi/2)| = {np.abs(ns['view_probabilities'](view, ang) - ns['view_probabilities'](view, shifted)).max():.2e}")
inst_c = insts[("2 main", "CPU + GPU (GPU 4x)")]; view_c = ns["t4_view"](inst_c)
ang = np.array([0.7, 1.1]); shifted = np.array([0.7, 1.1 - np.pi / 2])
print(f"   cpu_gpu (fields), p=1: max |P(beta) - P(beta - pi/2)| = {np.abs(ns['view_probabilities'](view_c, ang) - ns['view_probabilities'](view_c, shifted)).max():.2e} (not periodic, as expected); "
      f"beta period pi: {np.abs(ns['view_probabilities'](view_c, ang) - ns['view_probabilities'](view_c, np.array([0.7, 1.1 - np.pi]))).max():.2e}")
for p in (1, 2):
    a = ns["lr_angles"](p, 1.74, 1.45)
    print(f"   equal 'success-calibrated' (1.74, 1.45), p={p}: betas {np.round(a[p:], 3).tolist()} are equivalent to {np.round((a[p:] + np.pi / 4) % (np.pi / 2) - np.pi / 4, 3).tolist()} (mod pi/2); gammas {np.round(a[:p], 3).tolist()}")

print("4. normalisation: max|J| vs max|h| and the 'paper default' schedule")
for scen in ns["TARGET_SHARES"]:
    rows = []
    for case in ns["BENCHMARK_CASES"]:
        inst = insts[(case, scen)]; m = inst["model"]; view = ns["t4_view"](inst)
        jmax, hmax = np.abs(m["J"]).max(), np.abs(m["h"]).max()
        out = dict(jmax=jmax, hmax=hmax)
        for p in (1, 2):
            ang = ns["lr_angles"](p, 0.6, 0.3)
            pr = ns["qaoa_probabilities"](inst["energy_scaled"], ang[:p], ang[p:], inst["n"])                       # cell: max(|J|,|h|)
            prJ = ns["qaoa_probabilities"]((inst["energies"] - m["const"]) / jmax, ang[:p], ang[p:], inst["n"])     # paper main choice: max|J_ij|
            ang_c = ns["lr_angles"](p, *cal[scen]); prc = ns["view_probabilities"](view, ang_c)
            out[f"near_cell_p{p}"] = pr[inst["near_mask"]].sum(); out[f"near_maxJ_p{p}"] = prJ[inst["near_mask"]].sum(); out[f"near_cal_p{p}"] = prc[inst["near_mask"]].sum()
            out[f"ratio_cell_p{p}"] = (inst["e_max"] - pr @ inst["energies"]) / (inst["e_max"] - inst["e_min"])
            out[f"ratio_maxJ_p{p}"] = (inst["e_max"] - prJ @ inst["energies"]) / (inst["e_max"] - inst["e_min"])
        rows.append(out)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"   {scen}: mean max|J| = {mean['jmax']:.4f}, mean max|h| = {mean['hmax']:.4f}, scale used by the cell = max of both")
    print("      " + ", ".join(f"{k}={v:.4f}" for k, v in mean.items() if k not in ("jmax", "hmax")))

print("5. recovery angles are a stationary point of <E> (mixer B)")
for scen in ns["TARGET_SHARES"]:
    inst = insts[("2 main", scen)]
    _, spectral_state, _ = ns["spectral_split"](inst)
    thetas, axis, c, note = ns["warm_start_setup"](inst, "rounded spectral, mixer B", spectral_state)
    wview = ns["t4_view"](inst, thetas, axis)
    f = lambda a: float(ns["view_probabilities"](wview, a) @ inst["energies"])
    for p in (1, 2):
        x0 = np.concatenate([np.zeros(p), [np.pi / 2], np.zeros(p - 1)])
        g = np.array([(f(x0 + 1e-5 * e) - f(x0 - 1e-5 * e)) / 2e-5 for e in np.eye(2 * p)])
        H = np.array([[(f(x0 + 1e-3 * (ei + ej)) - f(x0 + 1e-3 * (ei - ej)) - f(x0 - 1e-3 * (ei - ej)) + f(x0 - 1e-3 * (ei + ej))) / 4e-6 for ej in np.eye(2 * p)] for ei in np.eye(2 * p)])
        print(f"   {scen} p={p}: <E>(recovery) = {f(x0):.4f} (E spectral {inst['energies'][spectral_state]:.4f}, E_min {inst['e_min']:.4f}); |grad| = {np.abs(g).max():.2e}; Hessian eigenvalues {np.round(np.linalg.eigvalsh((H + H.T) / 2), 3).tolist()}")
