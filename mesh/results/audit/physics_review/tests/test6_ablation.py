"""Is the entangling part load-bearing in the CPU+GPU scenario? Ablation: fields-only cost layer (product state, no rzz) vs full LR-QAOA.
Also: case 8 vs case 9 fingerprints, calibration vs scored instance fingerprints."""
import numpy as np
from nsload import load_defs
ns = load_defs("small")
n0, seed0 = ns["RUN"]["region_counts"][0], ns["RUN"]["seeds"][0]
cal = {"2 equal devices": (0.26, 0.2), "CPU + GPU (GPU 4x)": (0.96, 0.55)}
scen = "CPU + GPU (GPU 4x)"
rows = []
for case in ns["BENCHMARK_CASES"]:
    inst = ns["t4_build_instance"](ns["region_cases"][(case, n0, seed0)], case, n0, seed0, scen)
    n, m = inst["n"], inst["model"]
    k = np.arange(2**n); z = 1 - 2 * ((k[:, None] >> np.arange(n)[None, :]) & 1)
    E_fields = z @ m["h_scaled"]                              # cost layer with all J set to zero: single-qubit RZ only -> product state
    E_J = inst["energy_scaled"] - E_fields
    out = dict(uniform=inst["near_mask"].mean(), uniform_ratio=(inst["e_max"] - inst["e_mean"]) / (inst["e_max"] - inst["e_min"]))
    share = np.ones(1)
    for _ in range(n):
        share = np.kron(share, np.array([0.2, 0.8]))
    out["bernoulli0.8_near"] = share[inst["near_mask"]].sum(); out["bernoulli0.8_ratio"] = (inst["e_max"] - share @ inst["energies"]) / (inst["e_max"] - inst["e_min"])
    for p in (1, 2):
        ang = ns["lr_angles"](p, *cal[scen])
        full = ns["qaoa_probabilities"](inst["energy_scaled"], ang[:p], ang[p:], n)
        fo = ns["qaoa_probabilities"](E_fields, ang[:p], ang[p:], n)
        jo = ns["qaoa_probabilities"](E_J, ang[:p], ang[p:], n)
        for name, pr in (("full", full), ("fields_only", fo), ("J_only", jo)):
            out[f"{name}_near_p{p}"] = pr[inst["near_mask"]].sum()
            out[f"{name}_ratio_p{p}"] = (inst["e_max"] - pr @ inst["energies"]) / (inst["e_max"] - inst["e_min"])
            out[f"{name}_opt_p{p}"] = pr[inst["optimal_states"]].sum()
        # check that fields-only really is a product state: P = prod of marginals
        marg = np.array([fo[((k >> i) & 1) == 1].sum() for i in range(n)])
        prod = np.prod(np.where(((k[:, None] >> np.arange(n)) & 1) == 1, marg, 1 - marg), axis=1)
        out[f"fields_only_product_err_p{p}"] = np.abs(prod - fo).max()
        out[f"fields_only_P1_min_p{p}"], out[f"fields_only_P1_max_p{p}"] = marg.min(), marg.max()
    rows.append(out)
mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
print("CPU + GPU scenario, n = 12, mean over the 9 cases, LR-QAOA with the calibrated (0.96, 0.55):")
for key, value in mean.items():
    print(f"   {key:<28} {value:.5f}")

fp = {}
for case in ns["BENCHMARK_CASES"]:
    for s in ns["TARGET_SHARES"]:
        fp[(case, s)] = ns["t4_build_instance"](ns["region_cases"][(case, n0, seed0)], case, n0, seed0, s)["fingerprint"]
print("case 8 vs case 9 fingerprints equal:", {s: fp[("8 coupling", s)] == fp[("9 priority", s)] for s in ns["TARGET_SHARES"]})
print("distinct fingerprints per scenario:", {s: len({fp[(c, s)] for c in ns["BENCHMARK_CASES"]}) for s in ns["TARGET_SHARES"]})
# calibration instances differ from the scored ones
T = ns["TASK04"]
cfp = set()
for case_name in T["calibration_cases"]:
    for cs in T["calibration_seeds"]:
        spec = ns["MESH_CASES"][case_name]
        pts, tri = ns["build_case_mesh"](spec, cs)
        mesh = ns["add_case_weights"](spec, ns["check_case_mesh"](case_name, spec, pts, tri))
        regions = ns["coarsen_mesh"](mesh, T["calibration_n"], cs)
        for s in ns["TARGET_SHARES"]:
            cfp.add(ns["t4_build_instance"](regions, case_name, T["calibration_n"], cs, s)["fingerprint"])
print("calibration fingerprints:", len(cfp), "| overlap with scored instances:", len(cfp & set(fp.values())))
