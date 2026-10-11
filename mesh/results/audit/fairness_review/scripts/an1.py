import numpy as np, hashlib, collections
from rev_base import get_ns
ns = get_ns()
np.set_printoptions(precision=4, suppress=True, linewidth=200)
T = ns["TASK04"]; build = ns["t4_build_instance"]; region_cases = ns["region_cases"]; CASES = ns["BENCHMARK_CASES"]; SC = list(ns["TARGET_SHARES"])
insts = {(c, s): build(region_cases[(c, 12, 7)], c, 12, 7, s) for c in CASES for s in SC}
print("== 1. normalisation constants: e_mean vs const, e_std vs formula (no enumeration needed?)")
for (c, s), inst in insts.items():
    m = inst["model"]
    print(f"  {c:<16}{s[:8]:<9} e_mean-const={inst['e_mean']-m['const']:+.2e}  e_std-formula={inst['e_std']-inst['e_std_formula']:+.2e}  e_min={inst['e_min']:.4f} e_max={inst['e_max']:.3f} e_mean={inst['e_mean']:.3f} e_std={inst['e_std']:.3f} n_opt={len(inst['optimal_states'])} n_near={int(inst['near_mask'].sum())} fp={inst['fingerprint']}")
print("== 2. duplicated / nested instances")
for s in SC:
    fps = collections.defaultdict(list)
    for c in CASES: fps[insts[(c, s)]["fingerprint"]].append(c)
    print("  ", s, {k: v for k, v in fps.items() if len(v) > 1})
for a, b in (("6 three sources", "7 materials"), ("7 materials", "8 coupling"), ("8 coupling", "9 priority"), ("5 L uniform", "6 three sources"), ("1 control", "5 L uniform")):
    ra, rb = region_cases[(a, 12, 7)], region_cases[(b, 12, 7)]
    same_cells = ra["cell_region"].shape == rb["cell_region"].shape and np.array_equal(ra["cell_region"], rb["cell_region"])
    print(f"  {a} vs {b}: same cell->region map: {same_cells}; same region_pairs: {ra['region_pairs'].shape == rb['region_pairs'].shape and np.array_equal(ra['region_pairs'], rb['region_pairs'])}; "
          f"work equal: {ra['region_work'].shape == rb['region_work'].shape and np.allclose(ra['region_work'], rb['region_work'])}; cost equal: {ra['region_boundary_cost'].shape == rb['region_boundary_cost'].shape and np.allclose(ra['region_boundary_cost'], rb['region_boundary_cost'])}")
print("== 3. calibration instances vs scored instances of the same case")
for s in SC:
    for ci in ns["calibration_instances"][s]:
        sc = insts[(ci["case"], s)]
        print(f"  {s[:8]:<9} calib {ci['case']} seed {ci['seed']}: fp {ci['fingerprint']} vs scored fp {sc['fingerprint']}; e_min {ci['e_min']:.4f} vs {sc['e_min']:.4f}; e_std {ci['e_std']:.3f} vs {sc['e_std']:.3f}; "
              f"n_near {int(ci['near_mask'].sum())} vs {int(sc['near_mask'].sum())}; #pairs {len(ci['model']['pairs'])} vs {len(sc['model']['pairs'])}; "
              f"sorted work max|diff| {np.max(np.abs(np.sort(ci['model']['a_tilde'])-np.sort(sc['model']['a_tilde']))):.4f}")
print("== 5. shared RNG streams between classical baselines")
for m in ("uniform random sampling", "random with target share", "simulated annealing", "local search (restarts)"):
    print(f"   len({m!r}) = {len(m)}")
inst = insts[("2 main", SC[0])]
seed_base = [T["classical_seed"], 12, 7, CASES.index("2 main"), 0]
r1 = np.random.default_rng(seed_base + [len("uniform random sampling"), 0]).integers(0, 2, size=(1000, 12))
r2 = np.random.default_rng(seed_base + [len("local search (restarts)"), 0]).integers(0, 2, size=12)
print("   first uniform-sampling candidate:", r1[0], "\n   first local-search start      :", r2, "\n   identical:", np.array_equal(r1[0], r2))
print("== 4. simulated annealing / local search correctness on scored instances")
import itertools
sa, ls = ns["simulated_annealing"], ns["local_search"]
for (c, s) in [("2 main", SC[0]), ("3 hard", SC[1])]:
    inst = insts[(c, s)]
    bad = 0
    for rep in range(200):
        e, st = sa(inst, np.random.default_rng([1, rep]), 1000)
        bad += abs(e - inst["energies"][st]) > 1e-9
        e2, st2 = ls(inst, np.random.default_rng([2, rep]), 1000)
        bad += abs(e2 - inst["energies"][st2]) > 1e-9
    # delta formula vs brute force
    m = inst["model"]; J_sym = m["J"] + m["J"].T
    rng = np.random.default_rng(0); worst = 0
    for _ in range(200):
        z = 1.0 - 2.0 * rng.integers(0, 2, size=12); i = int(rng.integers(12))
        d = -2.0 * z[i] * (J_sym @ z + m["h"])[i]
        z2 = z.copy(); z2[i] = -z2[i]
        worst = max(worst, abs(d - (ns["ising_energy"](z2, m["J"], m["h"], m["const"]) - ns["ising_energy"](z, m["J"], m["h"], m["const"]))))
    print(f"  {c} / {s[:8]}: returned energy != energies[state] in {bad} of 400 runs; max |delta formula - brute force| = {worst:.1e}; diag(J_sym) max = {np.abs(np.diag(J_sym)).max():.1e}")
print("== 6. stopping reasons and evaluations by optimiser/objective (all runs)")
cnt = collections.Counter((r["variant"].split(" (")[0], r["optimiser"], r["objective"], r["stopping_reason"][:60]) for r in ns["task04_table"] if r["eval_budget"])
for k, v in sorted(cnt.items()): print("  ", v, k)
ev = collections.defaultdict(list)
for r in ns["task04_table"]:
    if r["eval_budget"]: ev[(r["variant"].split(" (")[0], r["optimiser"], r["objective"], r["p"])].append(r["n_evals"])
for k, v in sorted(ev.items()): print(f"   evals {k}: mean {np.mean(v):.0f} min {min(v)} max {max(v)} share at budget {np.mean(np.array(v) >= 200):.2f}")
