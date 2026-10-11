CFG = dict(workdir="wd_s01", mode="small", run=dict(quantum_cases=["2 main", "9 priority"]),
           overrides=dict(eval_budget=40, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 12), lr_grid_beta=np.linspace(0.1, 1.2, 6), classical_repeats=5, use_checkpoint=False, calibration_seeds=[101], sa_calibration_repeats=3),
           show_tests=False)

def post(ns):
    import numpy as np, json, os, collections, copy, io, contextlib
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    P = lambda *a: print("   ", *a)
    qr, cr, cases = ns["qaoa_results"], ns["classical_results"], ns["task04_ising_cases"]
    # ---- 1-3 structure
    P("[S1] qaoa key lens:", collections.Counter(len(k) for k in qr), "types:", [type(x).__name__ for x in next(iter(qr))], "example:", next(iter(qr)))
    need = ["init", "init_seed", "run_seed", "initial_angles", "final_angles", "objective", "eval_budget", "n_evals", "p_opt", "p_near", "mean_cost", "mean_ratio",
            "p_balanced", "sampled_p_opt", "sampled_p_near", "sampled_mean_cost", "sampled_p_balanced", "history", "stopping_reason", "outcome", "wall_seconds",
            "optimiser", "p", "shots", "variant", "n_qubits"]
    miss = collections.Counter()
    for k, g in qr.items():
        for r in g["runs"]:
            for f in need:
                if f not in r: miss[f] += 1
    P("[S2] fields missing from the per-run dict (count of runs):", dict(miss), "| group-level keys:", list(next(iter(qr.values())).keys()))
    P("[S3] classical key lens:", collections.Counter(len(k) for k in cr), "methods:", sorted({k[4] for k in cr}), "records:", len(cr), "instances:", len({k[:4] for k in cr}))
    # ---- 4 calibration instance vs scored instance
    for scen, insts in ns["calibration_instances"].items():
        for ci in insts:
            sk = (ci["case"], ci["n"], 7, scen)
            if sk in cases:
                si = cases[sk]
                J1, J2 = ci["model"]["J"], si["model"]["J"]
                P(f"[S4] calibration {ci['case']} seed {ci['seed']} vs scored seed 7 ({scen[:9]}): fingerprint equal = {ci['fingerprint'] == si['fingerprint']}; "
                  f"max|J diff| = {np.max(np.abs(J1 - J2)):.4f} (max|J| = {np.max(np.abs(J2)):.4f}); corr(J) = {np.corrcoef(J1[np.triu_indices(12, 1)], J2[np.triu_indices(12, 1)])[0, 1]:.4f}; e_min {ci['e_min']:.5f} vs {si['e_min']:.5f}")
    # ---- 5 case 8 vs 9
    for scen in ns["TARGET_SHARES"]:
        a, b = cases[("8 coupling", 12, 7, scen)], cases[("9 priority", 12, 7, scen)]
        P(f"[S5] case 8 vs case 9 ({scen[:9]}): fingerprint equal = {a['fingerprint'] == b['fingerprint']}, mu = {b['mu']}, priority_active = {b['priority_active']}")
    # ---- 6 relaxation warm start c values
    for k, g in qr.items():
        if k[4] == "warm-start QAOA (relaxation, mixer A)" and k[5].startswith("fixed") and k[6] == 1:
            c = np.array(g["warm_start_c"])
            P(f"[S6] relaxation warm-start c for {k[0]} / {k[3][:9]}: min {c.min():.6f} max {c.max():.6f} (alpha = {ns['TARGET_SHARES'][k[3]]}) -> all equal to alpha: {np.allclose(c, ns['TARGET_SHARES'][k[3]], atol=1e-6)}")
    # ---- 7 below-uniform float comparison
    for k, g in qr.items():
        if k[5] == "initial state only" and "relaxation" in k[4]:
            r = g["runs"][0]
            P(f"[S7] {k[0]} / {k[3][:9]} relax start only: p_near - uniform_p_near = {r['p_near'] - g['uniform_p_near']:.3e}; counted as below uniform: {r['p_near'] < g['uniform_p_near']}")
    # ---- 8/9/10 non-enumerated code path equivalence at n = 12
    heavy = ns["t4_heavy_fields"]
    for scen, inst in ns["t4_examples"].items():
        inst2 = {k: v for k, v in inst.items() if k not in heavy}
        inst2["exact"] = False
        rng = np.random.default_rng(1)
        worst = 0.0; diff_fields = collections.Counter()
        for s in rng.integers(0, 2 ** inst["n"], 300):
            a, b = ns["modelled_outcome"](inst, int(s)), ns["modelled_outcome"](inst2, int(s))
            for f in a:
                d = abs(a[f] - b[f])
                if d > 1e-9: diff_fields[f] += 1; worst = max(worst, d)
            worst = max(worst, abs(ns["state_cost"](inst, int(s)) - ns["state_cost"](inst2, int(s))))
        P(f"[S8] {scen[:9]}: modelled_outcome/state_cost enumerated vs non-enumerated branch on 300 random states: max abs diff {worst:.2e}, differing fields {dict(diff_fields)}")
        ra, rb = ns["run_classical"](inst), ns["run_classical"](inst2)
        for m in ra:
            same = np.allclose(ra[m]["best_costs"], rb[m]["best_costs"], atol=1e-9) and ra[m]["best_state"] == rb[m]["best_state"]
            P(f"[S10] {scen[:9]} run_classical exact vs non-exact path, {m}: same best costs/state = {same}; hit_opt {ra[m]['hit_opt']} vs {rb[m]['hit_opt']} ({rb[m]['reference_kind']}); ref {ra[m]['reference_cost']:.5f} vs {rb[m]['reference_cost']:.5f}; per_shot in non-exact: {'per_shot' in rb[m]}")
    # ---- 11 RNG stream collision between methods
    names = ["uniform random sampling", "random with target share", "simulated annealing", "local search (restarts)"]
    P("[S11] len(method) used as seed component:", {m: len(m) for m in names})
    inst = next(iter(ns["t4_examples"].values()))
    seed_base = [ns["TASK04"]["classical_seed"], inst["n"], inst["seed"], ns["BENCHMARK_CASES"].index(inst["case"]), list(ns["TARGET_SHARES"]).index(inst["scenario"])]
    r1 = np.random.default_rng(seed_base + [len(names[0]), 0]).integers(0, 2, size=(1000, inst["n"]))
    r2 = np.random.default_rng(seed_base + [len(names[3]), 0]).integers(0, 2, size=inst["n"])
    P("[S11] first candidate of 'uniform random sampling' repeat 0:", r1[0].tolist(), "| first start of 'local search' repeat 0:", r2.tolist(), "| identical:", bool(np.array_equal(r1[0], r2)))
    # ---- 15 key seed collisions for the large default configuration
    ks = collections.Counter()
    for ci, c in enumerate(ns["BENCHMARK_CASES"]):
        for si in range(2):
            for n in (12, 16, 20, 24):
                for seed in (7, 8, 9):
                    ks[1000 * ci + 100 * si + 10 * n + seed] += 1
    P("[S15] t4_key_seed collisions in the large default grid:", sum(v > 1 for v in ks.values()), "of", len(ks),
      "| example with seed >= 10: key_seed(n=12, seed=17) =", 10 * 12 + 17, "== key_seed(n=13, seed=7) =", 10 * 13 + 7)
    # ---- 19 sampled vs exact z-scores
    zs = []
    for k, g in qr.items():
        for r in g["runs"]:
            p = r["p_near"]
            if 0.005 < p < 0.995: zs.append((r["sampled_p_near"] - p) / np.sqrt(p * (1 - p) / ns["T4_SHOTS"]))
    zs = np.array(zs)
    P(f"[S19] z-scores sampled_p_near vs exact p_near over {len(zs)} runs: mean {zs.mean():+.3f}, std {zs.std():.3f}, max |z| {np.abs(zs).max():.2f}")
    # ---- 20/21 best_state consistency for every run (incl. symmetry-reduced), with independent recomputation
    def pieces(x, pairs):
        parent = list(range(len(x)))
        def find(a):
            while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i, j in pairs:
            if x[i] == x[j]: parent[find(i)] = find(j)
        roots = {find(i): x[i] for i in range(len(x))}
        return sum(1 for v in roots.values() if v == 0), sum(1 for v in roots.values() if v == 1)
    bad = collections.Counter(); nrun = 0
    for k, g in qr.items():
        inst = cases[k[:4]]; model = inst["model"]; n = inst["n"]
        for r in g["runs"]:
            nrun += 1
            s = r["best_state"]; x = np.array([(s >> i) & 1 for i in range(n)]); z = 1 - 2 * x
            i_, j_ = model["pairs"][:, 0], model["pairs"][:, 1]
            cut = float(np.sum(model["c_tilde"] * (x[i_] != x[j_])))
            share = float(model["a_tilde"] @ x)
            cost = cut + inst["lam"] * 4 * (inst["alpha"] - share) ** 2 + inst["mu"] * float(np.sum(model["p_tilde"] * (1 - x)))
            o = r["outcome"]
            if abs(cost - r["sampled_best_cost"]) > 1e-9: bad["cost"] += 1
            if o["state"] != s: bad["state"] += 1
            if abs(o["share_b"] - share) > 1e-9: bad["share_b"] += 1
            if abs(o["cut_fraction"] - cut) > 1e-9: bad["cut"] += 1
            if abs(o["imbalance"] - abs(share - inst["alpha"])) > 1e-9: bad["imbalance"] += 1
            if (o["pieces_a"], o["pieces_b"]) != pieces(x, inst["regions"]["region_pairs"]): bad["pieces"] += 1
            sa, sb = inst["speeds"]; H = ns["TASK04"]["job_reference_hours"]
            hours = H * max((1 - share) / sa, share / sb) + ns["TASK04"]["comm_fraction_if_all_borders_cut"] * H * cut
            if abs(o["modelled_hours"] - hours) > 1e-9: bad["hours"] += 1
            if g["n_qubits"] == n - 1 and (s & 1): bad["reduced_x0_not_0"] += 1
            if r["outcome_runtime_scored"]["modelled_hours"] > o["modelled_hours"] + 1e-9: bad["runtime_scored_worse_than_lowest_cost"] += 1
    P(f"[S20] independent recomputation of best_state cost / share / cut / pieces / hours for {nrun} runs: mismatches {dict(bad)}")
    # classical records too
    badc = collections.Counter()
    for k, c in cr.items():
        inst = cases[k[:4]]; model = inst["model"]; n = inst["n"]; s = c["best_state"]; x = np.array([(s >> i) & 1 for i in range(n)])
        i_, j_ = model["pairs"][:, 0], model["pairs"][:, 1]
        cost = float(np.sum(model["c_tilde"] * (x[i_] != x[j_]))) + inst["lam"] * 4 * (inst["alpha"] - float(model["a_tilde"] @ x)) ** 2
        if abs(cost - c["best_cost"]) > 1e-9: badc["cost"] += 1
        if (c["outcome"]["pieces_a"], c["outcome"]["pieces_b"]) != pieces(x, inst["regions"]["region_pairs"]): badc["pieces"] += 1
    P(f"[S20c] classical best_state cost/pieces recomputed for {len(cr)} records: mismatches {dict(badc)}")
    # ---- 22-24 independent brute force + independent Qiskit circuit on the example instances
    for scen, inst in ns["t4_examples"].items():
        model, n = inst["model"], inst["n"]
        X = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
        i_, j_ = model["pairs"][:, 0], model["pairs"][:, 1]
        share = X @ model["a_tilde"]
        E = (X[:, i_] != X[:, j_]) @ model["c_tilde"] + inst["lam"] * 4 * (inst["alpha"] - share) ** 2
        P(f"[S22] {scen[:9]} brute-force energies vs inst['energies']: max diff {np.max(np.abs(E - inst['energies'])):.2e}; optimum {E.min():.6f}; #opt {np.sum(E <= E.min() + 1e-9)}; #near {np.sum(E <= 1.05 * E.min() + 1e-9)}")
        key = (inst["case"], n, inst["seed"], scen)
        for p in ns["T4_LAYERS"]:
            g = qr[key + ("LR-QAOA", "fixed schedule", p)]; r = g["runs"][0]
            dg, db = ns["T4_LR_DELTAS"][scen]["LR-QAOA"]
            kk = np.arange(1, p + 1); gam, bet = kk / p * dg, (1 - (kk - 1) / p) * db
            assert np.allclose(np.concatenate([gam, bet]), r["final_angles"]), "stored angles differ from the LR formula"
            qc = QuantumCircuit(n)
            Js, hs = model["J"] / model["scale"], model["h"] / model["scale"]
            qc.h(range(n))
            for l in range(p):
                for a in range(n):
                    for b in range(a + 1, n):
                        if Js[a, b] != 0: qc.rzz(2 * gam[l] * Js[a, b], a, b)
                for a in range(n):
                    if hs[a] != 0: qc.rz(2 * gam[l] * hs[a], a)
                for a in range(n): qc.rx(-2 * bet[l], a)
            pr = Statevector(qc).probabilities()
            mine = dict(p_opt=pr[E <= E.min() + 1e-9].sum(), p_near=pr[E <= 1.05 * E.min() + 1e-9].sum(), mean_cost=pr @ E, mean_ratio=(E.max() - pr @ E) / (E.max() - E.min()),
                        p_balanced=pr[np.abs(share - inst["alpha"]) <= 0.05].sum())
            P(f"[S23] {scen[:9]} LR-QAOA p={p}: independent Qiskit+brute force vs stored: " + ", ".join(f"{f} {mine[f]:.6f}/{r[f]:.6f}" for f in mine) + f" -> max diff {max(abs(mine[f] - r[f]) for f in mine):.2e}")
    # ---- 24 small dense check of the simulator against matrix exponentials (n = 3 and 5) and per-qubit path (n = 14) against Qiskit
    Xm, Zm, I2 = np.array([[0, 1], [1, 0]], complex), np.array([[1, 0], [0, -1]], complex), np.eye(2)
    def op(single, q, n):
        m = np.eye(1)
        for k in range(n - 1, -1, -1): m = np.kron(m, single if k == q else I2)
        return m
    rng = np.random.default_rng(5)
    for n in (3, 5):
        J = np.triu(rng.normal(size=(n, n)), 1); h = rng.normal(size=n)
        Zs = [op(Zm, q, n) for q in range(n)]; Xs = [op(Xm, q, n) for q in range(n)]
        Hc = sum(J[a, b] * Zs[a] @ Zs[b] for a in range(n) for b in range(a + 1, n)) + sum(h[a] * Zs[a] for a in range(n))
        Ediag = np.real(np.diag(Hc)); gam, bet = rng.uniform(0.1, 1, 2), rng.uniform(0.1, 1, 2)
        psi = np.full(2 ** n, 2 ** (-n / 2), complex)
        for l in range(2): psi = expm(1j * bet[l] * sum(Xs)) @ (expm(-1j * gam[l] * Hc) @ psi)
        mine = np.abs(psi) ** 2; theirs = ns["qaoa_probabilities"](Ediag, gam, bet, n)
        th = rng.uniform(0.3, 2.6, n); start = np.ones(1, complex)
        for q in range(n - 1, -1, -1): start = np.kron(start, [np.cos(th[q] / 2), np.sin(th[q] / 2)])
        res = {}
        for name, sgn in (("A", 1.0), ("B", -1.0)):
            psi = start.copy()
            for l in range(2):
                mix = sum(sgn * np.sin(th[q]) * Xs[q] + np.cos(th[q]) * Zs[q] for q in range(n))
                psi = expm(1j * bet[l] * mix) @ (expm(-1j * gam[l] * Hc) @ psi)
            res[name] = np.max(np.abs(np.abs(psi) ** 2 - ns["qaoa_probabilities"](Ediag, gam, bet, n, th, sgn * th)))
        P(f"[S24] n={n} simulator vs matrix exponentials: standard {np.max(np.abs(mine - theirs)):.2e}, mixer A {res['A']:.2e}, mixer B {res['B']:.2e}")
    n = 14
    J = np.triu(rng.normal(size=(n, n)), 1); h = rng.normal(size=n); gam, bet = np.array([0.31, 0.17]), np.array([0.42, 0.23])
    z = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    Ediag = np.sum((z @ J) * z, axis=1) + z @ h
    qc = ns["build_qaoa_circuit"](J, h, gam, bet)
    P(f"[S24] n=14 (per-qubit mixer path) numpy vs Qiskit: {np.max(np.abs(Statevector(qc).probabilities() - ns['qaoa_probabilities'](Ediag, gam, bet, n))):.2e}")
    th = rng.uniform(0.3, 2.6, n)
    for name, sgn in (("A", 1.0), ("B", -1.0)):
        qc = ns["build_qaoa_circuit"](J, h, gam, bet, thetas=th, modified_mixer=(name == "B"))
        P(f"[S24] n=14 warm mixer {name} numpy vs Qiskit: {np.max(np.abs(Statevector(qc).probabilities() - ns['qaoa_probabilities'](Ediag, gam, bet, n, th, sgn * th))):.2e}")
    # symmetry-reduced circuit (never built by the cell): Qiskit circuit from view J_scaled/h_scaled vs the simulated reduced distribution
    inst = ns["t4_examples"]["2 equal devices"]; red = ns["t4_reduced_view"](inst); ang = ns["lr_angles"](2, *ns["T4_LR_DELTAS"]["2 equal devices"]["LR-QAOA"])
    qc = ns["build_qaoa_circuit"](red["J_scaled"], red["h_scaled"], *ns["split_angles"](ang))
    P(f"[S24] symmetry-reduced circuit (11 qubits) Qiskit vs numpy: {np.max(np.abs(Statevector(qc).probabilities() - ns['view_probabilities'](red, ang))):.2e}")
    # ---- 26 SPSA / CVaR convergence history vs reported final
    for k, g in qr.items():
        if k[:4] == (inst["case"], inst["n"], inst["seed"], "2 equal devices") and k[4] == "QAOA" and k[6] == 2:
            for r in g["runs"]:
                P(f"[S26] example equal p=2 {k[5]} ({r['init'][:6]}, objective {r['objective']}): last history value {r['history'][-1]:.4f} vs reported final mean cost {r['mean_cost']:.4f}")
    # ---- 13 idempotence: re-run the cell in the same namespace and compare
    before = {k: [r["p_opt"] for r in g["runs"]] for k, g in qr.items()}
    cal_before = {s: (c["energy"], c["success"]) for s, c in ns["lr_calibration"].items()}
    code = open(os.path.join(os.path.dirname(os.path.abspath(ns["__file_runner__"])) if "__file_runner__" in ns else "..", "task04_cell.py")).read() if False else open("../task04_cell.py").read()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "task04_cell.py", "exec"), ns)
        after = {k: [r["p_opt"] for r in g["runs"]] for k, g in ns["qaoa_results"].items()}
        same = before.keys() == after.keys() and all(np.allclose(before[k], after[k], rtol=0, atol=0) for k in before)
        P("[S13] re-running the cell in the same namespace: bit-identical p_opt for all runs =", same, "| calibration identical =", cal_before == {s: (c["energy"], c["success"]) for s, c in ns["lr_calibration"].items()})
    except Exception as e:
        P("[S13] RE-RUN CRASHED:", type(e).__name__, e)
