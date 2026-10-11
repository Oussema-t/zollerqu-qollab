CFG = dict(workdir="wd_largeA", mode="large", run=dict(region_counts=[12, 20, 24], seeds=[7], quantum_cases=["2 main"]),
           overrides=dict(eval_budget=14, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 8), lr_grid_beta=np.linspace(0.1, 1.2, 5),
                          classical_repeats=3, use_checkpoint=False, sa_calibration_repeats=2),
           show_stdout=True)
def pre(ns):
    print("selected_mu =", ns["selected_mu"], "| LOCAL_SV", ns["LOCAL_STATEVECTOR_MAX_QUBITS"], "EXACT", ns["EXACT_BRUTE_FORCE_MAX_QUBITS"], "| RUN", ns["RUN"], flush=True)
def post(ns):
    qr, cr = ns.get("qaoa_results", {}), ns.get("classical_results", {})
    from collections import Counter
    print("qaoa groups per n:", Counter(k[1] for k in qr))
    print("classical per n:", Counter(k[1] for k in cr))
    print("references non-empty per n:", Counter(k[1] for k, v in ns.get("task04_references", {}).items() if v), "empty:", Counter(k[1] for k, v in ns.get("task04_references", {}).items() if not v))
    print("examples:", {k: (v["case"], v["n"]) for k, v in ns.get("t4_examples", {}).items()})
    for k, c in cr.items():
        if k[1] == 24 and k[0] == "2 main":
            print("  n=24", k[3], k[4], "best", round(c["best_cost"], 5), "ref", round(c["reference_cost"], 5), c["reference_kind"], "hit_opt", c["hit_opt"], "hit_near", c["hit_near"], "per_shot" in c)
    for k, g in qr.items():
        if k[1] == 20:
            r = g["runs"][0]
            print("  n=20", k[3][:6], k[4], k[5], k[6], "p_opt %.3e p_near %.3e ratio %.4f evals %d wall %.1f" % (r["p_opt"], r["p_near"], r["mean_ratio"], r["n_evals"], r["wall_seconds"]))
