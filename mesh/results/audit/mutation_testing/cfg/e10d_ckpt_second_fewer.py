CFG = dict(workdir="wd_ckpt10r", mode="small", run=dict(quantum_cases=['2 main']), overrides=dict(eval_budget=40, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 12), lr_grid_beta=np.linspace(0.1, 1.2, 6), classical_repeats=5, use_checkpoint=True, calibration_seeds=[101], sa_calibration_repeats=3), show_tests=False, mutations=[])

def post(ns):
    import json, os
    out = open("stdout.txt").read()
    keep = [l for l in out.split("\n") if l.startswith("TASK 04 RESULT") or "measured wall-clock" in l or "QAOA methods at n" in l or "failures:" in l or "instance(s):" in l or "interpretation" in l or "vs classical" in l]
    for l in keep: print("    | " + l[:330])
    print("    qaoa groups:", len(ns.get("qaoa_results", {})), "| cases with QAOA:", sorted({k[0] + "/" + k[3][:5] for k in ns.get("qaoa_results", {})}), "| reused", ns.get("t4_reused"), "computed", ns.get("t4_computed"))
