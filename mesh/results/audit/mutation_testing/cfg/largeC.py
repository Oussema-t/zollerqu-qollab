CFG = dict(workdir="wd_largeC", mode="large", run=dict(region_counts=[20, 24], seeds=[7], quantum_cases=["2 main"]),
           overrides=dict(eval_budget=12, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 8), lr_grid_beta=np.linspace(0.1, 1.2, 5),
                          classical_repeats=3, use_checkpoint=False, sa_calibration_repeats=2), show_tests=True)
def post(ns):
    import collections
    out = open("stdout.txt").read()
    for l in out.split("\n"):
        if l.startswith("TASK 04 RESULT") or "measured wall-clock" in l or "QAOA methods at n" in l or "failures:" in l or "instance(s):" in l or "depth stress" in l or ", n = 1" in l or "vs classical" in l or "uniform control" in l or "LR-QAOA  " in l or "QAOA COBYLA " in l:
            print("    | " + l[:300])
    qr = ns.get("qaoa_results", {})
    print("    groups per (n, seed):", sorted(collections.Counter((k[1], k[2]) for k in qr).items()))
    ms = ns.get("t4_method_summary", [])
    print("    method_summary instances field:", sorted({m["instances"] for m in ms}), "| T4_QUANTUM_CASES:", ns.get("T4_QUANTUM_CASES"))
    for f in sorted(__import__("os").listdir("results")):
        if f.startswith("task04"): print("    file", f, __import__("os").path.getsize("results/" + f))
