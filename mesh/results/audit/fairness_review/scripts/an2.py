import csv, gzip, json, numpy as np, collections
rows = list(csv.DictReader(open("full/results/task04_runs_small.csv")))
for r in rows:
    for k in ("p", "n", "n_evals"): r[k] = int(r[k])
    for k in ("p_opt", "p_near", "hit_near", "hit_opt", "uniform_p_near", "uniform_p_opt", "mean_cost", "mean_ratio", "p_near_noise_estimate", "modelled_hours", "sampled_best_cost"): r[k] = float(r[k])
ck = json.load(gzip.open("full/results/task04_checkpoint_small.json.gz", "rt"))["instances"]
CASES = ["1 control", "2 main", "3 hard", "4 structural", "5 L uniform", "6 three sources", "7 materials", "8 coupling", "9 priority"]
SC = ["2 equal devices", "CPU + GPU (GPU 4x)"]
def q(scen, variant, opt, p, case, init_random=False):
    return [r for r in rows if r["scenario"] == scen and r["variant"] == variant and r["optimiser"] == opt and r["p"] == p and r["case"] == case and (r["init"] == "random") == init_random]
print("== 9. per-case probability that 1000 candidates contain a split within 5 % (hit_near)")
for scen in SC:
    print(" ", scen)
    print(f"    {'case':<17}{'LR p2':>8}{'COBYLA p2':>10}{'WSA start':>10}{'uniform':>8}{'tgt share':>10}{'local':>7}{'SA':>6}{'spectral':>9}   COBYLA-local  LR-local")
    d1, d2 = [], []
    for c in CASES:
        cl = ck[f"{c}|12|7|{scen}"]["classical"]
        lr = q(scen, "LR-QAOA", "fixed schedule", 2, c)[0]["hit_near"]; co = q(scen, "QAOA", "COBYLA", 2, c)[0]["hit_near"]
        ws = q(scen, "warm-start QAOA (rounded spectral, mixer A)", "initial state only", 0, c)[0]["hit_near"]
        loc, sa, sp = cl["local search (restarts)"]["hit_near"], cl["simulated annealing"]["hit_near"], cl["spectral bisection"]["hit_near"]
        un, ts = cl["uniform random sampling"]["per_shot"]["hit_near"], cl["random with target share"]["per_shot"]["hit_near"]
        d1.append(co - loc); d2.append(lr - loc)
        print(f"    {c:<17}{lr:>8.3f}{co:>10.3f}{ws:>10.3f}{un:>8.3f}{ts:>10.3f}{loc:>7.2f}{sa:>6.2f}{sp:>9.2f}   {co-loc:>+10.3f}  {lr-loc:>+8.3f}")
    for name, d in (("QAOA COBYLA - local search", d1), ("LR-QAOA - local search", d2)):
        d = np.array(d); d8 = d[:8]
        print(f"    {name}: mean {d.mean():+.3f}, sd {d.std(ddof=1):.3f}, se(9 cases) {d.std(ddof=1)/3:.3f}, t = {d.mean()/(d.std(ddof=1)/3):.2f}; cases ahead/behind {int((d>0).sum())}/{int((d<0).sum())}; "
              f"8 distinct instances: mean {d8.mean():+.3f}, t = {d8.mean()/(d8.std(ddof=1)/np.sqrt(8)):.2f}; without '2 main': mean {np.delete(d,1).mean():+.3f}")
print("== 10. plot 6: is the classical 'best split' the best over ALL repeats (repeats x budget candidates)?")
for scen in SC:
    for m in ("uniform random sampling", "local search (restarts)", "simulated annealing"):
        recs = [ck[f"{c}|12|7|{scen}"]["classical"][m] for c in CASES]
        print(f"    {scen[:8]:<9}{m:<26} repeats {recs[0]['repeats']} x {recs[0]['evaluations_per_repeat']} evals; best over repeats == exact optimum in {sum(abs(r['best_cost']-r['reference_cost'])<1e-9 for r in recs)}/9 cases; "
              f"share of single repeats that reach the optimum (hit_opt): mean {np.mean([r['hit_opt'] for r in recs]):.2f}, min {min(r['hit_opt'] for r in recs):.2f}")
    lrb = [q(scen, "LR-QAOA", "fixed schedule", 2, c)[0] for c in CASES]
    print(f"    {scen[:8]:<9}LR-QAOA p=2 (one run of 1000 shots): sampled best == optimum in {sum(abs(r['sampled_best_cost'] - ck[chr(124).join([r['case'],'12','7',scen])]['classical']['simulated annealing']['reference_cost']) < 1e-9 for r in lrb)}/9 cases")
print("== 11. LR-QAOA p_near versus p, per case (title: 'More layers help without noise')")
for scen in SC:
    non_mono = 0
    for c in CASES:
        v = [q(scen, "LR-QAOA", "fixed schedule", p, c)[0]["p_near"] for p in (1, 2, 3, 4, 5)]
        nz = [q(scen, "LR-QAOA", "fixed schedule", p, c)[0]["p_near_noise_estimate"] for p in (1, 2, 3, 4, 5)]
        mono = all(b >= a for a, b in zip(v, v[1:])); non_mono += not mono
        print(f"    {scen[:8]:<9}{c:<17}" + " ".join(f"{x:.4f}" for x in v) + ("   monotone" if mono else "   NOT monotone") + "   noise est: " + " ".join(f"{x:.4f}" for x in nz))
    print(f"    {scen[:8]}: cases where more layers do not always help: {non_mono}/9")
for variant in ("LR-QAOA (success-calibrated)", "LR-QAOA (paper default 0.6/0.3)"):
    for scen in SC:
        v = [np.mean([q(scen, variant, "fixed schedule", p, c)[0]["p_near"] for c in CASES]) for p in (1, 2, 3, 4, 5)]
        print(f"    {variant} {scen[:8]}: mean p_near by p = " + " ".join(f"{x:.4f}" for x in v))
print("== 12. 'below uniform' counts: size of the differences")
for scen in SC:
    for variant, opt, p in (("warm-start QAOA (relaxation, mixer A)", "initial state only", 0), ("warm-start QAOA (rounded spectral, mixer B)", "COBYLA", 2), ("warm-start QAOA (rounded spectral, mixer A)", "COBYLA", 1), ("QAOA", "SPSA", 1), ("CVaR-QAOA", "SPSA", 1)):
        rs = [q(scen, variant, opt, p, c)[0] for c in CASES]
        print(f"    {scen[:8]:<9}{variant[:42]:<43}{opt[:10]:<11}p={p}: below {sum(r['p_near'] < r['uniform_p_near'] for r in rs)}/9; p_near - uniform: " + " ".join(f"{r['p_near']-r['uniform_p_near']:+.1e}" for r in rs))
print("== 13. per-case 'x uniform' ratios (table shows ratio of means)")
for scen in SC:
    for label, variant, opt, p in (("LR-QAOA", "LR-QAOA", "fixed schedule", 2), ("QAOA COBYLA", "QAOA", "COBYLA", 2), ("CVaR SPSA", "CVaR-QAOA", "SPSA", 2), ("LR success-cal", "LR-QAOA (success-calibrated)", "fixed schedule", 2), ("QAOA SPSA", "QAOA", "SPSA", 2)):
        rs = [q(scen, variant, opt, p, c)[0] for c in CASES]
        ratios = np.array([r["p_near"] / r["uniform_p_near"] for r in rs])
        print(f"    {scen[:8]:<9}{label:<15} ratio of means {np.mean([r['p_near'] for r in rs]) / np.mean([r['uniform_p_near'] for r in rs]):6.1f}; mean of ratios {ratios.mean():6.1f}; per case " + " ".join(f"{x:.1f}" for x in ratios))
print("== 13b. random starts vs uniform: share of random-start runs below uniform (table only counts the LR start)")
for scen in SC:
    for variant, opt in (("QAOA", "COBYLA"), ("QAOA", "Nelder-Mead"), ("QAOA", "Powell"), ("QAOA", "L-BFGS-B"), ("QAOA", "SPSA"), ("CVaR-QAOA", "COBYLA"), ("CVaR-QAOA", "SPSA")):
        for p in (1, 2):
            rs = [r for c in CASES for r in q(scen, variant, opt, p, c, init_random=True)]
            print(f"    {scen[:8]:<9}{variant:<10}{opt:<12}p={p}: random-start runs {len(rs)}; below uniform {sum(r['p_near'] < r['uniform_p_near'] for r in rs)}; median p_near {np.median([r['p_near'] for r in rs]):.4f}; mean {np.mean([r['p_near'] for r in rs]):.4f}")
print("== WS variants vs their own start state (p=2, COBYLA from given start)")
for scen in SC:
    for fl in ("relaxation, mixer A", "rounded spectral, mixer A", "rounded spectral, mixer B"):
        v = f"warm-start QAOA ({fl})"
        worse = 0; out = []
        for c in CASES:
            s0 = q(scen, v, "initial state only", 0, c)[0]["p_near"]; s2 = q(scen, v, "COBYLA", 2, c)[0]["p_near"]
            sp = ck[f"{c}|12|7|{scen}"]["classical"]["spectral bisection"]["hit_near"]
            worse += s2 < s0 - 1e-12; out.append(f"{s0:.3f}>{s2:.3f}(sp {sp:.0f})")
        print(f"    {scen[:8]:<9}{fl:<27} optimised p=2 below own start state in {worse}/9 cases: " + " ".join(out))
