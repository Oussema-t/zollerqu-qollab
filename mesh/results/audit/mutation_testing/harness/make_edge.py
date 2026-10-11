SMOKE_D = "dict(eval_budget=40, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 12), lr_grid_beta=np.linspace(0.1, 1.2, 6), classical_repeats=5, use_checkpoint=False, calibration_seeds=[101], sa_calibration_repeats=3)"
INJECT_ANCHOR = '    view, key_seed, scenario = t4_view(inst), t4_key_seed(key), key[3]\n'
def inject(case, scen):
    return [(INJECT_ANCHOR, INJECT_ANCHOR + f'    if key[0] == "{case}" and key[3] == "{scen}":\n        raise RuntimeError("injected failure for the review")\n', 1)]
POST_SUMMARY = '''
def post(ns):
    import json, os
    out = open("stdout.txt").read()
    keep = [l for l in out.split("\\n") if l.startswith("TASK 04 RESULT") or "measured wall-clock" in l or "QAOA methods at n" in l or "failures:" in l or "instance(s):" in l or "interpretation" in l or "vs classical" in l]
    for l in keep: print("    | " + l[:330])
    print("    qaoa groups:", len(ns.get("qaoa_results", {})), "| cases with QAOA:", sorted({k[0] + "/" + k[3][:5] for k in ns.get("qaoa_results", {})}), "| reused", ns.get("t4_reused"), "computed", ns.get("t4_computed"))
'''
E = {
 "e01_layers_1": dict(run="dict(quantum_cases=['2 main', '9 priority'], qaoa_layers=[1])", ov=SMOKE_D),
 "e02_random_starts_0": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("random_starts=1", "random_starts=0")),
 "e03_single_case": dict(run="dict(quantum_cases=['2 main'])", ov=SMOKE_D),
 "e04_no_2main": dict(run="dict(quantum_cases=['9 priority'])", ov=SMOKE_D),
 "e04b_two_other": dict(run="dict(quantum_cases=['3 hard', '6 three sources'])", ov=SMOKE_D),
 "e05_shots_100": dict(run="dict(quantum_cases=['2 main', '9 priority'], shots=100)", ov=SMOKE_D),
 "e06_mu_positive": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D, pre="def pre(ns):\n    ns['selected_mu'] = 2.0\n"),
 "e07_no_quantum_cases": dict(run="dict(quantum_cases=[])", ov=SMOKE_D),
 "e08_fail_nonexample": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D, mut=inject("9 priority", "CPU + GPU (GPU 4x)")),
 "e09_fail_example": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D, mut=inject("2 main", "2 equal devices")),
 "e10a_ckpt_first": dict(run="dict(quantum_cases=['2 main'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=True"), wd="wd_ckpt10"),
 "e10b_ckpt_second_more_cases": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=True"), wd="wd_ckpt10"),
 "e11a_ckpt_with_failure": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=True"), wd="wd_ckpt11", mut=inject("9 priority", "CPU + GPU (GPU 4x)")),
 "e11b_ckpt_rerun_fixed": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=True"), wd="wd_ckpt11"),
 "e12a_ckpt_buggy_code": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=True"), wd="wd_ckpt12",
                              mut=[("(1.0 - (k - 1) / p) * delta_beta", "(k / p) * delta_beta", 1)]),
 "e12b_ckpt_fixed_code": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=True"), wd="wd_ckpt12"),
 "e13_classical_budget_200": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("classical_repeats=5", "classical_repeats=40, classical_budget=200")),
 "e14_eval_budget_8": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("eval_budget=40", "eval_budget=8")),
 "e15_stress_layers_empty": dict(run="dict(quantum_cases=['2 main', '9 priority'])", ov=SMOKE_D.replace("use_checkpoint=False", "use_checkpoint=False, stress_layers=[]")),
 "e16_layers_2_3": dict(run="dict(quantum_cases=['2 main', '9 priority'], qaoa_layers=[2, 3])", ov=SMOKE_D),
}
for name, e in E.items():
    wd = e.get("wd", "wd_" + name)
    s = f'CFG = dict(workdir="{wd}", mode="small", run={e["run"]}, overrides={e["ov"]}, show_tests=False, mutations={e.get("mut", [])!r})\n'
    s += e.get("pre", "") + POST_SUMMARY
    open(f"cfg/{name}.py", "w").write(s)
print(len(E))
