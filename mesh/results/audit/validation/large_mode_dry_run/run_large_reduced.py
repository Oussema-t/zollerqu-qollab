import sys, time, os, numpy as np
sys.path.insert(0, "..")
from load_nb import load
os.makedirs("results", exist_ok=True)
t0 = time.time()
ns = load("large", upto=7)
print(f"[notebook cells in large mode: {time.time() - t0:.0f} s]", flush=True)
ns["RUN"]["quantum_cases"] = ["2 main"]
ns["TASK04_OVERRIDES"] = dict(eval_budget=30, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 8), lr_grid_beta=np.linspace(0.1, 1.2, 5),
                              classical_repeats=3, use_checkpoint=False, calibration_seeds=[101], sa_calibration_repeats=2,
                              spsa=dict(alpha=0.602, gamma=0.101, stability_fraction=0.1, calibration_pairs=5, perturbation_grid=(0.05,), first_step_grid=(0.03,)),
                              sa_start_fraction_grid=(0.3,), sa_end_ratio_grid=(1e-2,))
t0 = time.time()
code = "".join(open(f"../task04_part{i}.py").read() for i in range(1, 7))
exec(compile(code, "task04_cell", "exec"), ns)
print(f"[task 04 large reduced-settings run: {time.time() - t0:.0f} s]")
