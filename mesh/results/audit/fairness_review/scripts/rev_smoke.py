"""Reviewer smoke driver. usage: rev_smoke.py <workdir> <use_ckpt 0/1> <cases ;-separated> [inject.py]"""
import sys, time, os, io, contextlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
workdir, use_ckpt, cases = sys.argv[1], bool(int(sys.argv[2])), sys.argv[3].split(";")
inject = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(os.path.join(workdir, "results"), exist_ok=True)
os.chdir(workdir)
from load_nb import load
ns = load("small", upto=7)
ns["RUN"]["quantum_cases"] = cases
ns["TASK04_OVERRIDES"] = dict(eval_budget=40, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 12), lr_grid_beta=np.linspace(0.1, 1.2, 6),
                              classical_repeats=5, use_checkpoint=use_ckpt, calibration_seeds=[101], sa_calibration_repeats=3)
code = open(os.path.join(HERE, "task04_cell.py")).read()
if inject:
    code = open(inject).read() + "\n" + code if not inject.endswith(".patch.py") else code
    if inject.endswith(".patch.py"):
        patch_ns = {}
        exec(open(inject).read(), patch_ns)
        code = patch_ns["patch"](code)
t0 = time.time()
exec(compile(code, "task04_cell", "exec"), ns)
print(f"[rev smoke: {time.time() - t0:.1f} s; computed {ns['t4_computed']}, reused {ns['t4_reused']}, failures {len(ns['task04_failures'])}, qaoa groups {len(ns['qaoa_results'])}, rows {len(ns['task04_table'])}]")
