import io, contextlib, time, sys, numpy as np, nbformat, matplotlib
matplotlib.use("Agg")
simulate_other_machine = len(sys.argv) > 1
nb = nbformat.read("/Users/t/Downloads/zollerqu_mesh_workspace/mesh_idea_ZollerQu.ipynb", as_version=4)
code = [c.source for c in nb.cells if c.cell_type == "code"]
ns = {}
t0 = time.time()
def run(src):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(src, ns)
    return buf.getvalue()
for src in code:
    if src.startswith("# TASK_01_IMPLEMENTATION"): continue
    if src.startswith("# TASK_02_IMPLEMENTATION"):
        if simulate_other_machine:      # pretend this machine built case 2 slightly differently
            m = ns["mesh_cases"]["2 main"]; pts = m["points"].copy(); pts[5] += 1e-9
            ns["mesh_cases"]["2 main"] = ns["add_case_weights"](ns["MESH_CASES"]["2 main"], ns["check_case_mesh"]("2 main", ns["MESH_CASES"]["2 main"], pts, m["triangles"]))
            print("simulated a different mesh for case 2:", ns["mesh_cases"]["2 main"]["fingerprint"])
        print(run(open("../task01d_cell.py").read()).strip())
        reference_fp = {k: v["fingerprint"] for k, v in ns["mesh_cases"].items()}
        run(src)
        print(run(open("../task02b_cell.py").read()).strip())
        continue
    if src.startswith("# TASK_04"): break
    run(src)
print(f"[{time.time() - t0:.0f} s]")
# compare with the regions of the committed notebook run (seed 7, n = 12)
import gzip, json
ck = json.load(gzip.open("/Users/t/Downloads/zollerqu_mesh_workspace/results/task04_checkpoint_small.json.gz", "rt"))["instances"]
exec(open("../task04_cell.py").read().split("# ------------------------------------------------- calibration of LR-QAOA ----")[0].replace('assert RUN["quantum_cases"]', 'pass  #'), ns)
same = 0
for key, record in ck.items():
    case, n, seed, scen = key.split("|")
    inst = ns["t4_build_instance"](ns["region_cases"][(case, int(n), int(seed))], case, int(n), int(seed), scen)
    same += inst["fingerprint"] == record["fingerprint"]
print(f"instances identical to the committed Task 04 run: {same}/{len(ck)}")
