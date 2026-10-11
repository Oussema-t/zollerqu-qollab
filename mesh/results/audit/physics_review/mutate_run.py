"""Mutation test of the cell's own sign tests. Usage: mutate_run.py <mutation> [fullgrid]
Runs parts 1-4 (config, defs, calibration, benchmark on 2 cases, tests) of a MUTATED copy of the cell (never the original file)."""
import sys, time, os, traceback
import numpy as np
from load_nb import load
mutation = sys.argv[1]
fullgrid = len(sys.argv) > 2 and sys.argv[2] == "fullgrid"
os.makedirs("results", exist_ok=True)
code = "".join(open(f"task04_part{i}.py").read() for i in (1, 2, 3, 4))

def rep(src, old, new, count=1):
    assert src.count(old) == count, (old, src.count(old))
    return src.replace(old, new)

if mutation in ("cost_both", "cost_numpy"):
    code = rep(code, "state = state * (np.cos(phase) - 1j * np.sin(phase))", "state = state * (np.cos(phase) + 1j * np.sin(phase))")
if mutation in ("cost_both", "cost_qiskit"):
    code = rep(code, "circuit.rzz(float(2 * gamma * J_scaled[i, j]), i, j)", "circuit.rzz(float(-2 * gamma * J_scaled[i, j]), i, j)")
    code = rep(code, "circuit.rz(float(2 * gamma * h_scaled[q]), q)", "circuit.rz(float(-2 * gamma * h_scaled[q]), q)")
if mutation == "mixer_both":
    code = rep(code, "return (hadamard * np.exp(1j * beta * weights)) @ hadamard", "return (hadamard * np.exp(-1j * beta * weights)) @ hadamard")
    code = rep(code, "c, s = np.cos(beta), 1j * np.sin(beta)", "c, s = np.cos(beta), -1j * np.sin(beta)")
    code = rep(code, "circuit.rx(float(-2 * beta), q)", "circuit.rx(float(2 * beta), q)")
    code = rep(code, "circuit.rz(float(-2 * beta), q)", "circuit.rz(float(2 * beta), q)")
assert mutation in ("none", "cost_both", "cost_numpy", "cost_qiskit", "mixer_both")
ns = load("small", upto=7)
ns["RUN"]["quantum_cases"] = ["1 control", "2 main", "9 priority"]
ov = dict(eval_budget=40, random_starts=1, classical_repeats=5, use_checkpoint=False, sa_calibration_repeats=3)
if not fullgrid:
    ov.update(lr_grid_gamma=np.linspace(0.05, 1.5, 12), lr_grid_beta=np.linspace(0.1, 1.2, 6), calibration_seeds=[101])
ns["TASK04_OVERRIDES"] = ov
t0 = time.time()
try:
    exec(compile(code, "task04_cell_mutated", "exec"), ns)
    print(f"[{mutation}] CELL TESTS COMPLETED WITHOUT ASSERTION ({time.time() - t0:.0f} s)")
except AssertionError as error:
    tb = traceback.extract_tb(sys.exc_info()[2])[-1]
    print(f"[{mutation}] AssertionError: {error} (line {tb.lineno}: {code.splitlines()[tb.lineno - 1].strip()[:150]})")
except Exception as error:
    print(f"[{mutation}] {type(error).__name__}: {error}")
if "lr_calibration" in ns:
    for scen, cal in ns["lr_calibration"].items():
        print(f"[{mutation}] LR calibration {scen}: energy {cal['energy']} score {cal['energy_score']:.4f}; success {cal['success']} score {cal['success_score']:.4f}; edge {cal['at_grid_edge']}")
for name, result in ns.get("t4_tests", {}).items():
    print(f"[{mutation}] TEST {name}: {result}")
for c in ns.get("qiskit_checks", []):
    print(f"[{mutation}] QISKIT {c['scenario']}: err {c['max_probability_error']:.1e} mean_cost {c['mean_cost']:.4f} uniform {c['uniform_mean_cost']:.4f} wrong-sign {c['wrong_sign_mean_cost']:.4f}")
if ns.get("task04_failures"):
    print(f"[{mutation}] recorded failures:", ns["task04_failures"][:3])
# independent ground truth: with the calibrated LR angles, what does the (possibly mutated) simulator give vs uniform, per example instance
if "t4_examples" in ns:
    for scen, inst in ns["t4_examples"].items():
        view = ns["t4_view"](inst)
        for p in ns["T4_LAYERS"]:
            ang = ns["lr_angles"](p, *ns["T4_LR_DELTAS"][scen]["LR-QAOA"])
            pr = ns["view_probabilities"](view, ang)
            print(f"[{mutation}] example {scen} p={p}: <E>={pr @ inst['energies']:.4f} (uniform {inst['e_mean']:.4f}), p_near={pr[inst['near_mask']].sum():.5f} (uniform {inst['near_mask'].mean():.5f})")
