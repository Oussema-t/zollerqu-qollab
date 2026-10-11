"""Figures and numbers for the mesh video deck: slides/figures/*.png and slides/figures/numbers.json.
Every number comes from a result file in this repo (listed per figure in SOURCES); nothing is typed by hand.
usage: python slides/make_figures.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "slides" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
# calm palette in the direction of the IonQ and Qollab sites (near-black, off-white, neutral greys, one soft orange); no logos
# ACCENT = soft blue for labels and neutral data; HIGHLIGHT = soft orange, used ONLY for our IonQ noise-model result
BG, PANEL, WHITE, MUTED, ACCENT, HIGHLIGHT, GREY = "#0F161B", "#1A242B", "#FAFAFA", "#9AA3AA", "#9FB4D1", "#F89B33", "#5F6B75"
GPU_FILL, CPU_FILL = "#4E6E96", "#C9D1D9"
plt.rcParams.update({"font.family": "Arial", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": WHITE, "axes.labelcolor": WHITE, "xtick.color": WHITE, "ytick.color": WHITE, "axes.edgecolor": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2, "grid.color": "#26323B",
                     "legend.frameon": False, "mathtext.fontset": "cm"})
SOURCES = {}

# ---------------------------------------------------------------- numbers from the result files ----
selection = json.loads((ROOT / "qollab" / "selected_circuits.json").read_text())
run_path = ROOT / "qollab" / "results_ionq" / "2026-10-11_varqite_zy_equal_ionq_simulator.json"
run = json.loads(run_path.read_text())
equal, cpu_gpu = selection["instances"]["2 equal devices"], selection["instances"]["CPU + GPU (GPU 4x)"]
zy = next(c for c in selection["circuits"] if c["name"].startswith("varQITE ZY"))
cvar = next(c for c in selection["circuits"] if c["name"].startswith("CVaR"))
lemma = selection["extras"][0]
summary = run["summary"][0]
counts = sorted(run["counts"].values(), reverse=True)
gates = json.loads((ROOT / "results" / "ionq_selection" / "heavy_hex_gate_counts.json").read_text())["rows"]
validation = json.loads((ROOT / "results" / "hpc_validation" / "hpc_n12" / "2_main_seed7_equal.json").read_text())
candidates = list(csv.DictReader(open(ROOT / "results" / "ionq_selection" / "candidates_table.csv")))
scaling = json.loads((ROOT / "report" / "scaling.json").read_text())
q29 = list(csv.DictReader(open(ROOT / "results" / "hpc_n29" / "summary_q29.csv")))


def q29_hit(scenario, method_prefix):
    values = [float(r["hit_near_1000"]) for r in q29 if r["scenario"] == scenario and r["method"].startswith(method_prefix)]
    assert values, (scenario, method_prefix)
    return float(np.mean(values))


numbers = dict(
    ionq_p_opt=summary["p_opt"], ionq_shots=summary["shots"], ionq_optimal_shots=round(summary["p_opt"] * summary["shots"]),
    ionq_standard_error=run["comparison"]["binomial_standard_error"], ionq_p_balanced=summary["p_balanced"],
    ionq_most_frequent_is_optimum=summary["most_frequent_is_optimum"], ionq_distinct_splits=summary["distinct_splits"],
    ionq_shots_per_optimum=summary["shots"] / round(summary["p_opt"] * summary["shots"]), ionq_date=run["time"][:10],
    noiseless_p_opt=zy["predicted"]["p_opt"], crude_estimate_p_opt=zy["predicted"]["crude_p_opt"], aer_estimate_p_opt=zy["predicted"]["aer_p_opt"],
    noiseless_shots_per_optimum=1 / zy["predicted"]["p_opt"],
    random_p_opt=equal["uniform_p_opt"], random_tries_per_optimum=1 / equal["uniform_p_opt"], gain_over_random=summary["p_opt"] / equal["uniform_p_opt"],
    zy_two_qubit_gates=zy["gates"]["two_qubit"], zy_training_circuits=zy["training_done_offline_circuits"],
    annealing_median_evaluations=equal["classical_samples_to_solution"]["simulated annealing"]["median"],
    local_search_median_evaluations=equal["classical_samples_to_solution"]["local search (restarts)"]["median"],
    annealing_hit_1000=validation["classical"]["simulated annealing"]["hit_near"], local_search_hit_1000=validation["classical"]["local search (restarts)"]["hit_near"],
    n_splits=2 ** selection["n"], n_candidates=len(candidates),
    cpu_gpu_random_p_opt=cpu_gpu["uniform_p_opt"], cpu_gpu_best_quantum=cvar["name"], cpu_gpu_best_quantum_p_opt=cvar["predicted"]["p_opt"],
    cpu_gpu_best_quantum_gates=cvar["gates"]["two_qubit"], cpu_gpu_varqite_zy_p_opt=lemma["predicted"]["p_opt"],
    gates_12=dict(all_to_all=gates[0]["all_to_all_two_qubit_gates"], heavy_hex=gates[0]["heavy_hex_two_qubit_gates"]["median"], ratio=gates[0]["ratio_median"]),
    gates_29=dict(all_to_all=gates[1]["all_to_all_two_qubit_gates"], heavy_hex=gates[1]["heavy_hex_two_qubit_gates"]["median"], ratio=gates[1]["ratio_median"]),
    gates_zy=dict(all_to_all=gates[2]["all_to_all_two_qubit_gates"], heavy_hex=gates[2]["heavy_hex_two_qubit_gates"]["median"], ratio=gates[2]["ratio_median"]),
    q29=dict(equal=dict(qaoa=q29_hit("2 equal devices", "LR-QAOA (paper default"), local=q29_hit("2 equal devices", "Local search"),
                        annealing=q29_hit("2 equal devices", "Simulated annealing"), spectral=q29_hit("2 equal devices", "Spectral")),
             cpu_gpu=dict(qaoa=q29_hit("CPU + GPU (GPU 4x)", "LR-QAOA (success-calibrated"), local=q29_hit("CPU + GPU (GPU 4x)", "Local search"),
                          annealing=q29_hit("CPU + GPU (GPU 4x)", "Simulated annealing"), spectral=q29_hit("CPU + GPU (GPU 4x)", "Spectral")),
             instances=len({(r["case"], r["seed"], r["scenario"]) for r in q29})))

# ---------------------------------------------------------------- 1. the problem: one mesh, two devices ----
mesh = np.load(ROOT / "data" / "frozen_meshes" / "2_main_seed7.npz")
points, triangles = mesh["points"], mesh["triangles"]
fingerprint = hashlib.sha256(np.ascontiguousarray(points).tobytes() + np.ascontiguousarray(triangles).tobytes()).hexdigest()[:12]
labels = np.load(ROOT / "data" / "frozen_regions.npz")[f"{fingerprint}|12|7"]
state = cpu_gpu["optimal_states"][0]
on_gpu = ((state >> labels) & 1).astype(bool)                      # bit 1 = device B = GPU
work = np.asarray(cpu_gpu["work"])
assert np.array_equal(np.bincount(labels, minlength=12), work.astype(int)), "region work differs from the instance"
gpu_share = float(work[[(state >> r) & 1 == 1 for r in range(12)]].sum() / work.sum())
numbers.update(triangles=int(len(triangles)), gpu_share_of_optimum=gpu_share, target_share=cpu_gpu["alpha"])
edge_owner, cut = {}, []
for t, tri in enumerate(triangles):
    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        key = (min(a, b), max(a, b))
        if key in edge_owner and on_gpu[edge_owner[key]] != on_gpu[t]:
            cut.append(points[[key[0], key[1]]])
        edge_owner[key] = t
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.add_collection(PolyCollection(points[triangles][on_gpu], facecolors=GPU_FILL, edgecolors=BG, linewidths=0.25))
ax.add_collection(PolyCollection(points[triangles][~on_gpu], facecolors=CPU_FILL, edgecolors=BG, linewidths=0.25))
ax.add_collection(LineCollection(cut, colors=WHITE, linewidths=4, capstyle="round"))
ax.text(0.47, 1.52, f"GPU\n{100 * gpu_share:.0f} % of the work", color=WHITE, fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(1.62, 0.5, f"CPU\n{100 * (1 - gpu_share):.0f} %", color=BG, fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(1.5, 1.52, "border =\ndata exchanged\nat every step", color=WHITE, fontsize=18, ha="center", va="center")
ax.set_xlim(-0.03, 2.03)
ax.set_ylim(-0.03, 2.03)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.1)
fig.savefig(FIG / "fig_problem.png", dpi=220)
plt.close(fig)
SOURCES["fig_problem.png"] = "data/frozen_meshes/2_main_seed7.npz, data/frozen_regions.npz, qollab/selected_circuits.json (exact optimum for CPU + GPU)"

# ---------------------------------------------------------------- 2. the formula ----
fig = plt.figure(figsize=(11.6, 2.3))
fig.text(0.5, 0.66, r"$E(z)\;=\;\sum_{\langle i,j\rangle} c_{ij}\,\frac{1-z_i z_j}{2}\;\;+\;\;\lambda\,\left(\sum_i a_i z_i-(1-2\alpha)\right)^{2}$",
         ha="center", va="center", fontsize=34, color=WHITE)
fig.text(0.31, 0.13, "border that is cut", ha="center", va="center", fontsize=20, color=ACCENT)
fig.text(0.72, 0.13, "penalty if the work share is off target", ha="center", va="center", fontsize=20, color=ACCENT)
fig.savefig(FIG / "fig_formula.png", dpi=220)
plt.close(fig)
SOURCES["fig_formula.png"] = "the Ising energy used in the notebook (Task 03) and in qollab/main.py"

# ---------------------------------------------------------------- 3. circuit sketch ----
fig, ax = plt.subplots(figsize=(7.0, 4.3))
wires = [3.2, 2.4, 1.6, 0.5]
names = ["region 1", "region 2", "region 3", "region 12"]
for y, name in zip(wires, names):
    ax.plot([0.9, 9.3], [y, y], color=MUTED, lw=1.5, zorder=1)
    ax.text(0.8, y, name, ha="right", va="center", fontsize=18, color=WHITE)
ax.text(0.35, 1.05, "...", ha="center", va="center", fontsize=24, color=MUTED)


def box(x, y, text, colour):
    ax.add_patch(FancyBboxPatch((x - 0.32, y - 0.3), 0.64, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=PANEL, edgecolor=colour, lw=2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=18, color=WHITE, zorder=4)


for y in wires:
    box(1.5, y, "H", MUTED)
    box(7.3, y, "rx", MUTED)
    box(8.9, y, "M", MUTED)
for x, (a, b) in zip((2.9, 3.7, 4.5, 5.3, 6.1), ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3))):
    ax.plot([x, x], [wires[a], wires[b]], color=ACCENT, lw=2.5, zorder=2)
    ax.scatter([x, x], [wires[a], wires[b]], s=90, color=ACCENT, zorder=3)
ax.add_patch(Rectangle((2.45, 0.05), 5.3, 3.6, fill=False, edgecolor=MUTED, lw=1.2, ls=(0, (4, 3))))
ax.text(4.5, 3.95, "cost layer: one rzz gate per pair", ha="center", va="center", fontsize=18, color=ACCENT)
ax.text(7.3, -0.35, "mixer", ha="center", va="center", fontsize=18, color=MUTED)
ax.text(1.5, -0.35, "start", ha="center", va="center", fontsize=18, color=MUTED)
ax.text(8.9, -0.35, "measure", ha="center", va="center", fontsize=18, color=MUTED)
ax.text(4.5, -0.35, "repeat p times", ha="center", va="center", fontsize=18, color=MUTED)
ax.set_xlim(-1.3, 9.6)
ax.set_ylim(-0.7, 4.3)
ax.axis("off")
fig.tight_layout(pad=0.1)
fig.savefig(FIG / "fig_circuit.png", dpi=220)
plt.close(fig)
SOURCES["fig_circuit.png"] = "sketch of the QAOA circuit built by qollab/main.py (build_circuit)"

# ---------------------------------------------------------------- 4. gate counts ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
groups = ["QAOA layer\n12 qubits", "varQITE ZY\n12 qubits", "QAOA layer\n29 qubits"]
ordered = [gates[0], gates[2], gates[1]]                         # dense 12, varQITE ZY 12, dense 29 (the order of the labels)
assert "12 qubits" in ordered[0]["circuit"] and "varQITE" in ordered[1]["circuit"] and "29 qubits" in ordered[2]["circuit"]
all_to_all = [g["all_to_all_two_qubit_gates"] for g in ordered]
heavy = [g["heavy_hex_two_qubit_gates"]["median"] for g in ordered]
x = np.arange(3)
ax.bar(x - 0.2, all_to_all, 0.4, color=ACCENT, label="all-to-all (IonQ)")
ax.bar(x + 0.2, heavy, 0.4, color=GREY, hatch="//", edgecolor=BG, label="routed on heavy-hex")
for xi, a, h in zip(x, all_to_all, heavy):
    ax.text(xi - 0.2, a + 25, f"{a:,}", ha="center", fontsize=18, color=WHITE, fontweight="bold")
    ax.text(xi + 0.2, h + 25, f"{h:,.0f}", ha="center", fontsize=18, color=WHITE)
ax.set_xticks(x, groups)
ax.set_ylabel("two-qubit gates (count)")
ax.set_ylim(0, max(heavy) * 1.18)
ax.legend(loc="upper left", fontsize=18)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_gates.png", dpi=220)
plt.close(fig)
SOURCES["fig_gates.png"] = "results/ionq_selection/heavy_hex_gate_counts.json (scripts/heavy_hex_gate_counts.py, transpiler only)"

# ---------------------------------------------------------------- 5. the IonQ result ----
fig, ax = plt.subplots(figsize=(6.3, 4.7))
values = [100 * numbers["random_p_opt"], 100 * numbers["ionq_p_opt"], 100 * numbers["noiseless_p_opt"]]
bars = ax.bar(["random\nguessing", "IonQ Forte 1\nnoise model", "noiseless\nsimulation"], values, color=[GREY, HIGHLIGHT, ACCENT], width=0.6)
bars[0].set_hatch("//")
bars[0].set_edgecolor(BG)
ax.errorbar([1], [values[1]], yerr=[100 * numbers["ionq_standard_error"]], color=WHITE, capsize=6, lw=2)
ax.text(0, values[0] * 1.3, f"{values[0]:.3f} %", ha="center", fontsize=22, fontweight="bold", color=WHITE)
ax.text(1, values[1] * 0.42, f"{values[1]:.1f} %", ha="center", fontsize=24, fontweight="bold", color=BG)
ax.text(2, values[2] * 0.42, f"{values[2]:.1f} %", ha="center", fontsize=24, fontweight="bold", color=BG)
ax.plot([0.66, 1.34], [100 * numbers["crude_estimate_p_opt"]] * 2, color=WHITE, lw=2.5, ls=(0, (4, 3)))
ax.text(0, 100 * numbers["crude_estimate_p_opt"] * 0.62, f"our estimate:\n{100 * numbers['crude_estimate_p_opt']:.0f} %", va="center", ha="center", fontsize=18, color=MUTED)
ax.set_yscale("log")
ax.set_ylim(0.01, 400)
ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
ax.set_ylabel("shots on the exact best split (%)")
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_ionq_result.png", dpi=220)
plt.close(fig)
SOURCES["fig_ionq_result.png"] = ("qollab/results_ionq/2026-10-11_varqite_zy_equal_ionq_simulator.json (IonQ Forte 1 noise model on Qollab), "
                                  "qollab/selected_circuits.json (noiseless value, estimate, random guessing)")

# ---------------------------------------------------------------- 6. scaling ----
sizes = [12, 16, 20, 29]
fig, axes = plt.subplots(1, 2, figsize=(7.7, 4.6), sharey=True)
styles = [("Spectral bisection", "spectral bisection", WHITE, "s", (0, (5, 3))), ("Local search", "local search", MUTED, "o", "-"),
          ("Simulated annealing", "simulated annealing", GREY, "^", (0, (1, 2))), ("QAOA+COBYLA $p=2$", "QAOA (noiseless)", ACCENT, "D", "-")]
keys29 = {"Spectral bisection": "spectral", "Local search": "local", "Simulated annealing": "annealing", "QAOA+COBYLA $p=2$": "qaoa"}
scaling_table = {}
for ax, (scenario, tag, title) in zip(axes, (("2 equal devices", "equal", "2 equal devices"), ("CPU + GPU (GPU 4x)", "cpu_gpu", "CPU + GPU"))):
    for key, label, colour, marker, style in styles:
        series = [100 * v for v in scaling[scenario][key][:3]] + [100 * numbers["q29"][tag][keys29[key]]]
        scaling_table.setdefault(scenario, {})[label] = series
        ax.plot(range(4), series, color=colour, marker=marker, ls=style, lw=3 if key.startswith("QAOA") else 2, ms=9, label=label)
    ax.set_xticks(range(4), [str(s) for s in sizes])
    ax.set_xlabel("qubits (= mesh regions)")
    ax.set_title(title, fontsize=18, color=WHITE)
    ax.set_ylim(-4, 104)
    ax.grid(axis="y", alpha=0.6)
axes[0].set_ylabel("runs of 1,000 tries that find\na near-best split (%)")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=18, handlelength=2.6, columnspacing=1.5)
fig.tight_layout(pad=0.3, rect=(0, 0.2, 1, 1))
fig.savefig(FIG / "fig_scaling.png", dpi=220)
plt.close(fig)
numbers["scaling_percent"] = dict(sizes=sizes, series=scaling_table)
SOURCES["fig_scaling.png"] = ("report/scaling.json (12, 16, 20 qubits: results_large_hpc/, server simulation 2026-10-10) and "
                              "results/hpc_n29/summary_q29.csv (29 qubits, server simulation 2026-10-10); QAOA at 29 qubits = best fixed schedule p = 5")

# ---------------------------------------------------------------- 7. samples to solution ----
fig, ax = plt.subplots(figsize=(4.7, 3.3))
names = ["random guessing", "simulated annealing", "local search", "IonQ noise model", "noiseless"]
tries = [numbers["random_tries_per_optimum"], numbers["annealing_median_evaluations"], numbers["local_search_median_evaluations"],
         numbers["ionq_shots_per_optimum"], numbers["noiseless_shots_per_optimum"]]
colours = [GREY, GREY, GREY, HIGHLIGHT, ACCENT]
bars = ax.barh(names[::-1], tries[::-1], color=colours[::-1], height=0.62)
for bar, value in zip(bars, tries[::-1]):
    ax.text(value * 1.25, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}" if value >= 10 else f"{value:.1f}", va="center", fontsize=18, color=WHITE, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(0.8, 60000)
ax.set_xticks([1, 100, 10000], ["1", "100", "10,000"])
ax.minorticks_off()
ax.set_xlabel("tries per optimum")
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_samples.png", dpi=220)
plt.close(fig)
SOURCES["fig_samples.png"] = ("qollab/results_ionq/2026-10-11_varqite_zy_equal_ionq_simulator.json (shots per optimum), qollab/selected_circuits.json "
                              "(classical median evaluations from results/ionq_selection/, noiseless value)")

# ---------------------------------------------------------------- 8. Qollab screenshot placeholder (recorded output) ----
fig = plt.figure(figsize=(5.75, 3.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.add_patch(FancyBboxPatch((0.015, 0.025), 0.97, 0.95, boxstyle="round,pad=0.0,rounding_size=0.03", facecolor="#060C18", edgecolor=MUTED, lw=1.5, transform=ax.transAxes))
lines = [("Qollab > Run > IonQ Forte 1 (29q)", ACCENT, True),
         ("Running on: ionq_simulator", WHITE, False),
         (f"varqite_zy_equal x {numbers['ionq_shots']} shots", WHITE, False),
         (f"P(optimum) measured {numbers['ionq_p_opt']:.4f}", HIGHLIGHT, True),
         (f"x{counts[0]}  OPTIMUM  111100110000", WHITE, False),
         (f"x{counts[1]}  OPTIMUM  000011001111", WHITE, False),
         ("[pending] live screenshot", MUTED, False)]
for index, (text, colour, bold) in enumerate(lines):
    ax.text(0.06, 0.88 - index * 0.128, text, transform=ax.transAxes, fontsize=18, color=colour, family="DejaVu Sans Mono", fontweight="bold" if bold else "normal", va="center")
ax.axis("off")
fig.savefig(FIG / "fig_qollab_placeholder.png", dpi=220)
plt.close(fig)
numbers["top_counts"] = counts[:2]
SOURCES["fig_qollab_placeholder.png"] = "qollab/results_ionq/2026-10-11_varqite_zy_equal_qollab_console.html (recorded console output); to be replaced by a live screenshot"

# ---------------------------------------------------------------- 9. the nine mesh objects (frozen seed-7 meshes) ----
import importlib.util  # noqa: E402
import os  # noqa: E402

_here = os.getcwd()
_spec = importlib.util.spec_from_file_location("make_mesh_figure", ROOT / "report" / "make_mesh_figure.py")
_mesh_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mesh_mod)                       # (it changes into the repo root; restored below)
mesh_ns = _mesh_mod.load_mesh_cells()
os.chdir(_here)
case_names = list(mesh_ns["BENCHMARK_CASES"])
fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6))
for ax, name in zip(axes.flat, case_names):
    spec, mesh = mesh_ns["MESH_CASES"][name], mesh_ns["mesh_cases"][name]
    ax.add_collection(PolyCollection(mesh["points"][mesh["triangles"]], facecolors=PANEL, edgecolors=ACCENT, linewidths=0.15, rasterized=True))
    for zone in spec.get("work_zones", []):
        x0, y0, x1, y1 = zone["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GPU_FILL, alpha=0.55, edgecolor="none"))
    for zone in spec.get("cost_zones", []):
        x0, y0, x1, y1 = zone["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=WHITE, hatch="...", lw=0.6))
    for zone in spec.get("priority_zones", []):
        ax.add_patch(plt.Circle(zone["center"], zone["radius"], fill=False, edgecolor=WHITE, lw=1.4))
    for feature in spec["features"]:
        if feature["kind"] == "heat source":
            ax.scatter(*feature["center"], s=30 + 90 * feature["strength"], marker="*", c=WHITE, edgecolors=BG, linewidths=0.5, zorder=5)
    lo, hi = spec["hull"].min(axis=0), spec["hull"].max(axis=0)
    ax.set_xlim(lo[0] - 0.05, hi[0] + 0.05)
    ax.set_ylim(lo[1] - 0.05, hi[1] + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{name}\n{len(mesh['triangles']):,} triangles", fontsize=12, color=WHITE)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(FIG / "fig_meshes.png", dpi=220)
plt.close(fig)
numbers.update(n_mesh_objects=len(case_names), mesh_case_names=case_names,
               mesh_triangles={name: int(len(mesh_ns["mesh_cases"][name]["triangles"])) for name in case_names},
               n_distinct_shapes=len({mesh_ns["MESH_CASES"][name].get("mesh_of", name) for name in case_names}))
SOURCES["fig_meshes.png"] = "data/frozen_meshes/*_seed7.npz via the notebook mesh cells (Tasks 01-01d); same objects as report/fig_meshes.pdf"

# ---------------------------------------------------------------- 10. how far QAOA itself improves (HPC tuning study) ----
tuning = json.loads((ROOT / "results" / "qaoa_tuning" / "tuning_n12_large_calibration.json").read_text())
scaling_files = [json.loads(path.read_text()) for path in sorted((ROOT / "results" / "qaoa_tuning" / "scaling").glob("n*_p*.json"))]
earlier = json.loads((ROOT / "scripts" / "q29_choice.json").read_text())["choice"]


def tuned12(scenario, p, kind="standard"):
    rows = [r for r in tuning["scenarios"][scenario]["rows"] if r["p"] == p]
    if kind == "LR":
        return next(r for r in rows if r["method"] == "LR-QAOA (energy-calibrated)")["test"]
    pool = [r for r in rows if ("warm" in r["method"]) == (kind == "warm")]
    return max(pool, key=lambda r: r["calibration_p_near"])["test"]


def tuned_n(scenario, n, p):
    rec = next((r for r in scaling_files if r["scenario"] == scenario and r["n"] == n and r["p"] == p), None)
    return None if rec is None else rec["methods"][f"re-optimised at n = {n}"]["test_mean"]


SC = {"2 equal devices": "equal devices", "CPU + GPU (GPU 4x)": "CPU + GPU"}
P_SHOW = 6                                                # the deepest depth finished at every size
fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4))
ax = axes[0]
for scenario, line in zip(SC, ("-", "--")):
    ax.plot(tuning["layers"], [100 * tuned12(scenario, p)["p_near"] for p in tuning["layers"]], color=ACCENT, lw=3, ls=line, marker="o",
            label=f"tuned QAOA, {SC[scenario]}")
    ax.plot(tuning["layers"], [100 * tuned12(scenario, p, "LR")["p_near"] for p in tuning["layers"]], color=GREY, lw=2, ls=line, marker="s", ms=5,
            label=f"LR-QAOA, {SC[scenario]}")
    old = tuning["scenarios"][scenario]["references"]["per-shot QAOA + COBYLA p=2 (existing)"]
    ax.scatter([2], [100 * old], marker="x", s=140, color=WHITE, zorder=6, label="before: trained per mesh, p = 2" if line == "-" else None)
ax.set_yscale("log")
ax.set_xticks(tuning["layers"])
ax.set_xlabel("QAOA layers p")
ax.set_ylabel("near-best splits per shot (%)")
ax.set_title("12 qubits, 9 test meshes", loc="left", fontsize=18, color=WHITE)
ax.legend(fontsize=14, loc="lower right")
ax.grid(alpha=0.6, which="major")
ax = axes[1]
for scenario, line in zip(SC, ("-", "--")):
    sizes = [12] + [n for n in (16, 20) if tuned_n(scenario, n, P_SHOW)]
    vals = [100 * tuned12(scenario, P_SHOW)["p_near"]] + [100 * tuned_n(scenario, n, P_SHOW)["p_near"] for n in sizes[1:]]
    ax.plot(sizes, vals, color=ACCENT, lw=3, ls=line, marker="o", label=f"tuned QAOA p = {P_SHOW}, {SC[scenario]}")
    ax.scatter([20], [100 * earlier[scenario]["best_fixed"]["p_near_n20"]], marker="x", s=140, color=WHITE, zorder=6,
               label="before: best setting at 20 qubits (p = 5)" if line == "-" else None)
    uni = [np.mean([row["per-shot random (uniform)"]["p_near"] for row in r["classical"]]) for n in sizes[1:]
           for r in scaling_files if r["scenario"] == scenario and r["n"] == n and r["p"] == P_SHOW]
    if line == "-":
        ax.plot(sizes[1:], [100 * u for u in uni], color=MUTED, lw=1.5, ls=":", label="random guess (equal devices)")
ax.set_yscale("log")
ax.set_xticks([12, 16, 20])
ax.set_xlabel("qubits (mesh regions)")
ax.set_title("bigger meshes (angles re-tuned on HPC)", loc="left", fontsize=18, color=WHITE)
ax.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
ax.grid(alpha=0.6, which="major")
fig.tight_layout()
fig.savefig(FIG / "fig_tuning.png", dpi=220)
plt.close(fig)
numbers["tuning"] = dict(
    n12={SC[s]: dict(p8=tuned12(s, 8)["p_near"], p8_opt=tuned12(s, 8)["p_opt"], p3_hit=tuned12(s, 3)["hit_near"], p8_hit=tuned12(s, 8)["hit_near"],
                     before=tuning["scenarios"][s]["references"]["per-shot QAOA + COBYLA p=2 (existing)"],
                     warm_p8=tuned12(s, 8, "warm")["p_near"], warm_p8_hit=tuned12(s, 8, "warm")["hit_near"]) for s in SC},
    n20={SC[s]: dict(p=P_SHOW, p_near=tuned_n(s, 20, P_SHOW)["p_near"], hit=tuned_n(s, 20, P_SHOW)["hit_near"],
                     p5=tuned_n(s, 20, 5)["p_near"], before_p5=earlier[s]["best_fixed"]["p_near_n20"]) for s in SC if tuned_n(s, 20, P_SHOW)},
    gates12_p8=8 * 66, calibration_meshes=18, test_meshes_12=len(tuning["scenarios"]["2 equal devices"]["test_instances"]))
SOURCES["fig_tuning.png"] = ("results/qaoa_tuning/tuning_n12_large_calibration.json (12 qubits, HPC), results/qaoa_tuning/scaling/n16_*, n20_* "
                             "(HPC), scripts/q29_choice.json (the earlier best setting at 20 qubits); noiseless simulation")

(FIG / "numbers.json").write_text(json.dumps(dict(numbers=numbers, figure_sources=SOURCES), indent=1))
print(f"{len(SOURCES)} figures and numbers.json written to {FIG}")
for key in ("ionq_p_opt", "noiseless_p_opt", "random_p_opt", "gain_over_random", "ionq_shots_per_optimum", "zy_training_circuits", "gpu_share_of_optimum", "n_candidates"):
    print(f"  {key} = {numbers[key]}")
print("  q29:", numbers["q29"])
