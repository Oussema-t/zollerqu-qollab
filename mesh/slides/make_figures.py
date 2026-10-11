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

(FIG / "numbers.json").write_text(json.dumps(dict(numbers=numbers, figure_sources=SOURCES), indent=1))
print(f"{len(SOURCES)} figures and numbers.json written to {FIG}")
for key in ("ionq_p_opt", "noiseless_p_opt", "random_p_opt", "gain_over_random", "ionq_shots_per_optimum", "zy_training_circuits", "gpu_share_of_optimum", "n_candidates"):
    print(f"  {key} = {numbers[key]}")
print("  q29:", numbers["q29"])
