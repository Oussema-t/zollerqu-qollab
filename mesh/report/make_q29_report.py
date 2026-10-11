"""29-qubit HPC results -> paper: report/table_q29.tex (\\TableQnine), report/numbers_q29.tex, report/fig_q29_split.pdf.
Reads results/hpc_n29/*.json (written by scripts/run_29_qubits.py) and the frozen region assignments.
usage: python report/make_q29_report.py
"""
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
N = 29
SCENARIOS = {"equal": "2 equal devices", "cpu_gpu": "CPU + GPU (GPU 4x)"}
CLASSICAL = [("Random, uniform", "uniform random sampling", "uniform_p_near"),
             ("Random, target share", "random with target share", "target_share_p_near"),
             ("Simulated annealing", "simulated annealing", None),
             ("Local search", "local search (restarts)", None),
             ("Spectral bisection", "spectral bisection", None)]
records = [json.loads(path.read_text()) for path in sorted((ROOT / "results" / f"hpc_n{N}").glob("*_seed*_*.json"))]   # instance files only
assert records, "no 29-qubit results yet"
by_scenario = {tag: [r for r in records if r["scenario"] == name] for tag, name in SCENARIOS.items()}
roles = ["best fixed schedule", "transferred trained angles", "continuity"]


def mean(values):
    return float(np.mean(list(values)))


def pct(x, digits=None):
    x = 100 * x
    if x == 0:
        return "0"
    if digits is not None:
        return f"{x:.{digits}f}"
    if x < 0.001:                                             # tiny percentages in LaTeX math, one significant digit
        mantissa, exponent = f"{x:.0e}".split("e")
        return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}" if x >= 0.95 else f"{x:.2g}"


def sci(x):
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}{{\\times}}10^{{{int(exponent)}}}"            # math-mode content (no dollars)


def gap_text(values):
    g = mean(values)
    assert g > -1e-6, "a best-of-1000 cost below the exact optimum"
    return "0" if g < 0.5 else f"{g:.0f}"


def thin(n):
    return f"{n:,}".replace(",", "\\,")


def quantum(scen, role):
    return [next(q for q in r["quantum"] if q["role"] == role) for r in by_scenario[scen]]


# ---------------------------------------------------------------- table ----
rows = []
for label, method, per_shot_key in CLASSICAL:
    cells = []
    for scen in SCENARIOS:
        rs = by_scenario[scen]
        per_shot = f"${sci(mean(r['per_shot_random'][per_shot_key] for r in rs))}$" if per_shot_key else "--"
        if per_shot_key:                                       # random samplers: exact, like QAOA (not a 100-run count)
            hit = pct(mean(1 - (1 - r["per_shot_random"][per_shot_key]) ** 1000 for r in rs))
        else:
            hit = pct(mean(r["classical"][method]["hit_near"] for r in rs))
        gap = gap_text(100 * (r["classical"][method]["mean_best_cost"] / r["e_min"] - 1) for r in rs)
        cells += [per_shot, hit, gap]
    evaluations = by_scenario["equal"][0]["classical"][method]["evaluations"]
    rows.append(f"{label} & -- & {thin(evaluations)} & " + " & ".join(cells) + r" \\")
rows.append(r"\midrule")
names = {"best fixed schedule": "Best fixed schedule$^a$, $p{=}5$", "transferred trained angles": "QAOA+COBYLA angles$^b$, $p{=}2$",
         "continuity": "LR-QAOA $p{=}2$"}
for role in roles:
    cells = []
    for scen in SCENARIOS:
        qs, rs = quantum(scen, role), by_scenario[scen]
        cells += [f"${sci(mean(q['p_near'] for q in qs))}$", pct(mean(q["hit_near"] for q in qs)),
                  gap_text(100 * (q["sampled_best_cost"] / r["e_min"] - 1) for q, r in zip(qs, rs))]
    gates = quantum("equal", role)[0]["two_qubit_gates"]
    rows.append(f"{names[role]} & {thin(gates)} & 1\\,000 & " + " & ".join(cells) + r" \\")
(ROOT / "report" / "table_q29.tex").write_text("\\newcommand{\\TableQnine}{%\n" + "\n".join(rows) + "\n}\n")

# ---------------------------------------------------------------- numbers ----
num = {}
num["QnInstances"] = str(len(records))
num["QnSeeds"] = ", ".join(str(s) for s in sorted({r["seed"] for r in records}))
num["QnStates"] = f"{2**N:,}".replace(",", "\\,")
num["QnGates"] = str(N * (N - 1) // 2)
for scen, short in (("equal", "Eq"), ("cpu_gpu", "Cpu")):
    rs = by_scenario[scen]
    best = quantum(scen, "best fixed schedule")
    num[f"QnBestShot{short}"] = sci(mean(q["p_near"] for q in best))
    num[f"QnBestHit{short}"] = pct(mean(q["hit_near"] for q in best))
    num[f"QnUniformShot{short}"] = sci(mean(r["per_shot_random"]["uniform_p_near"] for r in rs))
    num[f"QnGainUniform{short}"] = f"{mean(q['p_near'] for q in best) / mean(r['per_shot_random']['uniform_p_near'] for r in rs):,.0f}".replace(",", "\\,")
    share_gain = mean(q['p_near'] for q in best) / mean(r['per_shot_random']['target_share_p_near'] for r in rs)
    num[f"QnGainShare{short}"] = f"{share_gain:.1f}" if share_gain < 10 else thin(round(share_gain))
    num[f"QnCobylaHit{short}"] = pct(mean(q["hit_near"] for q in quantum(scen, "transferred trained angles")))
    num[f"QnLRHit{short}"] = pct(mean(q["hit_near"] for q in quantum(scen, "continuity")))
    for label, method, per_shot_key in CLASSICAL:                # same definition as the table (exact for the samplers)
        key = {"Random, uniform": "RandomUniform", "Random, target share": "RandomShare"}.get(label, label.split()[0])
        if per_shot_key:
            num[f"Qn{key}Hit{short}"] = pct(mean(1 - (1 - r["per_shot_random"][per_shot_key]) ** 1000 for r in rs))
        else:
            num[f"Qn{key}Hit{short}"] = pct(mean(r["classical"][method]["hit_near"] for r in rs))
    num[f"QnBalancedOpt{short}"] = str(sum(r["optimum_balanced"] for r in rs))
    num[f"QnNoiseBest{short}"] = sci(mean(q["p_near_noise_estimate"] for q in best))
    num[f"QnFidBest{short}"] = sci(best[0]["noise_fidelity_estimate"])
    num[f"QnNearStates{short}"] = f"{min(r['n_near'] for r in rs)}--{max(r['n_near'] for r in rs)}"
num["QnBestFixedEqName"] = by_scenario["equal"][0]["quantum"][0]["label"].split(",")[0].replace("LR-QAOA (", "").replace(")", "")
num["QnSimMinutes"] = f"{mean(sum(q['simulation_seconds'] for q in r['quantum']) for r in records) / 60:.0f}"
num["QnInstanceMinutes"] = f"{mean(r['seconds'] for r in records) / 60:.0f}"
num["QnEnumSeconds"] = f"{mean(r['enumeration_seconds'] for r in records):.0f}"
num["QnNormDev"] = sci(max(abs(q["norm"] - 1) for r in records for q in r["quantum"]))
# validation of the 29-qubit simulator against the notebook's float64 simulator (results/hpc_validation, written in test mode)
validation = {}
for path in sorted((ROOT / "results" / "hpc_validation").glob("hpc_n*/*.json")):
    v = json.loads(path.read_text())
    validation[v["n"]] = max(validation.get(v["n"], 0.0), max(q["test_max_probability_difference"] for q in v["quantum"]))
assert sorted(validation) == [12, 16, 20], f"validation files missing: {sorted(validation)}"
num["QnValidationMaxDiff"] = sci(max(validation.values()))
num["QnValidationSizes"] = ", ".join(str(n) for n in sorted(validation))
choice_file = json.loads((ROOT / "scripts" / "q29_choice.json").read_text())
num["QnAngleSpread"] = f"{max(max(c['cobyla_angles_n20'][str(c['trained']['p'])]['std']) for c in choice_file['choice'].values()):.2f}"
(ROOT / "report" / "numbers_q29.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()))

# ---------------------------------------------------------------- flat summary + rerun check (saved for later display) ----
import csv
rows_csv = []
for r in records:
    base = dict(case=r["case"], seed=r["seed"], scenario=r["scenario"], alpha=r["alpha"], e_min=r["e_min"], n_optimal=r["n_optimal"], n_near=r["n_near"])
    for label, method, key in CLASSICAL:
        c = r["classical"][method]
        per_shot = r["per_shot_random"][key] if key else ""
        hit = 1 - (1 - per_shot) ** 1000 if key else c["hit_near"]
        rows_csv.append(dict(base, kind="classical", method=label, p=0, two_qubit_gates=0, evaluations=c["evaluations"], per_shot_p_near=per_shot,
                             hit_near_1000=hit, gap_best_percent=100 * (c["mean_best_cost"] / r["e_min"] - 1), p_opt="", mean_cost=""))
    for q in r["quantum"]:
        rows_csv.append(dict(base, kind="quantum", method=q["label"], p=q["p"], two_qubit_gates=q["two_qubit_gates"], evaluations=q["shots"],
                             per_shot_p_near=q["p_near"], hit_near_1000=q["hit_near"], gap_best_percent=100 * (q["sampled_best_cost"] / r["e_min"] - 1),
                             p_opt=q["p_opt"], mean_cost=q["mean_cost"]))
with open(ROOT / "results" / f"hpc_n{N}" / "summary_q29.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows_csv[0]))
    writer.writeheader()
    writer.writerows(rows_csv)
first = {(r["case"], r["seed"], r["scenario"]): r for r in
         (json.loads(path.read_text()) for path in sorted((ROOT / "results" / f"hpc_n{N}" / "first_run").glob("*.json")))}
if first:
    diffs = {}
    for r in records:
        old_r = first.get((r["case"], r["seed"], r["scenario"]))
        if old_r is None:
            continue
        pairs = [("e_min", r["e_min"], old_r["e_min"])]
        pairs += [(f"classical {m} hit_near", c["hit_near"], old_r["classical"][m]["hit_near"]) for m, c in r["classical"].items()]
        for q, qo in zip(r["quantum"], old_r["quantum"]):
            pairs += [(f"quantum {k}", q[k], qo[k]) for k in ("p_opt", "p_near", "mean_cost", "p_balanced", "sampled_best_cost", "best_state")]
        for name, a, b in pairs:
            key = name.split(" hit")[0] if name.startswith("classical") else name
            diffs[key] = max(diffs.get(key, 0.0), abs(float(a) - float(b)))
    check = dict(instances_compared=sum((r["case"], r["seed"], r["scenario"]) in first for r in records), max_abs_difference=diffs,
                 identical=all(v == 0 for v in diffs.values()),
                 note="first run: results/hpc_n29/first_run (metrics only); rerun: results/hpc_n29 (metrics + raw arrays), same seeds and code path")
    (ROOT / "results" / f"hpc_n{N}" / "rerun_check.json").write_text(json.dumps(check, indent=1))
    print(f"rerun check: {check['instances_compared']} instances, identical = {check['identical']}, "
          f"largest difference {max(diffs.values()):.1e}")
(ROOT / "report" / "q29_numbers.json").write_text(json.dumps(num, indent=1))


# ---------------------------------------------------------------- split figure ----
def load_regions():
    nb = nbformat.read(ROOT / "mesh_idea_ZollerQu.ipynb", as_version=4)
    ns = {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
            continue
        if cell.source.startswith("# TASK_03_IMPLEMENTATION"):
            return ns
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell.source, f"cell{index}", "exec"), ns)


COLOUR_A, COLOUR_B, COLOUR_CUT = "#56B4E9", "#E69F00", "#D55E00"


def draw(ax, spec, regions, state, title):
    mesh = regions["mesh"]
    tri_xy = mesh["points"][mesh["triangles"]]
    centres, pairs, cost, work = regions["region_centroids"], regions["region_pairs"], regions["region_boundary_cost"], regions["region_work"]
    bits = np.zeros(N, dtype=int) if state is None else (int(state) >> np.arange(N)) & 1
    if state is None:
        ax.add_collection(PolyCollection(tri_xy, facecolors="#F2F2F2", edgecolors="#DDDDDD", linewidths=0.15, rasterized=True))
    else:
        on_b = bits[regions["cell_region"]].astype(bool)
        ax.add_collection(PolyCollection(tri_xy[~on_b], facecolors="#DCEFFA", edgecolors="#DCEFFA", linewidths=0.15, rasterized=True))
        ax.add_collection(PolyCollection(tri_xy[on_b], facecolors="#FBE7C4", edgecolors="#FBE7C4", linewidths=0.15, rasterized=True))
    for cut in spec["cutouts"]:
        ax.add_patch(plt.Polygon(cut, closed=True, facecolor="#BBBBBB", edgecolor="#888888"))
    for loop in spec["loops"]:
        ax.add_patch(plt.Polygon(loop, closed=True, fill=False, edgecolor="black", lw=0.8))
    widths = 0.5 + 2.5 * cost / cost.max()
    is_cut = bits[pairs[:, 0]] != bits[pairs[:, 1]]
    ax.add_collection(LineCollection(centres[pairs][~is_cut], colors="#555555", linewidths=widths[~is_cut], zorder=3))
    if is_cut.any():
        ax.add_collection(LineCollection(centres[pairs][is_cut], colors=COLOUR_CUT, linewidths=widths[is_cut], linestyles=(0, (2, 1.2)), zorder=3))
    sizes = 12 + 70 * work / work.max()
    for side, colour, marker in ((0, COLOUR_A if state is not None else "white", "o"), (1, COLOUR_B, "s")):
        chosen = bits == side
        if chosen.any():
            ax.scatter(centres[chosen, 0], centres[chosen, 1], s=sizes[chosen], c=colour, marker=marker, edgecolors="black", linewidths=0.5, zorder=4)
    lo, hi = spec["hull"].min(axis=0), spec["hull"].max(axis=0)
    ax.set_xlim(lo[0] - 0.03, hi[0] + 0.03)
    ax.set_ylim(lo[1] - 0.03, hi[1] + 0.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.6)


def direct_cost(regions, record, x):
    """Cost of a split from the regions alone (same formula as the notebook): cut fraction + 4 lambda (alpha - share)^2."""
    work, pairs, cost = regions["region_work"], regions["region_pairs"], regions["region_boundary_cost"]
    share = work[x == 1].sum() / work.sum()
    return cost[x[pairs[:, 0]] != x[pairs[:, 1]]].sum() / cost.sum() + 4 * record["lam"] * (record["alpha"] - share) ** 2


def describe(regions, record, state, method):
    x = (int(state) >> np.arange(N)) & 1
    work, pairs, cost = regions["region_work"], regions["region_pairs"], regions["region_boundary_cost"]
    share = work[x == 1].sum() / work.sum()
    cut = cost[x[pairs[:, 0]] != x[pairs[:, 1]]].sum() / cost.sum()
    gap = 100 * (direct_cost(regions, record, x) / record["e_min"] - 1)
    return f"{method}\nB share {share:.2f} (target {record['alpha']})\ncut {cut:.3f}, " + ("optimal" if gap < 1e-7 else f"cost +{gap:.0f}%")


ns = load_regions()
SEED, CASE = 7, "2 main"
spec = ns["MESH_CASES"][CASE]
regions = ns["coarsen_mesh"](ns["mesh_cases"][CASE], N, SEED)
fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.55))
for row, scen in enumerate(SCENARIOS):
    record = next(r for r in by_scenario[scen] if r["case"] == CASE and r["seed"] == SEED)
    best_q = next(q for q in record["quantum"] if q["role"] == "best fixed schedule")
    local = record["classical"]["local search (restarts)"]
    alpha_text = r"equal devices, $\alpha$ = 0.5" if scen == "equal" else r"CPU+GPU, $\alpha$ = 0.8"
    draw(axes[row, 0], spec, regions, None, f"{alpha_text}\n29 regions\n{len(regions['region_pairs'])} borders")
    draw(axes[row, 1], spec, regions, record["optimal_states"][0],
         describe(regions, record, record["optimal_states"][0], "exact optimum"))
    gap_q = best_q["best_gap_percent"]
    draw(axes[row, 2], spec, regions, best_q["best_state"],
         describe(regions, record, best_q["best_state"], "QAOA p = 5, best of 1,000 shots"))
    gap_l = local["best_gap_percent"]
    draw(axes[row, 3], spec, regions, local["best_state"],
         describe(regions, record, local["best_state"], "local search, best of 100 runs"))
handles = [Line2D([], [], marker="o", ls="none", markerfacecolor=COLOUR_A, markeredgecolor="black", ms=6, label="device A (CPU)"),
           Line2D([], [], marker="s", ls="none", markerfacecolor=COLOUR_B, markeredgecolor="black", ms=6, label="device B (GPU / 2nd CPU)"),
           Line2D([], [], color=COLOUR_CUT, lw=1.5, ls=(0, (2, 1.2)), label="cut border")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.4, w_pad=0.2)
fig.savefig(ROOT / "report" / "fig_q29_split.pdf", dpi=300, metadata={"CreationDate": None})
fig.savefig(ROOT / "results" / f"graph_cut_2_main_n{N}_seed{SEED}.png", dpi=160)
print(f"{len(records)} instances; table, numbers and figure written")
for k, v in num.items():
    print(f"  {k} = {v}")
