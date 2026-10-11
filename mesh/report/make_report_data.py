"""Builds the report figure and LaTeX tables directly from the saved result files (no number is typed by hand)."""
import csv, gzip, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LARGE = os.path.join(HERE, "..", "results_large_hpc")
S04 = json.load(open(os.path.join(RES, "task04_summary_small.json")))
SVQ = json.load(open(os.path.join(RES, "varqite_summary_small.json")))
SCEN = ["2 equal devices", "CPU + GPU (GPU 4x)"]


def summary(scen, method, p):
    return next(m for m in S04["method_summary"] if m["scenario"] == scen and m["method"] == method and m["p"] == p)


def qvc(scen, label):
    return S04["quantum_vs_classical"][f"{scen} | hit_near | {label}"][0]


def vq(scen, method, field):
    return float(np.mean([r[field] for r in SVQ["rows"] if r["scenario"] == scen and r["method"] == method and r["n"] == 12]))


def pct(x):
    if x is not None and x < 0.00005:
        return "$<$0.01"
    return "--" if x is None else (f"{100 * x:.0f}" if x >= 0.095 else (f"{100 * x:.1f}" if x >= 0.0095 else f"{100 * x:.2f}"))


SHOTS = S04["run_settings"]["shots"]
def big(x):
    return f"{x:,.0f}".replace(",", "\\,")
rows = []   # (label, 2q gates, training circuits, total evaluations, [(per-shot, hit) per scenario], group)
u = [summary(s, "uniform control", 0) for s in SCEN]
rows.append(("Uniform random sampling", "0", "0", big(SHOTS), [(u[i]["p_near"], qvc(s, "uniform random")) for i, s in enumerate(SCEN)], "c"))
rows.append(("Random with target share", "0", "0", big(SHOTS), [(u[i]["p_near_target_share_sampler"], qvc(s, "random, target share")) for i, s in enumerate(SCEN)], "c"))
for label, key, total in (("Simulated annealing", "simulated annealing", SHOTS), ("Local search (restarts)", "local search", SHOTS), ("Spectral bisection", "spectral bisection", 22)):
    rows.append((label, "0", "0", big(total), [(None, qvc(s, key)) for s in SCEN], "c"))
for label, method, p in (("LR-QAOA, $p=2$", "LR-QAOA", 2), ("QAOA + COBYLA, $p=2$", "QAOA COBYLA", 2), ("CVaR-QAOA + COBYLA, $p=2$", "CVaR COBYLA", 2),
                         ("Warm start (spectral)$^\\dagger$, $p=2$", "WS rounded A", 2)):
    ms = [summary(s, method, p) for s in SCEN]
    training = float(np.mean([m["evaluations"] for m in ms]))
    rows.append((label, "132", f"{training:.0f}", big((training + 1) * SHOTS), [(m["p_near"], m["hit_near"]) for m in ms], "q"))
for label, method in (("varQITE ZY (paper, all pairs)", "varQITE ZY (paper, all pairs)"), ("varQITE A (all pairs)", "varQITE A (all pairs)"),
                      ("varQITE A$^{+}$ (edges, our ext.)", "varQITE A+ (graph edges, ry layer)"), ("varQITE B (IonQ-native, edges)", "varQITE B (IonQ-native, graph edges)")):
    gates = vq(SCEN[0], method, "two_qubit_gates")
    circuits = float(np.mean([vq(s, method, "training_circuits") for s in SCEN]))
    rows.append((label, f"{gates:.0f}", big(circuits), big((circuits + 1) * SHOTS), [(vq(s, method, "p_near"), vq(s, method, "hit_near")) for s in SCEN], "v"))
with open(os.path.join(HERE, "table_n12.tex"), "w") as f:   # rows as a macro (an \\input inside tabular breaks \\bottomrule)
    f.write("\\newcommand{\\TableNtwelve}{%\n")
    previous = None
    for label, gates, circuits, total, values, group in rows:
        if previous and group != previous:
            f.write("\\midrule\n")
        cells = " & ".join(f"{pct(p)} & {pct(h)}" for p, h in values)
        f.write(f"{label} & {gates} & {circuits} & {total} & {cells} \\\\\n")
        previous = group
    f.write("}\n")

# ---- scaling data (server, large mode) ----
lrows = list(csv.DictReader(open(os.path.join(LARGE, "task04_runs_large.csv"))))
ck = json.load(gzip.open(os.path.join(LARGE, "task04_checkpoint_large.json.gz"), "rt"))["instances"]
def qhit(scen, variant, opt, p, n):
    v = [float(r["hit_near"]) for r in lrows if r["scenario"] == scen and r["variant"] == variant and r["optimiser"] == opt and int(r["p"]) == p and int(r["n"]) == n and r["init"] != "random"]
    return np.mean(v) if v else np.nan
def chit(scen, method, n):
    v = [inst["classical"][method]["hit_near"] for key, inst in ck.items() if key.split("|")[1] == str(n) and key.split("|")[3] == scen and key.split("|")[0] in ("2 main", "9 priority")]
    return np.mean(v) if v else np.nan
NS = [12, 16, 20, 24]
scaling = {s: {"Local search": [chit(s, "local search (restarts)", n) for n in NS], "Spectral bisection": [chit(s, "spectral bisection", n) for n in NS],
               "Simulated annealing": [chit(s, "simulated annealing", n) for n in NS], "Random, target share": [chit(s, "random with target share", n) for n in NS],
               "LR-QAOA $p=2$": [qhit(s, "LR-QAOA", "fixed schedule", 2, n) for n in NS], "QAOA+COBYLA $p=2$": [qhit(s, "QAOA", "COBYLA", 2, n) for n in NS]} for s in SCEN}
json.dump({s: {k: [None if np.isnan(x) else x for x in v] for k, v in d.items()} for s, d in scaling.items()}, open(os.path.join(HERE, "scaling.json"), "w"), indent=1)

# ---- figure ----
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 5.8, "lines.linewidth": 1.2})
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.25), constrained_layout=True)
style = {"Local search": ("#0072B2", "o", "-"), "Spectral bisection": ("#009E73", "s", "-"), "Simulated annealing": ("#56B4E9", "^", "-"),
         "Random, target share": ("#999999", "x", ":"), "LR-QAOA $p=2$": ("#E69F00", "D", "--"), "QAOA+COBYLA $p=2$": ("#D55E00", "P", "--")}
for ax, s, tag in zip(axes[:2], SCEN, ("(a) equal devices", "(b) CPU + GPU (GPU 4x)")):
    for name, values in scaling[s].items():
        c, m, ls = style[name]
        ax.plot(NS, values, color=c, marker=m, ls=ls, ms=3.5, label=name)
    ax.set_xticks(NS); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    ax.set_xlabel("regions = qubits $n$"); ax.set_title(tag, loc="left")
axes[0].set_ylabel("P(1000 tries contain\ncost $\\leq$ 1.05 opt.)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower left", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.0)
vq_ck = json.load(gzip.open(os.path.join(RES, "varqite_checkpoint_small.json.gz"), "rt"))["entries"]
vq_style = {"varQITE ZY (paper, all pairs)": ("#D55E00", "ZY (paper)"), "varQITE A (all pairs)": ("#56B4E9", "A, all pairs"),
            "varQITE A+ (graph edges, ry layer)": ("#009E73", "A$^+$ (ours)"), "varQITE B (IonQ-native, graph edges)": ("#CC79A7", "B (native)"),
            "varQITE A (graph edges)": ("#0072B2", "A, edges")}
for ax, s, tag in zip(axes[2:], SCEN, ("(c) varQITE, equal", "(d) varQITE, CPU + GPU")):
    for name, (c, short) in vq_style.items():
        curves = [np.array([h["ratio"] for h in e["history"]]) for k, e in vq_ck.items() if not k.startswith("calibration") and k.split("|")[3] == s and k.split("|")[4] == name]
        if not curves: continue
        L = max(len(x) for x in curves)
        mean = np.mean([np.concatenate([x, np.full(L - len(x), x[-1])]) for x in curves], axis=0)
        ax.plot(mean, color=c, label=short)
    for method, ls, lab in (("QAOA COBYLA", "--", "QAOA+COBYLA"), ("uniform control", ":", "uniform")):
        r = summary(s, method, 2 if method != "uniform control" else 0)["mean_ratio"]
        ax.axhline(r, color="black", ls=ls, lw=0.9, label=lab)
    ax.set_xlabel("varQITE step"); ax.set_title(tag, loc="left"); ax.grid(alpha=0.3); ax.set_ylim(0.76, 1.005)
axes[2].set_ylabel("approx. ratio $r$ of $\\langle E\\rangle$")
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower right", ncol=4, frameon=False, handlelength=1.5, columnspacing=0.8)
fig.savefig(os.path.join(HERE, "fig_results.pdf"))
fig.savefig(os.path.join(HERE, "fig_results_preview.png"), dpi=250)

# ---- numbers used in the text ----
t = S04["workload"]; tests = S04["tests"]
macros = {
    "NQAOAruns": f"{t['qaoa_runs']:,}".replace(",", "\\,"), "NEvals": f"{t['objective_evaluations']:,}".replace(",", "\\,"),
    "NLargeRuns": f"{len(lrows):,}".replace(",", "\\,"), "NLargeInstances": str(len(ck)), "VQZYequal": pct(vq(SCEN[0], "varQITE ZY (paper, all pairs)", "p_near")),
    "COBYLAequal": pct(summary(SCEN[0], "QAOA COBYLA", 2)["p_near"]), "LRgainEqual": f"{summary(SCEN[0], 'LR-QAOA', 2)['p_near'] / u[0]['p_near']:.1f}",
    "LRgainShare": f"{summary(SCEN[1], 'LR-QAOA', 2)['p_near'] / u[1]['p_near_target_share_sampler']:.1f}",
    "LRgainUniformCPU": f"{summary(SCEN[1], 'LR-QAOA', 2)['p_near'] / u[1]['p_near']:.0f}",
    "COBYLAshare": f"{summary(SCEN[1], 'QAOA COBYLA', 2)['p_near'] / u[1]['p_near_target_share_sampler']:.1f}",
    "NoiseLR": pct(summary(SCEN[1], "LR-QAOA", 2)["p_near_noise_estimate"]), "IdealLR": pct(summary(SCEN[1], "LR-QAOA", 2)["p_near"]),
}
def sc(scen, method, n):
    value = scaling[scen][method][NS.index(n)]
    return f"{100 * value:.0f}"
macros.update(ScLocalTwelveEq=sc(SCEN[0], "Local search", 12), ScCobylaTwelveEq=sc(SCEN[0], "QAOA+COBYLA $p=2$", 12),
              ScCobylaTwentyEq=sc(SCEN[0], "QAOA+COBYLA $p=2$", 20), ScSpectralTwentyEq=sc(SCEN[0], "Spectral bisection", 20),
              ScLocalTwentyEq=sc(SCEN[0], "Local search", 20), ScCobylaTwentyCpu=sc(SCEN[1], "QAOA+COBYLA $p=2$", 20),
              ScSpectralTwentyCpu=sc(SCEN[1], "Spectral bisection", 20), ScLocalTwentyCpu=sc(SCEN[1], "Local search", 20),
              AblFields=pct(summary(SCEN[1], "ablation: fields only", 2)["p_near"]), AblCouplings=pct(summary(SCEN[1], "ablation: couplings only", 2)["p_near"]),
              VQZYcircuits=f"{vq(SCEN[0], 'varQITE ZY (paper, all pairs)', 'training_circuits'):,.0f}".replace(",", "\\,"),
              RuntimeSpread=f"{max(abs(v[0]) for k, v in S04['modelled_outcome_excess_percent'].items() if 'modelled_hours' in k):.1f}")
with open(os.path.join(HERE, "table_scaling.tex"), "w") as f:
    f.write("\\newcommand{\\TableScaling}{%\n")
    for name in ("Local search", "Spectral bisection", "Simulated annealing", "Random, target share", "LR-QAOA $p=2$", "QAOA+COBYLA $p=2$"):
        cells = []
        for scen in SCEN:
            cells += ["--" if scaling[scen][name][k] is None or np.isnan(scaling[scen][name][k]) else f"{100 * scaling[scen][name][k]:.0f}" for k in range(len(NS))]
        f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
    f.write("}\n")
VQL = json.load(open(os.path.join(LARGE, "varqite_summary_large.json")))
def vql(scen, method, field, n=16):
    v = [r[field] for r in VQL["rows"] if r["scenario"] == scen and r["method"] == method and r["n"] == n]
    return float(np.mean(v)) if v else float("nan")
macros.update(VQLApEq=f"{vql(SCEN[0], 'varQITE A+ (graph edges, ry layer)', 'mean_ratio'):.3f}", VQLApCpu=f"{vql(SCEN[1], 'varQITE A+ (graph edges, ry layer)', 'mean_ratio'):.3f}",
              VQLnearMax=pct(max(r["p_near"] for r in VQL["rows"] if r["n"] == 16 and r["method"].startswith("varQITE"))),
              COBYLAtotal=big((np.mean([summary(s, "QAOA COBYLA", 2)["evaluations"] for s in SCEN]) + 1) * SHOTS),
              VQZYtotal=big((np.mean([vq(s, "varQITE ZY (paper, all pairs)", "training_circuits") for s in SCEN]) + 1) * SHOTS))
BATCH = json.load(open(os.path.join(HERE, "..", "qollab", "mesh_batch.json")))
with open(os.path.join(HERE, "table_ionq.tex"), "w") as f:
    f.write("\\newcommand{\\TableIonQ}{%\n")
    names = [c["name"] for c in BATCH["scenarios"][0]["circuits"]]
    for k, name in enumerate(names):
        cs = [scen["circuits"][k] for scen in BATCH["scenarios"]]
        label = name.replace("varQITE A+ (graph edges, ry layer)", "varQITE A$^{+}$ (edges)").replace("varQITE ZY (paper, all pairs)", "varQITE ZY (paper)")
        f.write(f"{label} & {cs[0]['two_qubit_gates']} & " + " & ".join(f"{pct(c['predicted']['p_opt'])} & {pct(c['predicted']['p_near'])}" for c in cs) + " \\\\\n")
    f.write("}\n")
macros.update(NIonQ=str(sum(len(s["circuits"]) for s in BATCH["scenarios"])))
# meshes (frozen data files), the cross-platform check and the device-choice plans (Task 03b, saved as JSON)
import glob
triangles = [len(np.load(path)["triangles"]) for path in sorted(glob.glob(os.path.join(HERE, "..", "data", "frozen_meshes", "*_seed7.npz")))]
fp_mac = json.load(open(os.path.join(LARGE, "fingerprints_mac_arm64.json")))["meshes"]
fp_linux = json.load(open(os.path.join(LARGE, "fingerprints_linux_x86_64.json")))["meshes"]
POOL = json.load(open(os.path.join(RES, "hpc_pool_goals_seed7.json")))
macros.update(TriMin=f"{min(triangles):,}".replace(",", "\\,"), TriMax=f"{max(triangles):,}".replace(",", "\\,"),
              MeshDiffer=str(sum(fp_mac[k] != fp_linux.get(k) for k in fp_mac)), MeshTotal=str(len(fp_mac)), PoolDevices=str(len(POOL["devices"])))
for goal, tag in (("fastest", "Fast"), ("overnight", "Night"), ("lowest energy", "Eco")):
    macros[f"Pool{tag}Time"], macros[f"Pool{tag}Energy"] = f"{POOL['plans'][goal]['time_h']:.1f}", f"{POOL['plans'][goal]['energy_kwh']:.1f}"
macros["PoolNaiveTime"], macros["PoolNaiveEnergy"] = f"{POOL['no_planning']['time_h']:.1f}", f"{POOL['no_planning']['energy_kwh']:.1f}"
macros["PoolNightLimit"] = f"{POOL['overnight_hours']:g}"
with open(os.path.join(HERE, "numbers.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(os.path.join(HERE, "table_n12.tex")).read()); print(macros); print(json.dumps(scaling, default=float)[:600])
