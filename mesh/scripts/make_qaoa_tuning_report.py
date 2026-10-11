"""Figures and a short report for the QAOA tuning study (reads only saved results; recomputes nothing but circuit sizes).

inputs : results/qaoa_tuning/tuning_n12.json, tuning_n12_large_calibration.json (if present), scaling/n*_p*.json (if present),
         results/qaoa_tuning/hpc/hpc_n24, hpc_n29 (tuned settings at 24 / 29 qubits, if present), results/hpc_n29 (first 29-qubit run)
outputs: results/qaoa_tuning/figures/*.png, results/qaoa_tuning/REPORT.md, results/qaoa_tuning/summary.csv
usage  : python scripts/make_qaoa_tuning_report.py
"""
import ast
import csv
import glob
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qollab"))
from mesh_ionq import circuit_from_spec, round_robin_pairs  # noqa: E402  (the circuit builder used for the Qollab / IonQ runs)

DIR = ROOT / "results" / "qaoa_tuning"
FIG = DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SHORT = {"2 equal devices": "equal devices (alpha = 0.5, lambda = 10)", "CPU + GPU (GPU 4x)": "CPU + GPU (alpha = 0.8, lambda = 3.98)"}
# colour-blind-safe (Okabe-Ito); every series also has its own marker and line style
STYLE = {"LR energy": ("#999999", "s", ":"), "LR success": ("#E69F00", "v", "--"), "optimised": ("#0072B2", "o", "-"),
         "warm": ("#009E73", "D", "-."), "old": ("#D55E00", "x", "none"), "random": ("#000000", "", ":")}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})


def load(name):
    path = DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def best_rows(scenario_block, layers):
    """Per depth: the four rows the report compares (selection by CALIBRATION score only)."""
    out = {}
    for p in layers:
        rows = [r for r in scenario_block["rows"] if r["p"] == p]
        out[p] = {"LR energy": next(r for r in rows if r["method"] == "LR-QAOA (energy-calibrated)"),
                  "LR success": next(r for r in rows if r["method"] == "LR-QAOA (success-calibrated)"),
                  "optimised": max((r for r in rows if "warm" not in r["method"]), key=lambda r: r["calibration_p_near"]),
                  "warm": max((r for r in rows if "warm" in r["method"]), key=lambda r: r["calibration_p_near"])}
    return out


def circuit_size(n, p, warm=False):
    """Two-qubit gates and depth of the QAOA circuit on a real n-qubit test instance (J, h from the saved validation runs),
    measured on the circuit built by qollab/mesh_ionq.py (warm start: RY start and the RY-RZ-RY mixer A)."""
    raw = np.load(ROOT / "results" / "hpc_validation" / f"hpc_n{n}" / "2_main_seed7_equal.raw.npz")
    J, h = raw["J"] / raw["scale"], raw["h"] / raw["scale"]
    spec = dict(n_qubits=n, kind="qaoa", J_scaled=np.triu(J, 1).tolist(), h_scaled=h.tolist(), zz_pairs=round_robin_pairs(n),
                start="ry" if warm else "plus", thetas=[0.7] * n, mixer="warm A" if warm else "x", gammas=[0.1] * p, betas=[0.1] * p)
    circuit = circuit_from_spec(spec, measure=False)
    return dict(two_qubit=int(circuit.count_ops().get("rzz", 0)), depth=int(circuit.depth()))


def figure_n12(data, tag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (scenario, block) in zip(axes, data["scenarios"].items()):
        layers, rows = data["layers"], best_rows(block, data["layers"])
        for key, label in (("LR energy", "LR-QAOA, energy-calibrated"), ("LR success", "LR-QAOA, success-calibrated"),
                           ("optimised", "QAOA, all 2p angles optimised (best optimiser on calibration)"),
                           ("warm", "warm start from the spectral split (classical seed)")):
            colour, marker, line = STYLE[key]
            ax.plot(layers, [rows[p][key]["test"]["p_near"] for p in layers], color=colour, marker=marker, linestyle=line, label=label)
        old = block["references"].get("per-shot QAOA + COBYLA p=2 (existing)")
        if old:
            ax.scatter([2], [old], color=STYLE["old"][0], marker="x", s=80, zorder=5, label="before: QAOA + COBYLA p = 2, trained per test mesh")
        ax.set_yscale("log")
        ax.set_xlabel("QAOA depth p")
        ax.set_title(SHORT[scenario], fontsize=11)
        ax.set_xticks(layers)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("test: mean P(within 5 % of optimum) per shot")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"12 qubits, 9 test meshes per scenario, settings chosen on calibration meshes only ({tag})", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    path = FIG / f"tuning_n12_{tag.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_optimisers(data, tag):
    """Calibration score vs test score for every optimiser: points far below the diagonal are over-fitted."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    markers = dict(zip(["COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B", "SPSA"], "o^sDv"))
    for ax, (scenario, block) in zip(axes, data["scenarios"].items()):
        for r in block["rows"]:
            opt = next((o for o in markers if f"({o})" in r["method"] and "warm" not in r["method"]), None)
            if opt:
                ax.scatter(r["calibration_p_near"], r["test"]["p_near"], marker=markers[opt], color=plt.cm.viridis(r["p"] / 9), s=36,
                           label=opt if r["p"] == data["layers"][-1] else None, edgecolor="k", linewidth=0.3)
        lim = [1e-4, 1]
        ax.plot(lim, lim, color="grey", lw=0.8, linestyle="--")
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel("calibration meshes: mean P(within 5 %)")
        ax.set_title(SHORT[scenario], fontsize=11)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("test meshes: mean P(within 5 %)")
    axes[0].legend(fontsize=8, title="optimiser (colour = depth p)", title_fontsize=8)
    fig.suptitle(f"Does the calibration score predict the test score? ({tag})", fontsize=11)
    fig.tight_layout()
    path = FIG / f"optimisers_n12_{tag.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def scaling_records():
    records = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(DIR / "scaling" / "n*_p*.json")))]
    return records


def big_runs(folder):
    """Per (n, scenario): mean P(near) of every QAOA setting and the classical hit rates, from run_29_qubits.py outputs."""
    out = {}
    for f in sorted(glob.glob(str(folder / "*_seed*_*.json"))):
        rec = json.loads(Path(f).read_text())
        key = (rec["n"], rec["scenario"])
        slot = out.setdefault(key, dict(quantum={}, classical={}, random=[], files=0))
        slot["files"] += 1
        for q in rec["quantum"]:
            slot["quantum"].setdefault(q["label"], []).append((q["p_near"], q["hit_near"], q["p_opt"], q["hit_opt"], q["two_qubit_gates"], q["p"]))
        for m, c in rec["classical"].items():
            slot["classical"].setdefault(m, []).append(c["hit_near"])
        slot["random"].append(rec["per_shot_random"]["uniform_p_near"])
    return out


def figure_scaling(records, runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, scenario in zip(axes, SHORT):
        recs = [r for r in records if r["scenario"] == scenario]
        for p in sorted({r["p"] for r in recs}):
            pts = sorted((r["n"], r["methods"][k]["test_mean"]["p_near"]) for r in recs if r["p"] == p
                         for k in r["methods"] if k.startswith("re-optimised"))
            t12 = [(12, ROWS12[scenario][p]["optimised"]["test"]["p_near"])] if p in ROWS12[scenario] else []
            big = []
            for (n, sc), slot in sorted(runs.items()):
                for label, vals in slot["quantum"].items():
                    if sc == scenario and label.startswith(f"tuned QAOA p = {p} "):
                        big.append((n, float(np.mean([v[0] for v in vals]))))
            series = sorted(t12 + pts + big)
            if series:
                ax.plot(*zip(*series), marker="o", color=plt.cm.viridis(p / 9), label=f"tuned QAOA p = {p}")
        old = sorted((n, float(np.mean([v[0] for v in vals]))) for (n, sc), slot in runs.items() if sc == scenario
                     for label, vals in slot["quantum"].items() if "best fixed schedule" in label)
        if old:
            ax.scatter(*zip(*old), marker="x", s=70, color=STYLE["old"][0], zorder=5, label="first 29-qubit run (best n = 20 schedule, p = 5)")
        rnd = sorted((n, float(np.mean(slot["random"]))) for (n, sc), slot in runs.items() if sc == scenario)
        rnd += sorted((r["n"], float(np.mean([row["per-shot random (uniform)"]["p_near"] for row in r["classical"]]))) for r in recs if r["p"] == min(x["p"] for x in recs))
        if rnd:
            ax.plot(*zip(*sorted(set(rnd))), color="k", linestyle=":", label="uniform random guess (per shot)")
        ax.set_yscale("log")
        ax.set_xlabel("qubits n (= mesh regions)")
        ax.set_title(SHORT[scenario], fontsize=11)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("test: mean P(within 5 % of optimum) per shot")
    axes[0].legend(fontsize=8)
    fig.suptitle("Tuned settings at larger sizes (angles re-optimised at 16 / 20 on calibration meshes; frozen for 24 / 29)", fontsize=11)
    fig.tight_layout()
    path = FIG / "scaling.png"
    fig.savefig(path)
    plt.close(fig)
    return path


NOISE_ERROR = 0.005   # Task 04's crude model: error per two-qubit gate, global depolarising (an estimate, not a noise simulation)


def noise_estimate(test, gates, uniform):
    """Mean over test meshes of F P + (1 - F) P_uniform with F = (1 - e)^gates (Task 04's estimate)."""
    fidelity = (1 - NOISE_ERROR) ** gates
    return float(np.mean([fidelity * p + (1 - fidelity) * u for p, u in zip(test["per_instance"], uniform)]))


def pct(x):
    return f"{100 * x:.2f} %" if x >= 1e-4 else f"{100 * x:.1e} %"


if __name__ == "__main__":
    small, large = load("tuning_n12.json"), load("tuning_n12_large_calibration.json")
    main_data, main_tag = (large, "18 calibration meshes") if large else (small, "4 calibration meshes")
    ROWS12 = {s: best_rows(b, main_data["layers"]) for s, b in main_data["scenarios"].items()}
    figs = [figure_n12(d, t) for d, t in ((small, "4 calibration meshes"), (large, "18 calibration meshes")) if d]
    figs += [figure_optimisers(d, t) for d, t in ((small, "4 calibration meshes"), (large, "18 calibration meshes")) if d]
    records = scaling_records()
    runs = {}
    for folder in (DIR / "hpc" / "hpc_n24", DIR / "hpc" / "hpc_n29", ROOT / "results" / "hpc_n29"):   # tuned runs and the first 29-qubit run
        for key, slot in big_runs(folder).items():
            if key in runs:                       # same size and scenario: keep both sets of QAOA settings (classical numbers are identical)
                runs[key]["quantum"].update({f"{label} [first 29-qubit run]": v for label, v in slot["quantum"].items()})
            else:
                runs[key] = slot
    if records or runs:
        figs.append(figure_scaling(records, runs))

    lines = ["# QAOA tuning study (mesh partitioning)", "",
             "Generated by `scripts/make_qaoa_tuning_report.py` from saved results only. Noiseless statevector simulation; no claim of "
             "quantum advantage (the exact optimum of every instance is computed classically by enumeration).", "",
             "## Protocol (no leakage)", "", ast.get_docstring(ast.parse((ROOT / "scripts" / "qaoa_tuning.py").read_text())).split("usage:")[0].strip(),
             "", f"Main 12-qubit table below: calibration set '{main_data.get('calibration', 'small')}' ({main_tag}).", "",
             f"## 12 qubits ({main_tag}); test = 9 meshes per scenario", ""]
    csv_rows = []
    checkpoint = json.load(gzip.open(ROOT / "results" / "task04_checkpoint_small.json.gz", "rt"))["instances"]
    for scenario, block in main_data["scenarios"].items():
        ref = block["references"]
        uniform = [checkpoint["|".join(map(str, k))]["classical"]["uniform random sampling"]["per_shot"]["p_near"] for k in block["test_instances"]]
        lines += [f"### {SHORT[scenario]}", "",
                  "| p | setting (chosen on calibration) | test P(opt) | test P(within 5 %) | hit@1000 opt | hit@1000 within 5 % | 2q gates | depth | training evals x meshes | noise estimate P(within 5 %) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for p in main_data["layers"]:
            sizes = dict(plain=circuit_size(12, p), warm=circuit_size(12, p, warm=True))
            for key in ("LR energy", "LR success", "optimised", "warm"):
                size = sizes["warm" if key == "warm" else "plain"]
                r = ROWS12[scenario][p][key]
                t = r["test"]
                n_cal = 18 if large else 4
                lines.append(f"| {p} | {r['method']} | {pct(t['p_opt'])} | {pct(t['p_near'])} | {t['hit_opt']:.3f} | {t['hit_near']:.3f} | "
                             f"{size['two_qubit']} | {size['depth']} | {r['calibration_evaluations']} x {n_cal} | {pct(noise_estimate(t, size['two_qubit'], uniform))} |")
                csv_rows.append(dict(n=12, scenario=scenario, p=p, setting=r["method"], calibration=main_tag, test_p_opt=t["p_opt"],
                                     test_p_near=t["p_near"], hit_opt_1000=t["hit_opt"], hit_near_1000=t["hit_near"],
                                     two_qubit_gates=size["two_qubit"], depth=size["depth"], training_evaluations=r["calibration_evaluations"]))
        lines += ["", f"Noise estimate: Task 04's crude model ({100 * NOISE_ERROR:.1f} % error per two-qubit gate, global depolarising), "
                  "not a noise simulation; it shows how the deeper circuits lose their noiseless gain. Per-shot uniform random: "
                  f"{pct(float(np.mean(uniform)))}.", "", "References on the same 9 test meshes (existing benchmark, not recomputed): " + "; ".join(
            f"{k}: {pct(v) if 'per-shot' in k else f'{v:.3f}'}" for k, v in ref.items()), ""]
    if records:
        lines += ["## Larger sizes: 16 and 20 qubits (test = cases 2 and 9, mesh seeds 7-9)", "",
                  "| n | scenario | p | setting | training evals x meshes | test P(opt) | test P(within 5 %) | hit@1000 within 5 % | 2q gates | depth |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(records, key=lambda r: (r["n"], r["scenario"], r["p"])):
            for label, m in r["methods"].items():
                t = m["test_mean"]
                lines.append(f"| {r['n']} | {r['scenario']} | {r['p']} | {label} | {m['training_evaluations']} x {m['training_instances']} | "
                             f"{pct(t['p_opt'])} | {pct(t['p_near'])} | {t['hit_near']:.3f} | {r['circuit']['two_qubit_gates']} | {r['circuit']['depth']} |")
                csv_rows.append(dict(n=r["n"], scenario=r["scenario"], p=r["p"], setting=label, calibration=f"{len(r['calibration_instances'])} meshes at n",
                                     test_p_opt=t["p_opt"], test_p_near=t["p_near"], hit_opt_1000=t["hit_opt"], hit_near_1000=t["hit_near"],
                                     two_qubit_gates=r["circuit"]["two_qubit_gates"], depth=r["circuit"]["depth"], training_evaluations=m["training_evaluations"]))
        classical = {}
        for r in records:
            for row in r["classical"]:
                for m, v in row.items():
                    if "hit_near" in v:
                        classical.setdefault((r["n"], r["scenario"], m), []).append(v["hit_near"])
        lines += ["", "Classical baselines on the same test meshes (hit rate of a within-5 % split per 1,000-evaluation run): " + "; ".join(
            f"n = {n}, {s.split(' (')[0]}, {m}: {np.mean(v):.2f}" for (n, s, m), v in sorted(classical.items())), ""]
    if runs:
        lines += ["## 24 and 29 qubits (settings frozen from n <= 20; run_29_qubits.py engine)", "",
                  "| n | scenario | setting | test meshes | mean P(within 5 %) | mean hit@1000 | 2q gates |", "|---|---|---|---|---|---|---|"]
        for (n, scenario), slot in sorted(runs.items()):
            for label, vals in slot["quantum"].items():
                lines.append(f"| {n} | {scenario} | {label} | {len(vals)} | {pct(float(np.mean([v[0] for v in vals])))} | "
                             f"{np.mean([v[1] for v in vals]):.4f} | {vals[0][4]} |")
            lines.append(f"| {n} | {scenario} | uniform random guess | {len(slot['random'])} | {pct(float(np.mean(slot['random'])))} | | 0 |")
            for m, v in slot["classical"].items():
                lines.append(f"| {n} | {scenario} | classical: {m} (hit rate per 1,000-evaluation run) | {len(v)} | | {np.mean(v):.2f} | 0 |")
    lines += ["", "## Figures", ""] + [f"- `{f.relative_to(ROOT)}`" for f in figs]
    (DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    with open(DIR / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {DIR / 'REPORT.md'}, {DIR / 'summary.csv'} and {len(figs)} figures")
