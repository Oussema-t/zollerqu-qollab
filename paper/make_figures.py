"""Figures and facts for the paper. Run from the repo root: python paper/make_figures.py

Reads only data/recorded_local.json, data/champaign.json and the package qopt.
Writes sweep.pdf, scaling.pdf, rounds.pdf, map.pdf and facts.json into paper/figures/.
"""
import json
import sys
from math import cos, radians, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qopt import bridge
from qopt.search import success_theory

OUT = ROOT / "paper" / "figures"
W = 3.3
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})
COLORS = ["#1b6ca8", "#d95f02", "#2a9d5c"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_text = (ROOT / "data" / "champaign.json").read_text(encoding="utf-8")
    data = json.loads(data_text)
    take = json.loads((ROOT / "data" / "recorded_local.json").read_text(encoding="utf-8"))

    # Same path as the page: load_data, then the JSON entry points with the take's params.
    bridge.load_data(data_text)
    params_json = json.dumps(take["params"])
    sweep = json.loads(bridge.sweep_analyze_json(params_json, json.dumps(take["plan"]),
                                                 json.dumps(take["sweep_counts"])))
    n_expected = sum(len(s["js"]) for s in take["plan"])
    if n_expected != len(take["sweep_counts"]):
        print(f"WARNING: plan expects {n_expected} circuits, take has {len(take['sweep_counts'])} count dicts")
    if [s["M"] for s in sweep] != [s["M"] for s in take["plan"]]:
        print("WARNING: sweep_analyze M values differ from the plan")

    # 1. sweep
    fig, ax = plt.subplots(figsize=(W, 2.4))
    for s, col in zip(sweep, COLORS):
        js = [p["j"] for p in s["points"]]
        jj = np.linspace(0, max(js), 300)
        ax.plot(jj, [success_theory(s["M"], s["N"], j) for j in jj], color=col, lw=1)
        ax.plot(js, [p["measured"] for p in s["points"]], "o", color=col, ms=3, label=f"M = {s['M']}")
    ax.set_xlabel("Grover iterations j")
    ax.set_ylabel("P(marked placement)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2)
    fig.savefig(OUT / "sweep.pdf")
    plt.close(fig)

    # 2. scaling
    study = take["study"]
    x = np.array([sqrt(e["N"]) for e in study])
    y = np.array([e["calls_mean"] for e in study])
    c = float((x * y).sum() / (x * x).sum())
    print(f"c = {c:.4f}")
    fig, ax = plt.subplots(figsize=(W, 2.6))
    lo = y - np.array([e["calls_min"] for e in study])
    hi = np.array([e["calls_max"] for e in study]) - y
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=COLORS[0], ms=3, capsize=2, lw=1, label="Durr-Hoyer (mean, min..max)")
    for e, xi, yi in zip(study, x, y):
        ax.annotate(f"m = {e['m']}", (xi, yi), textcoords="offset points", xytext=(4, -9))
    xx = np.linspace(0, x.max() * 1.05, 100)
    ax.plot(xx, xx ** 2, "--", color="black", lw=1, label="exact scan, N")
    ax.plot(xx, c * xx, ":", color=COLORS[1], lw=1.2, label=f"{c:.1f} sqrt N")
    ax.set_xlabel("sqrt N")
    ax.set_ylabel("placements scored")
    ax.set_ylim(0, (x.max() ** 2) * 1.05)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "scaling.pdf")
    plt.close(fig)

    # 3. rounds
    dh = take["dh"]
    rounds = dh["rounds"]
    r = np.arange(1, len(rounds) + 1)
    best = dh["result"]["best_cost"]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.step(r, [q["T"] for q in rounds], where="post", color=COLORS[0], lw=1, label="threshold T")
    ax.plot(r, [q["best_sample_cost"] for q in rounds], "o", color=COLORS[1], ms=2.5, label="measured sample")
    ax.axhline(best, color="black", ls="--", lw=0.8, label="exact minimum")
    ax.set_xlabel("round")
    ax.set_ylabel("cost")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "rounds.pdf")
    plt.close(fig)

    # 4. map
    blocks, sites, whs = data["blocks"], data["sites"], data["warehouses"]
    best_sites = dh["result"]["best_sites"]
    pop = np.array([b["pop"] for b in blocks], dtype=float)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.scatter([b["lon"] for b in blocks], [b["lat"] for b in blocks], s=pop / pop.max() * 40,
               color="#bbbbbb", lw=0, zorder=1)
    ax.scatter([w["lon"] for w in whs], [w["lat"] for w in whs], marker="s", s=28, color="#1b6ca8", zorder=3)
    ax.scatter([s["lon"] for s in sites], [s["lat"] for s in sites], s=12, color="black", zorder=4)
    chosen = [sites[i] for i in best_sites]
    ax.scatter([s["lon"] for s in chosen], [s["lat"] for s in chosen], s=45, color="#d62728", zorder=5)
    for s in sites:
        ax.annotate(str(s["id"]), (s["lon"], s["lat"]), textcoords="offset points", xytext=(4, 3), zorder=6)
    ax.set_aspect(1 / cos(radians(np.mean([b["lat"] for b in blocks]))))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.savefig(OUT / "map.pdf")
    plt.close(fig)

    # facts
    cl = json.loads(bridge.classical(params_json))
    ts = np.asarray(data["t_site_site"], dtype=float)
    off = ts[~np.eye(len(ts), dtype=bool)]
    facts = {
        "n_sweep_circuits": len(take["sweep_counts"]),
        "max_dev_per_M": {str(s["M"]): s["max_dev"] for s in sweep},
        "max_dev_overall": max(s["max_dev"] for s in sweep),
        "fitted_c": c,
        "calls_over_sqrtN_per_m": {str(e["m"]): e["calls_mean"] / sqrt(e["N"]) for e in study},
        "dh_best_cost": best,
        "dh_best_sites": best_sites,
        "dh_oracle_calls": dh["result"]["oracle_calls"],
        "dh_grover_iterations": dh["result"]["grover_iterations"],
        "dh_rounds": len(rounds),
        "N": dh["result"]["N"],
        "brute_force": {"cost": cl["brute_force"]["cost"], "sites": cl["brute_force"]["sites"],
                        "evaluations": cl["brute_force"]["evaluations"]},
        "greedy": {"cost": cl["greedy"]["cost"], "sites": cl["greedy"]["sites"],
                   "evaluations": cl["greedy"]["evaluations"]},
        "local_search": {"cost": cl["local_search"]["best_cost"], "sites": cl["local_search"]["sites"],
                         "local_minima": cl["local_search"]["local_minima"],
                         "evaluations": cl["local_search"]["evaluations"],
                         "restarts": bridge.LOCAL_SEARCH_RESTARTS},
        "total_population": float(pop.sum()),
        "n_blocks": len(blocks),
        "min_site_site_time": float(off.min()),
    }
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
