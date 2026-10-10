"""Figures of the two-page paper, regenerated from the saved Task 3/4 results (no recomputation of the benchmark).

Run from the repo root:  python paper/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `python paper/make_figures.py` works

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from stocks import ising, kaggle_data, qaoa_sim, task3_report
from stocks import task4_report as t4

OUT = Path(__file__).resolve().parent / "figures"
MODES = list(ising.MODES)
MODE_NAME = {"long_short": "Mode A: long/short", "buy_or_skip": "Mode B: buy or skip"}
BLUE, ORANGE, TEAL, GREY, BROWN = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9", "#a6611a"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = ("Utilities", "Real estate", "Materials (copper)")
OTHER_LABEL = "Utilities, real estate, copper"
GROUP_COLOR = dict(zip([g for g in kaggle_data.GROUPS if g not in OTHER] + [OTHER_LABEL], SLOTS))
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5})


def color_of(ticker):
    group = kaggle_data.GROUP_OF[ticker]
    return GROUP_COLOR[OTHER_LABEL if group in OTHER else group]


def best_extra(extra):
    if extra.get("status") != "completed" or "summary" not in extra or not extra["summary"]["best_by_p"]:
        return None
    best_by_p = extra["summary"]["best_by_p"]
    p = min(best_by_p, key=lambda q: best_by_p[q]["objective_value"])
    return p, best_by_p[p]["metrics"]


def ladder(results):
    """Every method on one scale: probability that one shot / one try is the optimal portfolio."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, m in zip(axes, MODES):
        r = results["models"][t4.key_of(m, False)]
        rows = [("p = 0 control (uniform)", r["baselines"]["p0_control_uniform"])]
        rows += [(f"LR-QAOA p={p}", r["lr"][p]["best"]) for p in sorted(r["lr"], key=int)]
        rows += [(f"free-angle QAOA p={p}", r["standard"]["best_by_p"][p]["metrics"]) for p in sorted(r["standard"]["best_by_p"], key=int)]
        for name, label in (("cvar_0.1", "CVaR 0.1"), ("cvar_0.25", "CVaR 0.25"), ("multi_angle", "multi-angle"),
                            ("dc", "counterdiabatic"), ("varqite", "VarQITE (McLachlan)")):
            picked = best_extra(results["extras"][t4.key_of(m, False)].get(name, {}))
            if picked:
                rows.append((f"{label} p={picked[0]}", picked[1]))
        v = results["extras"][t4.key_of(m, False)]["varqite_pauli"]
        rows.append((f"varQITE Pauli-update (ansatz {v['best']['kind']})", v["best"]["metrics"]))
        floor = 1e-5
        names = [n for n, _ in rows]
        values = [met["p_optimum_per_shot"] for _, met in rows]
        colors = [GREY if "control" in n else BLUE if n.startswith("LR") else ORANGE if n.startswith("free") else TEAL for n in names]
        y = np.arange(len(rows))
        ax.barh(y, [max(v, floor) for v in values], color=colors, height=0.7)
        for yy, v in zip(y, values):
            ax.annotate(f"{v:.4f}" if v >= floor else f"<{floor:g}", (max(v, floor), yy), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=6)
        handles = []
        for (name, c), style in zip(r["baselines"]["classical"].items(), (":", "--", "-.")):
            if name == "random":
                continue
            ax.axvline(max(c["hit_rate_per_try"], floor), color=BROWN, linestyle=style, linewidth=1.2)
            handles.append(Line2D([], [], color=BROWN, linestyle=style, label=f"{name}: {c['hit_rate_per_try']:.2f} per try"))
        ax.set_xscale("log")
        ax.set_xlim(left=floor * 0.8, right=3.0)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("P(one shot / one try is the optimal portfolio)")
        ax.set_title(MODE_NAME[m], loc="left")
        ax.grid(True, axis="x", color="#e5e4df", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"ladder.{ext}", dpi=220, bbox_inches="tight")


def cut(parameters):
    """Mode A, 12 assets: the correlation graph and the circuit's most likely long/short split."""
    ctx = task3_report.load_context()
    setup = task3_report.size_setup(ctx, 12)
    model = task3_report.build_model(setup, "long_short", 1.0, False)
    sim = qaoa_sim.QaoaSimulator(model)
    lr = parameters["modes"]["long_short"]["linear_ramp"]
    best_p = max(lr, key=lambda p: lr[p]["p_optimum_per_shot"])
    probs = sim.probabilities(lr[best_p]["gammas"], lr[best_p]["betas"])
    states = ising.all_states(12)
    top = int(np.argmax(probs))
    spins, tickers, rho = states[top], setup["tickers"], setup["stats"].corr
    order = [t for t in kaggle_data.TICKERS if t in tickers]
    links = [(i, j) for i in range(12) for j in range(i + 1, 12)]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.05), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 12, endpoint=False)
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(order, angles)}
    for i, j in links:
        (x1, y1), (x2, y2) = pos[tickers[i]], pos[tickers[j]]
        ax.plot([x1, x2], [y1, y2], color=TEAL if rho[i, j] > 0 else BROWN, linewidth=0.2 + 2.2 * abs(rho[i, j]), alpha=0.6, zorder=1)
    for t, (x, y) in pos.items():
        ax.scatter(x, y, s=55, color=color_of(t), edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(t, (1.27 * x, 1.27 * y), ha="center", va="center", fontsize=5.5)
    ax.set_title("(a) the graph: 66 links", loc="left")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    ax = axes[1]
    side = {t: spins[tickers.index(t)] for t in order}
    columns = {1: [t for t in order if side[t] > 0], -1: [t for t in order if side[t] < 0]}
    pos2 = {}
    for x, s in ((0.0, 1), (1.0, -1)):
        ys = np.linspace(1.0, -1.0, len(columns[s])) if len(columns[s]) > 1 else [0.0]
        for t, y in zip(columns[s], ys):
            pos2[t] = (x, y)
    for i, j in links:
        a, b = tickers[i], tickers[j]
        (x1, y1), (x2, y2) = pos2[a], pos2[b]
        if side[a] == side[b]:
            ax.plot([x1, x2], [y1, y2], color="#cfcdc4", linewidth=0.4, alpha=0.6, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color=TEAL if rho[i, j] > 0 else BROWN, linewidth=0.2 + 2.2 * abs(rho[i, j]), alpha=0.7, zorder=2)
    for t, (x, y) in pos2.items():
        ax.scatter(x, y, s=55, color=color_of(t), edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(t, (x - 0.14 if x == 0 else x + 0.14, y), ha="right" if x == 0 else "left", va="center", fontsize=5.5)
    ax.text(0.0, 1.28, f"LONG ({len(columns[1])})", ha="center", fontsize=6.5, fontweight="bold")
    ax.text(1.0, 1.28, f"SHORT ({len(columns[-1])})", ha="center", fontsize=6.5, fontweight="bold")
    ax.set_title(f"(b) most likely cut (p = {best_p})", loc="left")
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(-1.3, 1.5)
    ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=TEAL, linewidth=1.5, label="move together"),
                        Line2D([], [], color=BROWN, linewidth=1.5, label="move opposite"),
                        Line2D([], [], color="#cfcdc4", linewidth=1, label="inside one side")],
               loc="lower center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cut.{ext}", dpi=220, bbox_inches="tight")
    return top == int(np.argmin(sim.energies))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ladder(t4.load())
    print("ladder written; most likely state is the optimum:", cut(json.loads(t4.PARAMETERS_PATH.read_text())))
