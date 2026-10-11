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


def budget_rows(results, fair):
    """One row per method: hit@B (exact for circuits, measured for heuristics), P_opt per shot, quality, tuning cost."""
    budget = fair["budget"]
    rows = []  # (label, kind, {mode: dict})

    def pick(mode, extra_name=None, lr=False, standard=False):
        key = t4.key_of(mode, False)
        r = results["models"][key]
        if lr:
            entries = {p: e["best"] for p, e in r["lr"].items()}
            p = max(entries, key=lambda q: entries[q]["p_optimum_per_shot"])
            return p, entries[p], len(results["meta"]["tuning"]["lr_grid"]["dgamma"]) * len(results["meta"]["tuning"]["lr_grid"]["dbeta"])
        if standard:
            entries = r["standard"]["best_by_p"]
            p = max(entries, key=lambda q: entries[q]["metrics"]["p_optimum_per_shot"])
            return p, entries[p]["metrics"], sum(x["evaluations"] for x in r["standard_runs"] if str(x["p"]) == str(p))
        e = results["extras"][key][extra_name]
        if extra_name == "varqite_pauli":
            return None, e["best"]["metrics"], sum(x["evaluations"] for x in e["runs"])
        entries = e["summary"]["best_by_p"]
        p = max(entries, key=lambda q: entries[q]["metrics"]["p_optimum_per_shot"])
        return p, entries[p]["metrics"], sum(x["evaluations"] for x in e["runs"] if str(x["p"]) == str(p))

    def classical(name):
        return {m: {"hit": fair["modes"][m]["rows"][name]["hit_rate"], "p_shot": None, "quality": None, "tuning": 0, "p": None} for m in MODES}

    rows.append(("uniform random sampling", "control", {m: {"hit": fair["modes"][m]["rows"]["p = 0 control (uniform)"]["hit_rate_exact"],
                 "p_shot": 1 / 4096, "quality": results["models"][t4.key_of(m, False)]["baselines"]["p0_control_uniform"]["quality"], "tuning": 0, "p": None} for m in MODES}))
    rows.append(("greedy descent, restarted", "classical", classical("greedy descent (restarted)")))
    rows.append((f"simulated annealing, one {budget}-step run", "classical", classical("simulated annealing (one run)")))
    specs = [("LR-QAOA", "lr", dict(lr=True)), ("free-angle QAOA", "standard", dict(standard=True)),
             ("CVaR-QAOA ($\\alpha{=}0.1$)", "extra", dict(extra_name="cvar_0.1")), ("multi-angle QAOA", "extra", dict(extra_name="multi_angle")),
             ("counterdiabatic QAOA", "extra", dict(extra_name="dc")), ("VarQITE angle training", "extra", dict(extra_name="varqite")),
             ("varQITE Pauli-update~\\cite{varqite}", "extra", dict(extra_name="varqite_pauli"))]
    for label, kind, kwargs in specs:
        per_mode = {}
        for m in MODES:
            p, met, tuning_cost = pick(m, **kwargs)
            per_mode[m] = {"hit": met["p_optimum_in_shots"], "p_shot": met["p_optimum_per_shot"], "quality": met["quality"], "tuning": tuning_cost, "p": p}
        rows.append((label, kind, per_mode))
    return rows


def budget_table(results, fair, path):
    rows = budget_rows(results, fair)
    lines = ["\\begin{tabular}{@{}lrcccccc@{}}\\toprule",
             " & & \\multicolumn{3}{c}{Mode A: long/short} & \\multicolumn{3}{c}{Mode B: buy or skip}\\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             "Method (best $p$ A\\,/\\,B) & tuning & hit@1000 & $P_{\\mathrm{opt}}$/shot & quality & hit@1000 & $P_{\\mathrm{opt}}$/shot & quality\\\\\\midrule"]
    def fmt(v, digits=3):
        return "--" if v is None else (f"{v:.{digits}f}" if v >= 10 ** -digits or v == 0 else f"${{<}}10^{{-{digits}}}$")
    for label, kind, per in rows:
        a, b = per["long_short"], per["buy_or_skip"]
        name = label + (f" ({a['p']}\\,/\\,{b['p']})" if a["p"] is not None else "")
        thousands = lambda v: f"{v:,}".replace(",", "\\,")
        tuning = "0" if a["tuning"] == 0 else thousands(a["tuning"]) + "\\,/\\," + thousands(b["tuning"])
        cells = [name, tuning, fmt(a["hit"]), fmt(a["p_shot"], 4), fmt(a["quality"]), fmt(b["hit"]), fmt(b["p_shot"], 4), fmt(b["quality"])]
        line = " & ".join(cells) + "\\\\"
        if kind == "classical" and label.startswith("simulated"):
            line += "\\midrule"
        lines.append(line)
    lines.append("\\bottomrule\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")
    return rows


def budget_figure(rows, fair):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    colors = {"control": GREY, "classical": BROWN}
    for ax, m in zip(axes, MODES):
        labels, values, cols, notes = [], [], [], []
        for label, kind, per in rows:
            r = per[m]
            labels.append(label.replace("~\\cite{varqite}", "").replace("$\\alpha{=}0.1$", "0.1") + (f" p={r['p']}" if r["p"] else ""))
            values.append(r["hit"])
            cols.append(colors.get(kind, BLUE if label.startswith("LR") else ORANGE if label.startswith("free") else TEAL))
            notes.append(f"{r['hit']:.3f}" + (f" [{r['tuning']:,}]" if r["tuning"] else ""))
        y = np.arange(len(rows))
        ax.barh(y, values, color=cols, height=0.7)
        for yy, v, note in zip(y, values, notes):
            ax.annotate(note, (min(v, 0.98), yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6,
                        ha="left" if v < 0.6 else "right", color="black" if v < 0.6 else "white")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(f"hit@{fair['budget']}")
        ax.set_title(MODE_NAME[m], loc="left")
        ax.grid(True, axis="x", color="#e5e4df", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"budget.{ext}", dpi=220, bbox_inches="tight")


def grid(results):
    """LR-QAOA grid heat-maps for p = 1, 3, 5: the sign finding."""
    g = results["meta"]["tuning"]["lr_grid"]
    dg, db = g["dgamma"], g["dbeta"]
    ps = ("1", "3", "5")
    fig, axes = plt.subplots(2, len(ps), figsize=(7.0, 3.9), sharex=True, sharey=True)
    for row, m in enumerate(MODES):
        lr = results["models"][t4.key_of(m, False)]["lr"]
        for col, p in enumerate(ps):
            e, ax = lr[p], axes[row][col]
            img = ax.imshow(np.array(e["expectation"]).T, origin="lower", aspect="auto", cmap="viridis_r",
                            extent=(dg[0] - 0.05, dg[-1] + 0.05, -1.05, 1.05))
            ax.scatter([e["best_dgamma"]], [e["best_dbeta"]], marker="*", s=110, color="white", edgecolor="black", zorder=3)
            ax.scatter([0.6], [0.3], marker="x", s=45, color="#e34948", linewidth=1.5, zorder=3)
            ax.axhline(0, color="white", linewidth=0.5)
            ax.set_title(f"p = {p}: best ({e['best_dgamma']}, {e['best_dbeta']})", loc="left", fontsize=7.5)
            if row == 1:
                ax.set_xlabel(r"$\Delta\gamma$")
            if col == 0:
                ax.set_ylabel(MODE_NAME[m].replace(": ", "\n") + "\n" + r"$\Delta\beta$")
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, label="expected energy of the sampled portfolios")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"grid.{ext}", dpi=220, bbox_inches="tight")


def varqite_curves(results):
    """Energy per training step of every varQITE (Pauli-update) run."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    styles = {"A": "-", "B": "--"}
    colors = {0.1: BLUE, 0.4: ORANGE, 0.8: TEAL}
    for ax, m in zip(axes, MODES):
        key = t4.key_of(m, False)
        v = results["extras"][key]["varqite_pauli"]
        for run in v["runs"]:
            ax.plot(run["energy_per_step"], linestyle=styles[run["kind"]], color=colors[run["lr"]], linewidth=1.3,
                    label=f"ansatz {run['kind']}, lr {run['lr']} (quality {run['metrics']['quality']:.3f})")
        ax.axhline(results["models"][key]["baselines"]["exact"]["e_min"], color="black", linestyle=":", linewidth=1, label="exact optimum")
        ax.set_xlabel("training step")
        ax.set_ylabel("expected energy")
        ax.set_title(MODE_NAME[m], loc="left")
        ax.grid(True, color="#e5e4df", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [l.split(" (")[0] for l in labels], loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"varqite.{ext}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = t4.load()
    fair = json.loads((OUT.parent.parent / "results" / "stocks_fair_budget_n12.json").read_text())
    ladder(results)
    rows = budget_table(results, fair, OUT.parent / "table_budget.tex")
    budget_figure(rows, fair)
    grid(results)
    varqite_curves(results)
    print("ladder, budget figure and table written; most likely state is the optimum:", cut(json.loads(t4.PARAMETERS_PATH.read_text())))
