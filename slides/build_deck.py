"""Build the ZollerQu stocks slide deck from the result files in this repo.

    ./venv/bin/python slides/build_deck.py            # figures + PPTX + PDF + speaker notes
    ./venv/bin/python slides/build_deck.py --only 6   # one-slide PPTX/PDF in --out (used for render checks)

Outputs (all under slides/): stocks_ZollerQu.pptx, stocks_ZollerQu.pdf, figures/*.png, speaker_notes.md,
slide_numbers.json (every number on the slides with the result file and key it was read from).

Local tool only (not part of the Qollab project): needs python-pptx and pymupdf next to the repo's usual packages
(`./venv/bin/pip install python-pptx pymupdf`); Arial is read from the macOS system fonts.

Nothing is simulated here: numbers are read from results/ and qollab/. The only arithmetic is rounding, ratios, a Clopper-Pearson interval for the measured
hit count, and the median of a geometric distribution (shots until the first hit of a sampler with hit rate p).
The PPTX and the PDF are drawn from one layout description, with every line break fixed here, so both match.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE
from lxml import etree
from pptx.util import Emu, Pt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides"
FIG = OUT / "figures"
PROJECT = "stocks"

# ---------------------------------------------------------------- design tokens
BG, CARD, TINT = "#0A1020", "#131B2E", "#0F2A3A"       # slide surface, card surface, teal-tinted block
WHITE, MUTED = "#FFFFFF", "#A9B4C6"                    # primary and secondary text
TEAL, AMBER = "#17A2B8", "#D9791A"                     # accent / highlight for "our result" (validated pair)
GREY, LIGHT, GRID, WIRE = "#66728A", "#B4BECE", "#232C42", "#3A4560"
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
SLIDE_W, SLIDE_H, MX = 13.333, 7.5, 0.6                # inches
CW = SLIDE_W - 2 * MX
DPI = 200
MIN_PT = 18
FOOTER = "ZollerQu - Qollab x IonQ Hackathon 2026"
ATTRIBUTION = "This effort is supported by Qollab & IonQ"

IONQ_FILE = "results/ionq_B_cvar_p3_20261011-052116.json"
BUDGET_FILE = "results/stocks_fair_budget_n12.json"
TUNING_FILE = "results/stocks_qaoa_tuning_n12.json"
N29_FILE = "results/stocks_qaoa_n29.json"
CANDIDATES_FILE = "qollab/circuit_candidates.json"
EXP29_FILE = "results/stocks_exp29_summary.json"
LINKS = {"Qollab project": None, "Demo video": None}   # set a URL to show that line on the closing slide
CIRCUIT = "B CVaR 0.1 p=3"
MODE = "buy_or_skip"

NAMES = {"AAPL": "Apple", "GLD": "gold", "XOM": "oil", "JPM": "bank", "TLT": "US bonds", "JNJ": "health",
         "PG": "consumer", "XLU": "utilities", "VNQ": "real estate", "FCX": "copper", "MSFT": "Microsoft", "SLV": "silver"}


def load(rel):
    return json.loads((ROOT / rel).read_text())


# ---------------------------------------------------------------- numbers (each with its source)
def collect():
    n, src = {}, {}

    def put(key, value, source):
        n[key], src[key] = value, source

    run, budget, tuning, big, cand = load(IONQ_FILE), load(BUDGET_FILE), load(TUNING_FILE), load(N29_FILE), load(CANDIDATES_FILE)
    summary = run["summary"][0]
    shots = sum(run["counts"].values())
    hits = round(summary["p_optimum"] * shots)
    put("shots", shots, f"{IONQ_FILE}: sum(counts)")
    put("hits", hits, f"{IONQ_FILE}: summary[0].p_optimum x shots")
    put("p_noisy", hits / shots, f"{IONQ_FILE}: summary[0].p_optimum")
    put("within5_noisy", summary["p_within_5pct"], f"{IONQ_FILE}: summary[0].p_within_5pct")
    put("quality_noisy", summary["quality"], f"{IONQ_FILE}: summary[0].quality")
    put("job_id", run["run"]["job_id"], f"{IONQ_FILE}: run.job_id")
    put("backend", run["backend"], f"{IONQ_FILE}: backend")
    lo, hi = stats.beta.ppf(0.025, hits, shots - hits + 1), stats.beta.ppf(0.975, hits + 1, shots - hits)
    put("p_noisy_lo", float(lo), "Clopper-Pearson 95 % interval of hits/shots")
    put("p_noisy_hi", float(hi), "Clopper-Pearson 95 % interval of hits/shots")

    rows = budget["modes"][MODE]["rows"]
    states = 2 ** budget["n"]
    put("qubits", budget["n"], f"{BUDGET_FILE}: n")
    put("states", states, "2^n")
    put("p_random", rows["p = 0 control (uniform)"]["p_optimum_per_shot"], f"{BUDGET_FILE}: modes.{MODE}.rows['p = 0 control (uniform)'].p_optimum_per_shot")
    put("hit1000_random_12", rows["p = 0 control (uniform)"]["hit_rate_exact"], f"{BUDGET_FILE}: ...rows['p = 0 control (uniform)'].hit_rate_exact")
    put("hit1000_greedy_12", rows["greedy descent (restarted)"]["hit_rate"], f"{BUDGET_FILE}: ...rows['greedy descent (restarted)'].hit_rate")
    put("hit1000_sa_12", rows["simulated annealing (one run)"]["hit_rate"], f"{BUDGET_FILE}: ...rows['simulated annealing (one run)'].hit_rate")
    put("training_circuits", rows["CVaR-QAOA (0.1) p=3"]["offline_tuning_circuits"], f"{BUDGET_FILE}: ...rows['CVaR-QAOA (0.1) p=3'].offline_tuning_circuits")
    put("budget", budget["budget"], f"{BUDGET_FILE}: budget")
    put("budget_repetitions", budget["repetitions"], f"{BUDGET_FILE}: repetitions")

    row = next(r for r in cand["rows"] if r["name"] == CIRCUIT)
    put("p_noiseless", row["p_optimum_noiseless"], f"{CANDIDATES_FILE}: rows['{CIRCUIT}'].p_optimum_noiseless")
    put("within5_noiseless", row["p_within_5pct_noiseless"], f"{CANDIDATES_FILE}: rows['{CIRCUIT}'].p_within_5pct_noiseless")
    put("gates_circuit", row["two_qubit_gates"], f"{CANDIDATES_FILE}: rows['{CIRCUIT}'].two_qubit_gates")
    classical = cand["classical_evaluations_to_solution"][MODE]
    put("greedy_median", classical["greedy_median"], f"{CANDIDATES_FILE}: classical_evaluations_to_solution.{MODE}.greedy_median")
    put("sa_median", classical["annealing_median"], f"{CANDIDATES_FILE}: classical_evaluations_to_solution.{MODE}.annealing_median")
    put("classical_runs", classical["runs"], f"{CANDIDATES_FILE}: classical_evaluations_to_solution.{MODE}.runs")
    trained = [r for r in cand["rows"] if r["family"] != "control"]
    put("circuits_compared", len(trained), f"{CANDIDATES_FILE}: rows without the control")
    best12 = max(trained, key=lambda r: r["p_optimum_noiseless"])
    if best12["name"] != CIRCUIT or max(trained, key=lambda r: r["crude_noisy_p_optimum"])["name"] != CIRCUIT:
        raise ValueError(f"the circuit sent to IonQ ({CIRCUIT}) is no longer the best 12-qubit candidate ({best12['name']})")
    put("best12_name", best12["name"], f"{CANDIDATES_FILE}: row with the highest p_optimum_noiseless (also the highest crude noisy estimate)")
    put("selected_other", 4, "qollab/selected_circuits.json: batch has 5 circuits, 1 run so far")

    def median_shots(p):
        return math.log(0.5) / math.log(1.0 - p)

    put("median_random", median_shots(n["p_random"]), "ln(0.5) / ln(1 - p), p = 1/4096")
    put("median_noisy", median_shots(n["p_noisy"]), "ln(0.5) / ln(1 - p), p = hits/shots")
    put("median_noisy_lo", median_shots(n["p_noisy_hi"]), "same, upper end of the 95 % interval of p")
    put("median_noisy_hi", median_shots(n["p_noisy_lo"]), "same, lower end of the 95 % interval of p")
    put("median_noiseless", median_shots(n["p_noiseless"]), "ln(0.5) / ln(1 - p), noiseless p")
    put("ratio_random", n["p_noisy"] / n["p_random"], "p_noisy / p_random")
    put("ratio_random_lo", n["p_noisy_lo"] / n["p_random"], "interval end / p_random")
    put("ratio_random_hi", n["p_noisy_hi"] / n["p_random"], "interval end / p_random")
    put("noise_factor", n["p_noiseless"] / n["p_noisy"], "p_noiseless / p_noisy")

    r12, r29 = tuning["circuits"]["n12"][MODE]["routing_p1"], tuning["circuits"]["large"][MODE]["routing_p1"]
    for tag, r in (("12", r12), ("29", r29)):
        base = f"{TUNING_FILE}: circuits.{'n12' if tag == '12' else 'large'}.{MODE}.routing_p1"
        put(f"gates_all_{tag}", r["all_to_all_two_qubit_gates"], base + ".all_to_all_two_qubit_gates")
        put(f"gates_hh_{tag}", r["heavy_hex_two_qubit_gates"]["median"], base + ".heavy_hex_two_qubit_gates.median")
        put(f"gates_hh_min_{tag}", r["heavy_hex_two_qubit_gates"]["min"], base + ".heavy_hex_two_qubit_gates.min")
        put(f"gates_hh_max_{tag}", r["heavy_hex_two_qubit_gates"]["max"], base + ".heavy_hex_two_qubit_gates.max")
        put(f"gates_ratio_{tag}", r["heavy_hex_two_qubit_gates"]["median"] / r["all_to_all_two_qubit_gates"], "heavy-hex median / all-to-all")
    put("routing_seeds", len(r12["heavy_hex_seeds"]), f"{TUNING_FILE}: ...routing_p1.heavy_hex_seeds")
    put("qiskit", tuning["meta"]["versions"]["qiskit"], f"{TUNING_FILE}: meta.versions.qiskit")
    put("pairs_12", tuning["meta"]["kept_links"], f"{TUNING_FILE}: meta.kept_links")
    put("pairs_29", tuning["circuits"]["large"][MODE]["pair_terms"], f"{TUNING_FILE}: circuits.large.{MODE}.pair_terms")
    put("train_first", tuning["meta"]["training_first"], f"{TUNING_FILE}: meta.training_first")
    put("train_last", tuning["meta"]["training_last"], f"{TUNING_FILE}: meta.training_last")
    put("date_tuning", tuning["meta"]["timestamp_utc"][:10], f"{TUNING_FILE}: meta.timestamp_utc")
    put("date_n29", big["timestamp_utc"][:10], f"{N29_FILE}: timestamp_utc")

    transfer = tuning["transfer"]
    put("scale_qaoa_12", tuning["models"][MODE + "|balance_off"]["lr"]["3"]["best"]["p_optimum_in_shots"], f"{TUNING_FILE}: models['{MODE}|balance_off'].lr['3'].best.p_optimum_in_shots")
    put("scale_qaoa_16", transfer["tuned_size"]["modes"][MODE]["transferred"]["3"]["p_optimum_in_shots"], f"{TUNING_FILE}: transfer.tuned_size.modes.{MODE}.transferred['3'].p_optimum_in_shots")
    put("scale_qaoa_20", transfer["larger_size_dense"]["modes"][MODE]["transferred"]["3"]["p_optimum_in_shots"], f"{TUNING_FILE}: transfer.larger_size_dense.modes.{MODE}.transferred['3'].p_optimum_in_shots")
    put("scale_qaoa_29", big["modes"][MODE]["circuits"]["dense_p3"]["p_optimum_in_shots"], f"{N29_FILE}: modes.{MODE}.circuits.dense_p3.p_optimum_in_shots")
    put("scale_random_16", transfer["tuned_size"]["modes"][MODE]["uniform_p0"]["p_optimum_in_shots"], f"{TUNING_FILE}: transfer.tuned_size.modes.{MODE}.uniform_p0.p_optimum_in_shots")
    put("scale_random_20", transfer["larger_size_dense"]["modes"][MODE]["uniform_p0"]["p_optimum_in_shots"], f"{TUNING_FILE}: transfer.larger_size_dense.modes.{MODE}.uniform_p0.p_optimum_in_shots")
    put("scale_random_29", 1.0 - (1.0 - big["modes"][MODE]["uniform"]["p_optimum"]) ** big["budget"], f"{N29_FILE}: 1 - (1 - modes.{MODE}.uniform.p_optimum)^budget")
    put("scale_greedy_29", big["modes"][MODE]["classical_same_budget"]["greedy descent (restarted)"]["hit_rate"], f"{N29_FILE}: modes.{MODE}.classical_same_budget['greedy descent (restarted)'].hit_rate")
    put("scale_sa_29", big["modes"][MODE]["classical_same_budget"]["simulated annealing (one run)"]["hit_rate"], f"{N29_FILE}: modes.{MODE}.classical_same_budget['simulated annealing (one run)'].hit_rate")
    put("states_29", big["modes"][MODE]["exact_dense"]["states"], f"{N29_FILE}: modes.{MODE}.exact_dense.states")
    runs29 = big["modes"][MODE]["circuits"]
    key29 = max(runs29, key=lambda k: runs29[k]["p_optimum"])
    put("best29_run", key29, f"{N29_FILE}: modes.{MODE}.circuits, entry with the highest p_optimum (all entries are LR-QAOA)")
    put("best29_layers", runs29[key29]["p"], f"{N29_FILE}: modes.{MODE}.circuits.{key29}.p")
    put("best29_hit1000", runs29[key29]["p_optimum_in_shots"], f"{N29_FILE}: modes.{MODE}.circuits.{key29}.p_optimum_in_shots")
    put("n29_repetitions", big["repetitions"], f"{N29_FILE}: repetitions")

    exp = load(EXP29_FILE)
    rows = [r for r in exp["rows"] if r["mode"] == MODE and r["tuning_evaluations_29"] == 0]
    paper = {r["p"]: r["hit@1000"] for r in rows if r["source"] == "s0_baseline_ramp12" or "original grid" in r["label"]}
    wider = {r["p"]: r["hit@1000"] for r in rows if (r["source"] == "s0_baseline_ramp12" and r["p"] <= 4) or "wide grid" in r["label"]
             or ("deep grid" in r["label"] and r["p"] > 16)}
    put("depth_paper", {str(k): v for k, v in sorted(paper.items())}, f"{EXP29_FILE}: rows, ramp tuned at 12 with strength <= 1.5 (paper grid)")
    put("depth_wider", {str(k): v for k, v in sorted(wider.items())}, f"{EXP29_FILE}: rows, ramp tuned at 12 on the wider grids (strength <= 3.0 up to 16 layers, <= 6.0 above)")
    put("gates_layer_29", next(r["two_qubit_gates"] // r["p"] for r in rows), f"{EXP29_FILE}: two_qubit_gates / p")
    put("exp_greedy_1000", exp["classical"]["budgets"]["1000"]["greedy descent (restarted)"]["hit_rate"], f"{EXP29_FILE}: classical.budgets['1000'] greedy")
    put("exp_sa_1000", exp["classical"]["budgets"]["1000"]["simulated annealing (one run)"]["hit_rate"], f"{EXP29_FILE}: classical.budgets['1000'] annealing")
    put("exp_greedy_10000", exp["classical"]["budgets"]["10000"]["greedy descent (restarted)"]["hit_rate"], f"{EXP29_FILE}: classical.budgets['10000'] greedy")
    put("exp_repetitions", exp["classical"]["repetitions"], f"{EXP29_FILE}: classical.repetitions")

    return n, src


def training_stats():
    """Training-window statistics of the 12 assets (the same loader the notebook uses)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from stocks import task3_report
    setup = task3_report.size_setup(task3_report.load_context(), 12)
    return setup["tickers"], np.asarray(setup["stats"].corr, dtype=float)


def exact_optimum():
    """Exact best buy/skip portfolio at 12 assets: all 4,096 states of the Ising model embedded in the circuit selection."""
    selected = load("qollab/selected_circuits.json")
    model = next(c for c in selected["batch"] if c["name"] == CIRCUIT)["model"]
    tickers = selected["qubit_order"]["tickers"]
    count = len(tickers)
    spins = 1.0 - 2.0 * ((np.arange(2 ** count)[:, None] >> np.arange(count)) & 1)      # bit 0 <-> spin +1 (buy)
    energy = model["const"] + spins @ np.asarray(model["fields"], dtype=float)
    for i, j, coupling in model["pair_terms"]:
        energy = energy + coupling * spins[:, int(i)] * spins[:, int(j)]
    best = spins[int(np.argmin(energy))]
    return [ticker for ticker, spin in zip(tickers, best) if spin > 0]


# ---------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": MIN_PT, "text.color": WHITE, "figure.dpi": DPI, "savefig.dpi": DPI,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": WIRE, "axes.linewidth": 1.0, "axes.labelcolor": MUTED, "axes.labelsize": MIN_PT,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 1.0,
        "legend.frameon": False, "legend.fontsize": MIN_PT, "mathtext.fontset": "cm",
    })


def save(fig, name):
    """Save at the exact figure size and fail when a text is smaller than 18 pt or leaves the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_text().strip() or not text.get_visible():
            continue
        if text.get_fontsize() < MIN_PT - 1e-6:
            raise ValueError(f"{name}: text {text.get_text()!r} is {text.get_fontsize()} pt (< {MIN_PT})")
        extent = text.get_window_extent(renderer)
        if extent.x0 < box.x0 - 1 or extent.x1 > box.x1 + 1 or extent.y0 < box.y0 - 1 or extent.y1 > box.y1 + 1:
            raise ValueError(f"{name}: text {text.get_text()!r} leaves the figure ({extent})")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    return f"slides/figures/{name}"


def column(ax, x, width, value, color, radius_px=10):
    """A column with a rounded data end and a square baseline (corner radius in pixels, not data units)."""
    (x0, y0), (x1, y1) = ax.transData.transform([(x - width / 2, 0.0), (x + width / 2, value)])
    r = min(radius_px, (x1 - x0) / 2, max(y1 - y0, 0.0))
    verts = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + r, y1), (x1 - r, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.figure.add_artist(PathPatch(MplPath(verts, codes), transform=None, facecolor=color, edgecolor="none", zorder=3))


def graph_layout(tickers):
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(tickers)) / len(tickers)
    return np.cos(angles), np.sin(angles)


def fig_hero(tickers, corr):
    """Title-slide picture: the 12 assets as nodes, every pair linked, link strength = |correlation|."""
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    xs, ys = graph_layout(tickers)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            weight = abs(corr[i, j])
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=TEAL, lw=0.8 + 3.2 * weight, alpha=0.10 + 0.55 * weight, solid_capstyle="round", zorder=1)
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), 0.105, facecolor=BG, edgecolor="none", zorder=2))
        ax.add_patch(Circle((x, y), 0.085, facecolor=TEAL, edgecolor="none", zorder=3))
    return save(fig, "hero_graph.png")


def fig_graph(tickers, corr, buy):
    """Slide 3: the same graph with tickers; the exact best portfolio (buy) is filled."""
    fig = plt.figure(figsize=(5.5, 4.2))
    ax = fig.add_axes([0, 0.1, 1, 0.9])
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    xs, ys = graph_layout(tickers)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            weight = abs(corr[i, j])
            both = tickers[i] in buy and tickers[j] in buy
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=TEAL if both else LIGHT, lw=3.0 if both else 0.6 + 2.4 * weight,
                    alpha=1.0 if both else 0.10 + 0.45 * weight, solid_capstyle="round", zorder=2 if both else 1)
    for ticker, x, y in zip(tickers, xs, ys):
        chosen = ticker in buy
        ax.add_patch(Circle((x, y), 0.125, facecolor=BG, edgecolor="none", zorder=3))
        ax.add_patch(Circle((x, y), 0.1, facecolor=TEAL if chosen else CARD, edgecolor=TEAL if chosen else LIGHT, lw=1.6, zorder=4))
        ax.text(1.27 * x, (1.3 if abs(x) < 0.3 else 1.2) * y, ticker, ha="center" if abs(x) < 0.3 else ("left" if x > 0 else "right"), va="center",
                fontsize=MIN_PT, color=WHITE, fontweight="bold" if chosen else "normal")
    handles = [Line2D([], [], marker="o", ls="", ms=13, mfc=TEAL, mec=TEAL, label="buy (exact best)"),
               Line2D([], [], marker="o", ls="", ms=13, mfc=CARD, mec=LIGHT, mew=1.6, label="skip")]
    fig.legend(handles=handles, loc="lower center", ncol=2, handletextpad=0.3, columnspacing=1.6, borderaxespad=0.1, labelcolor=MUTED)
    return save(fig, "graph_optimum.png")


def fig_formula_score():
    width, height = 6.2, 1.45
    fig = plt.figure(figsize=(width, height))
    renderer = fig.canvas.get_renderer()
    parts = [(r"$J(x)\;=$", None), (r"$\dfrac{x^{\mathrm{T}}C\,x}{n^{2}\,V_{\mathrm{ref}}}$", "risk"),
             (r"$-\;\eta$", None), (r"$\dfrac{\mu^{\mathrm{T}}x}{n\,\sigma_{\mathrm{ref}}}$", "return")]
    x = 0.05
    for tex, label in parts:
        text = fig.text(x / width, 0.6, tex, fontsize=26, va="center", ha="left", color=WHITE)
        extent = text.get_window_extent(renderer).width / DPI
        if label:
            fig.text((x + extent / 2) / width, 0.11, label, fontsize=MIN_PT, color=MUTED, ha="center", va="center")
        x += extent + 0.16
    return save(fig, "formula_score.png")


def fig_formula_ising():
    fig = plt.figure(figsize=(6.2, 1.0))
    fig.text(0.008, 0.5, r"$E(s)\;=\;\sum_{i<j} J_{ij}\,s_i\,s_j\;+\;\sum_{i} h_i\,s_i\;+\;c$", fontsize=28, va="center", ha="left", color=WHITE)
    return save(fig, "formula_ising.png")


def fig_circuit(n):
    """Slide 4: schematic of the circuit that was sent (drawn in inches, 1 unit = 1 inch)."""
    width, height = CW, 2.6
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    wires = [("q0  AAPL", 1.85), ("q1  GLD", 1.47), ("q2  XOM", 1.09), ("q11  SLV", 0.45)]
    x_start, x_end = 1.75, 10.35
    for label, y in wires:
        ax.plot([x_start, x_end], [y, y], color=WIRE, lw=1.6, zorder=1)
        ax.text(x_start - 0.18, y, label, ha="right", va="center", fontsize=MIN_PT, color=MUTED)
    for dy in (-0.09, 0.0, 0.09):
        ax.add_patch(Circle((1.1, 0.77 + dy), 0.022, facecolor=MUTED, edgecolor="none"))

    def gate(x, y, label, w=0.46, h=0.3):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06", facecolor=CARD, edgecolor=LIGHT, lw=1.3, zorder=3))
        ax.text(x, y - 0.005, label, ha="center", va="center", fontsize=MIN_PT, color=WHITE, fontweight="bold", zorder=4)

    def block(x0, x1, lines):
        ax.add_patch(FancyBboxPatch((x0, 0.2), x1 - x0, 1.9, boxstyle="round,pad=0,rounding_size=0.12", facecolor=TINT, edgecolor=TEAL, lw=1.6, zorder=3))
        top = 1.15 + 0.17 * (len(lines) - 1)
        for k, (text, size, bold) in enumerate(lines):
            ax.text((x0 + x1) / 2, top - 0.34 * k, text, ha="center", va="center", fontsize=size, color=WHITE if bold else MUTED,
                    fontweight="bold" if bold else "normal", zorder=4)

    for _, y in wires:
        gate(2.2, y, "H")
        gate(9.75, y, "M")
    block(2.85, 6.55, [("cost layer", 20, True), (f"one ZZ gate per pair: {n['pairs_12']}", MIN_PT, False), ("one Z rotation per qubit", MIN_PT, False)])
    block(6.85, 9.05, [("mixer", 20, True), ("RX on every qubit", MIN_PT, False)])
    y_bracket = 2.22
    ax.plot([2.85, 2.85, 9.05, 9.05], [y_bracket - 0.07, y_bracket, y_bracket, y_bracket - 0.07], color=LIGHT, lw=1.3)
    ax.text((2.85 + 9.05) / 2, y_bracket + 0.19, "repeated 3 times (3 layers)", ha="center", va="center", fontsize=MIN_PT, color=WHITE)
    ax.text(10.55, 1.3, f"{n['shots']:,} shots", ha="left", va="center", fontsize=MIN_PT, color=WHITE, fontweight="bold")
    ax.text(10.55, 0.96, "= 1,000", ha="left", va="center", fontsize=MIN_PT, color=MUTED)
    ax.text(10.55, 0.66, "portfolios", ha="left", va="center", fontsize=MIN_PT, color=MUTED)
    return save(fig, "circuit_sketch.png")


def fig_gates(n):
    """Slide 5: two-qubit gates of one QAOA layer, all-to-all against heavy-hex routing."""
    fig = plt.figure(figsize=(7.0, 4.15))
    ax = fig.add_axes([0.125, 0.115, 0.86, 0.72])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2250)
    ax.set_yticks([0, 500, 1000, 1500])
    ax.set_yticklabels(["0", "500", "1,000", "1,500"])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"12 qubits · {n['pairs_12']} pairs", f"29 qubits · {n['pairs_29']} pairs"], color=WHITE)
    ax.tick_params(axis="x", pad=8)
    fig.canvas.draw()
    for k, tag in enumerate(("12", "29")):
        for dx, key, color in ((-0.17, "gates_all_", TEAL), (0.17, "gates_hh_", GREY)):
            value = n[key + tag]
            column(ax, k + dx, 0.3, value, color)
            ax.text(k + dx, value + 45, f"{value:,}", ha="center", va="bottom", fontsize=20, fontweight="bold", color=WHITE)
    handles = [Line2D([], [], marker="s", ls="", ms=14, mfc=TEAL, mec=TEAL, label="all-to-all (trapped ions)"),
               Line2D([], [], marker="s", ls="", ms=14, mfc=GREY, mec=GREY, label="heavy-hex grid, routed")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, borderaxespad=0.2, labelcolor=WHITE)
    fig.text(0.008, 0.94, "two-qubit gates in one QAOA layer", fontsize=MIN_PT, color=MUTED, ha="left", va="center")
    return save(fig, "gate_counts.png")


def fig_main(n):
    """Slide 6: evaluations until the best portfolio is first found (median), one unit for every method."""
    rows = [("Noiseless simulation", n["median_noiseless"], TEAL, None),
            ("Greedy (classical)", n["greedy_median"], LIGHT, None),
            ("IonQ Forte 1 noise model", n["median_noisy"], AMBER, (n["median_noisy_lo"], n["median_noisy_hi"])),
            ("Annealing (classical)", n["sa_median"], LIGHT, None),
            ("Random guessing", n["median_random"], GREY, None)]
    fig = plt.figure(figsize=(7.3, 4.15))
    left = 0.44
    ax = fig.add_axes([left, 0.2, 0.975 - left, 0.66])
    ax.set_xscale("log")
    x_min, x_max = 5, 40000
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.minorticks_off()
    for k, (label, value, color, interval) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.text(-0.04, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=MIN_PT, color=WHITE)
        ax.plot([x_min, value], [y, y], color=color, lw=3, alpha=0.55, solid_capstyle="butt", zorder=2)
        end = value
        if interval:
            ax.plot(interval, [y, y], color=WHITE, lw=1.6, zorder=3)
            for edge in interval:
                ax.plot([edge, edge], [y - 0.14, y + 0.14], color=WHITE, lw=1.6, zorder=3)
            end = interval[1]
        ax.plot([value], [y], marker="o", ms=17, mfc=color, mec=BG, mew=2.5, zorder=4)
        ax.text(end * 1.3, y, f"{value:,.0f}", ha="left", va="center", fontsize=20, fontweight="bold", color=WHITE)
    fig.text(0.008, 0.945, "evaluations until the best portfolio is found (median)", fontsize=MIN_PT, color=MUTED, ha="left", va="center")
    fig.text(left + (0.975 - left) / 2, 0.045, "log scale · fewer is better", fontsize=MIN_PT, color=MUTED, ha="center", va="center")
    return save(fig, "main_result.png")


def fig_depth29(n):
    """Slide 7: 29 assets, share of 1,000-shot runs that contain the exact best portfolio, against the number of layers."""
    paper = {int(k): 100 * v for k, v in n["depth_paper"].items()}
    wider = {int(k): 100 * v for k, v in n["depth_wider"].items()}
    classical = 100 * n["exp_greedy_1000"]
    fig = plt.figure(figsize=(7.3, 4.2))
    ax = fig.add_axes([0.1, 0.205, 0.88, 0.665])
    ax.set_xlim(0, 33.5)
    ax.set_ylim(-4, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([1, 3, 8, 12, 16, 20, 24, 32])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"layers ({n['gates_layer_29']} two-qubit gates each)", labelpad=6)
    ax.axhline(classical, color=LIGHT, ls=(0, (6, 4)), lw=2.2, zorder=2)
    ax.text(33.3, classical - 3, f"classical methods\n1,000 evaluations: {classical:.0f} %", ha="right", va="top", fontsize=MIN_PT, color=LIGHT,
            linespacing=1.15)
    for points, color, size, label in ((paper, GREY, 10, "ramp as in the paper"), (wider, TEAL, 12, "ramp tuned on a wider grid")):
        ps = sorted(points)
        ax.plot(ps, [points[p] for p in ps], color=color, lw=3, marker="o", ms=size, mfc=color, mec=BG, mew=2, zorder=4, label=label)
    for p, dx, dy, ha in ((3, 0.4, 7, "center"), (12, -0.8, 4, "right"), (16, -0.8, 3, "right"), (24, 0, 6, "center")):
        value = wider[p]
        ax.text(p + dx, value + dy, f"{value:.1f} %" if value < 10 or 99 < value < 100 else f"{value:.0f} %", ha=ha, va="bottom",
                fontsize=MIN_PT, fontweight="bold", color=WHITE, zorder=6)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.4, borderaxespad=0.0, labelcolor=WHITE, handlelength=1.4,
              frameon=True, facecolor=BG, edgecolor="none", framealpha=1.0)
    fig.text(0.008, 0.945, "29 assets: % of 1,000-shot runs that find the best", fontsize=MIN_PT, color=MUTED, ha="left", va="center")
    return save(fig, "depth29.png")


# ---------------------------------------------------------------- layout (one description, two renderers)
FONTS = {False: pymupdf.Font(fontfile=ARIAL), True: pymupdf.Font(fontfile=ARIAL_BOLD)}
PROBLEMS = []


def text_width(text, size, bold):
    return FONTS[bold].text_length(text, fontsize=size) / 72.0


def as_runs(line, size, bold, color):
    """A line is a string or a list of (text, overrides) runs."""
    if isinstance(line, str):
        line = [(line, {})]
    return [{"text": text, "size": opts.get("size", size), "bold": opts.get("bold", bold), "color": opts.get("color", color)} for text, opts in line]


class Slide:
    def __init__(self, number, kicker, title, notes, source=(), title_size=32, hero=False):
        self.number, self.kicker, self.title, self.notes, self.source, self.title_size = number, kicker, title, notes, list(source), title_size
        self.hero = hero                                # title-slide layout (first and last slide)
        self.items, self.alts = [], []

    def text(self, x, y, w, lines, size=20, color=WHITE, bold=False, align="l", gap=0.0, lead=1.2, name=None):
        """Lines stacked from y downwards; every line must fit the width w. Returns the y below the box."""
        paragraphs, cursor = [], y
        for k, line in enumerate(lines):
            runs = as_runs(line, size, bold, color)
            leading = lead * max(r["size"] for r in runs)
            width = sum(text_width(r["text"], r["size"], r["bold"]) for r in runs)
            for r in runs:
                if r["size"] < MIN_PT:
                    PROBLEMS.append(f"slide {self.number}: {r['text']!r} is {r['size']} pt")
                for ch in r["text"]:
                    if not FONTS[r["bold"]].has_glyph(ord(ch)):
                        PROBLEMS.append(f"slide {self.number}: Arial has no glyph for {ch!r} in {r['text']!r}")
            if width > 0.965 * w:
                PROBLEMS.append(f"slide {self.number}: line too wide ({width:.2f} in > {w:.2f} in): {''.join(r['text'] for r in runs)!r}")
            before = gap * 72 if k else 0.0
            cursor += before / 72
            paragraphs.append({"runs": runs, "leading": leading, "before": before, "top": cursor})
            cursor += leading / 72
        if x < 0.5 - 1e-6 or x + w > SLIDE_W - 0.5 + 1e-6 or cursor > SLIDE_H - 0.25:
            PROBLEMS.append(f"slide {self.number}: text box outside the margins at x={x:.2f}, y={y:.2f}")
        self.items.append({"kind": "text", "x": x, "y": y, "w": w, "h": cursor - y, "align": align, "paragraphs": paragraphs, "name": name})
        return cursor

    def rect(self, x, y, w, h, fill=CARD, radius=0.12, line=None, line_width=1.5, dash=False):
        self.items.append({"kind": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill, "radius": radius, "line": line, "line_width": line_width, "dash": dash})

    def dot(self, x, y, d, fill=TEAL):
        self.items.append({"kind": "oval", "x": x, "y": y, "w": d, "h": d, "fill": fill})

    def image(self, path, x, y, w, h, alt):
        self.items.append({"kind": "image", "path": path, "x": x, "y": y, "w": w, "h": h, "alt": alt})
        self.alts.append((path, alt))

    def bullets(self, x, y, w, items, size=20, gap=0.2, color=WHITE, dot_color=TEAL):
        """Items are lists of lines; a small dot marks the first line of each item."""
        for item in items:
            leading = 1.2 * size / 72
            self.dot(x, y + leading * 0.5 - 0.055, 0.11, dot_color)
            y = self.text(x + 0.3, y, w - 0.3, item, size=size, color=color) + gap
        return y - gap

    def stat(self, x, y, w, value, label_lines, size=60, color=TEAL, label_size=20, label_color=WHITE):
        y = self.text(x, y, w, [value], size=size, bold=True, color=color, lead=1.12)
        return self.text(x, y + 0.02, w, label_lines, size=label_size, color=label_color)

    def pill(self, x, y, w, text, size=20, h=0.62):
        self.rect(x, y, w, h, fill=CARD, radius=h / 2, line=TEAL, line_width=1.25)
        self.text(x, y + (h - 1.2 * size / 72) / 2 - 0.01, w, [text], size=size, align="c")


def frame(slide):
    """Kicker, footer, slide number and source lines: identical positions on every slide."""
    items = []
    keep, slide.items = slide.items, items
    if slide.kicker:
        slide.text(MX, 1.3 if slide.hero else 0.42, CW, [slide.kicker], size=MIN_PT, bold=True, color=TEAL)
    if slide.source:
        slide.text(MX, 6.63 - 0.3 * len(slide.source), CW, slide.source, size=MIN_PT, color=MUTED)
    slide.text(MX, 6.92, 8.0, [FOOTER], size=MIN_PT, color=MUTED)
    slide.text(SLIDE_W - MX - 1.5, 6.92, 1.5, [str(slide.number)], size=MIN_PT, color=MUTED, align="r")
    slide.items = keep
    return items


def title_box(slide):
    lines = slide.title if isinstance(slide.title, list) else [slide.title]
    probe = Slide(slide.number, None, None, None)
    y = 1.72 if slide.hero else 0.8
    probe.text(MX, y, 6.9 if slide.hero else CW, lines, size=slide.title_size, bold=True, lead=1.15)
    return probe.items[0]


def rgb(hex_color):
    return RGBColor.from_string(hex_color.lstrip("#"))


def rgb01(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def emu(inches):
    return Emu(int(round(inches * 914400)))


ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}


def pptx_fill_text(frame_, item):
    frame_.margin_left = frame_.margin_right = frame_.margin_top = frame_.margin_bottom = 0
    frame_.word_wrap = True
    frame_.auto_size = MSO_AUTO_SIZE.NONE
    frame_.vertical_anchor = MSO_ANCHOR.TOP
    for k, paragraph in enumerate(item["paragraphs"]):
        p = frame_.paragraphs[0] if k == 0 else frame_.add_paragraph()
        p.alignment = ALIGN[item["align"]]
        p.line_spacing = Pt(paragraph["leading"])
        p.space_before = Pt(paragraph["before"])
        p.space_after = Pt(0)
        for run in paragraph["runs"]:
            r = p.add_run()
            r.text = run["text"]
            r.font.name = "Arial"
            r.font.size = Pt(run["size"])
            r.font.bold = bool(run["bold"])
            r.font.color.rgb = rgb(run["color"])


def register_notes_master(deck):
    """python-pptx adds the notes master part but not its entry in presentation.xml; without the entry Apple's
    Office importer (Quick Look, Keynote) hangs on the file. PowerPoint reads both forms."""
    ns_p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    root = deck.part._element
    rid = next((rel.rId for rel in deck.part.rels.values() if rel.reltype == RELATIONSHIP_TYPE.NOTES_MASTER), None)
    if rid is None or root.find("{%s}notesMasterIdLst" % ns_p) is not None:
        return
    entry = etree.Element("{%s}notesMasterIdLst" % ns_p)
    etree.SubElement(entry, "{%s}notesMasterId" % ns_p).set("{%s}id" % ns_r, rid)
    root.find("{%s}sldMasterIdLst" % ns_p).addnext(entry)


def render_pptx(slides, path):
    deck = Presentation()
    deck.slide_width, deck.slide_height = emu(SLIDE_W), emu(SLIDE_H)
    deck.core_properties.title = "Portfolio Selection with QAOA - ZollerQu"
    deck.core_properties.author = "Oussama Turki, Tushar Pandey (Team ZollerQu)"
    deck.core_properties.subject = "Qollab x IonQ Hackathon 2026"
    layout = deck.slide_layouts[5]                     # "Title Only": a real title placeholder for screen readers
    for slide in slides:
        page = deck.slides.add_slide(layout)
        page.background.fill.solid()
        page.background.fill.fore_color.rgb = rgb(BG)
        title = title_box(slide)
        holder = page.shapes.title
        holder.left, holder.top, holder.width, holder.height = emu(title["x"]), emu(title["y"]), emu(title["w"]), emu(title["h"])
        pptx_fill_text(holder.text_frame, title)
        for item in slide.items + frame(slide):
            box = (emu(item["x"]), emu(item["y"]), emu(item["w"]), emu(item["h"]))
            if item["kind"] == "text":
                shape = page.shapes.add_textbox(*box)
                pptx_fill_text(shape.text_frame, item)
            elif item["kind"] in ("rect", "oval"):
                rounded = item["kind"] == "rect" and item["radius"] > 0
                kind = MSO_SHAPE.OVAL if item["kind"] == "oval" else (MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE)
                shape = page.shapes.add_shape(kind, *box)
                if rounded:
                    shape.adjustments[0] = min(0.5, item["radius"] / min(item["w"], item["h"]))
                if item["fill"]:
                    shape.fill.solid()
                    shape.fill.fore_color.rgb = rgb(item["fill"])
                else:
                    shape.fill.background()
                if item.get("line"):
                    shape.line.color.rgb = rgb(item["line"])
                    shape.line.width = Pt(item["line_width"])
                    if item["dash"]:
                        shape.line.dash_style = MSO_LINE.DASH
                else:
                    shape.line.fill.background()
                shape.shadow.inherit = False
            elif item["kind"] == "image":
                shape = page.shapes.add_picture(str(ROOT / item["path"]), *box)
                shape._element.nvPicPr.cNvPr.set("descr", item["alt"])
        page.notes_slide.notes_text_frame.text = slide.notes
    register_notes_master(deck)
    deck.save(path)


def pdf_text(page, item):
    for paragraph in item["paragraphs"]:
        width = sum(text_width(r["text"], r["size"], r["bold"]) for r in paragraph["runs"])
        x = item["x"] + {"l": 0.0, "c": (item["w"] - width) / 2, "r": item["w"] - width}[item["align"]]
        baseline = paragraph["top"] * 72 + 0.8 * paragraph["leading"]
        for run in paragraph["runs"]:
            writer = pymupdf.TextWriter(page.rect, color=rgb01(run["color"]))
            writer.append((x * 72, baseline), run["text"], font=FONTS[run["bold"]], fontsize=run["size"])
            writer.write_text(page)
            x += text_width(run["text"], run["size"], run["bold"])


def render_pdf(slides, path):
    doc = pymupdf.open()
    for slide in slides:
        page = doc.new_page(width=SLIDE_W * 72, height=SLIDE_H * 72)
        page.draw_rect(page.rect, color=None, fill=rgb01(BG))
        for item in slide.items:
            rect = pymupdf.Rect(item["x"] * 72, item["y"] * 72, (item["x"] + item["w"]) * 72, (item["y"] + item["h"]) * 72)
            if item["kind"] == "rect":
                page.draw_rect(rect, color=rgb01(item["line"]) if item.get("line") else None, fill=rgb01(item["fill"]) if item["fill"] else None,
                               width=item["line_width"] if item.get("line") else 0, dashes="[6 4] 0" if item["dash"] else None,
                               radius=min(0.5, item["radius"] / min(item["w"], item["h"])) if item["radius"] > 0 else None)
            elif item["kind"] == "oval":
                page.draw_oval(rect, color=None, fill=rgb01(item["fill"]))
            elif item["kind"] == "image":
                page.insert_image(rect, filename=str(ROOT / item["path"]))
        for item in [title_box(slide)] + [i for i in slide.items if i["kind"] == "text"] + frame(slide):
            pdf_text(page, item)
    doc.set_metadata({"title": "Portfolio Selection with QAOA - ZollerQu", "author": "Oussama Turki, Tushar Pandey (Team ZollerQu)",
                      "subject": "Qollab x IonQ Hackathon 2026"})
    doc.subset_fonts()
    doc.save(path, deflate=True, garbage=3)
    doc.close()


# ---------------------------------------------------------------- the slides
def per_thousand(p):
    value = 1000 * p
    return f"{value:.0f}" if value >= 10 or abs(value - round(value)) < 1e-9 else f"{value:.2f}"


def build_slides(n, figures, tickers, buy, corr):
    label_noisy = "IonQ Forte 1 noise model (Qollab)"
    hits, shots = n["hits"], n["shots"]
    ratio = f"{n['ratio_random']:.0f}×"
    lo, hi = math.floor(1000 * n["p_noisy_lo"] + 0.5), math.floor(1000 * n["p_noisy_hi"] + 0.5)
    pair = corr[tickers.index(buy[0]), tickers.index(buy[1])]
    slides = []

    # 1 ---------------------------------------------------------------- title
    s = Slide(1, "TEAM ZOLLERQU", ["Portfolio Selection", "with QAOA"], title_size=52, hero=True, notes=(
        "Hi, we are team ZollerQu. Our question is simple: which stocks should a fund buy? "
        "We turn it into an Ising problem, one qubit per asset, and sample it with QAOA on the IonQ Forte 1 noise model."))
    y = s.text(MX, 3.95, 6.9, ["Which stocks to buy? One qubit per asset,", "sampled on the IonQ Forte 1 noise model."], size=22, color=MUTED)
    s.text(MX, y + 0.4, 6.9, ["Oussama Turki  ·  Tushar Pandey"], size=22, bold=True)
    s.pill(MX, 5.75, 5.6, ATTRIBUTION)
    s.image(figures["hero"], 7.55, 1.0, 5.2, 5.2, "Network picture of the 12 assets: 12 dots on a circle, every pair joined by a line; thicker lines mean the two assets move more strongly together.")
    slides.append(s)

    # 2 ---------------------------------------------------------------- problem
    s = Slide(2, "PROBLEM", "Which assets should a fund buy?", source=[
        f"Data: weekly prices of 12 US stocks and funds, {n['train_first'][:4]}-{n['train_last'][:4]} (Kaggle Huge Stock Market Dataset, CC0)"], notes=(
        "Banks and funds build portfolios every day. They want low risk and a good return. "
        "Risk comes from assets that fall together, so the choices interact. "
        f"Twelve assets give {n['states']:,} portfolios; twenty-nine give over five hundred million."))
    y = s.bullets(MX, 1.85, 5.2, [["Banks and funds build", "portfolios every day"], ["They want low risk", "and a good return"],
                                  ["Risk comes from assets", "that fall together"]], size=22, gap=0.3)
    s.text(MX, 5.02, 5.2, ["Every extra asset", "doubles the choices"], size=22, bold=True)
    chip_w, chip_h, x0, y0, space = 1.5, 0.84, 6.28, 1.8, 0.15
    for k, ticker in enumerate(tickers):
        x, yy = x0 + (k % 4) * (chip_w + space), y0 + (k // 4) * (chip_h + space)
        s.rect(x, yy, chip_w, chip_h, fill=CARD, radius=0.12)
        s.text(x, yy + 0.1, chip_w, [[(ticker, {"size": 20, "bold": True})], [(NAMES[ticker], {"size": MIN_PT, "color": MUTED})]], align="c")
    s.stat(x0, 4.9, 2.5, f"{n['states']:,}", ["portfolios, 12 assets"], size=44, label_size=MIN_PT, label_color=MUTED)
    s.stat(x0 + 2.75, 4.9, 3.7, f"{n['states_29']:,}", ["portfolios, 29 assets"], size=44, label_size=MIN_PT, label_color=MUTED)
    slides.append(s)

    # 3 ---------------------------------------------------------------- formulation
    s = Slide(3, "FORMULATION", "One qubit per asset, one Ising energy", source=[
        f"Exact best portfolio found by checking all {n['states']:,} portfolios (classical); risk and return from {n['train_first'][:4]}-{n['train_last'][:4]} data"], notes=(
        "Each asset becomes one qubit: zero means buy, one means skip. "
        f"Risk minus return becomes an Ising energy with one coupling per pair of assets: {n['pairs_12']} couplings. "
        f"For twelve assets we can check every portfolio, so we know the exact best one: Apple plus long-term Treasury bonds."))
    s.text(MX, 1.75, 6.3, ["Score to minimise: risk minus return"], size=MIN_PT, color=MUTED)
    s.image(figures["score"], MX, 2.05, 6.2, 1.45, "Formula: J of x equals x transposed C x over n squared V ref (the risk), minus eta times mu transposed x over n sigma ref (the return).")
    s.text(MX, 3.58, 6.3, ["The same score as an Ising energy"], size=MIN_PT, color=MUTED)
    s.image(figures["ising"], MX, 3.9, 6.2, 1.0, "Formula: E of s equals the sum over pairs of J i j times s i s j, plus the sum of h i times s i, plus a constant.")
    s.text(MX, 5.05, 6.4, [[("1 qubit = 1 asset:  ", {"bold": True}), ("|0> buy,  |1> skip", {})],
                           [(f"12 qubits  ·  {n['pairs_12']} couplings  ·  {n['states']:,} states", {"color": MUTED})]], size=20, gap=0.05)
    s.image(figures["graph"], 7.25, 1.6, 5.5, 4.2, f"Network of the 12 assets with every pair linked. Two filled dots mark the exact best portfolio: buy {buy[0]} and {buy[1]}; the other ten are skipped.")
    slides.append(s)

    # 4 ---------------------------------------------------------------- quantum method
    s = Slide(4, "QUANTUM METHOD", "QAOA circuits, trained offline, one IonQ job", source=[
        f"12 qubits: {n['circuits_compared']} circuits + a random-guess control, noiseless simulation · 16-29 qubits: server simulation"], notes=(
        "The circuit is QAOA: Hadamard gates, a cost layer with one ZZ gate per pair, then a mixer. "
        f"We compared {n['circuits_compared']} circuits from seven families, all trained offline. "
        "At twelve qubits the best is CVaR-QAOA, which went to IonQ as a single job. "
        "At twenty-nine qubits we use linear-ramp QAOA, with angles reused from twelve."))
    s.image(figures["circuit"], MX, 1.62, CW, 2.6, "Circuit sketch: four of the twelve qubit lines. Each starts with an H gate, passes a cost layer block (one ZZ gate per pair, 66, and one Z rotation per qubit) and a mixer block (RX on every qubit), repeated three times, then is measured; 1,000 shots give 1,000 portfolios.")
    card_w, gap_w, y0, card_h = (CW - 0.5) / 3, 0.25, 4.4, 1.72
    cards = [("COMPARED", "7 method families", WHITE, [f"{n['circuits_compared']} circuits, trained offline", "on a classical simulator"]),
             ("BEST AT 12 QUBITS", "CVaR-QAOA", AMBER, [f"3 layers, {n['gates_circuit']} two-qubit gates,", "sent to IonQ as one job"]),
             ("AT 16, 20, 29 QUBITS", "Linear-ramp QAOA", WHITE, ["Angles reused from 12 qubits,", "no new training"])]
    for k, (label, head, color, body) in enumerate(cards):
        x = MX + k * (card_w + gap_w)
        s.rect(x, y0, card_w, card_h)
        yy = s.text(x + 0.2, y0 + 0.16, card_w - 0.4, [label], size=MIN_PT, bold=True, color=TEAL)
        yy = s.text(x + 0.2, yy + 0.03, card_w - 0.4, [head], size=22, bold=True, color=color)
        s.text(x + 0.2, yy + 0.04, card_w - 0.4, body, size=MIN_PT, color=MUTED)
    slides.append(s)

    # 5 ---------------------------------------------------------------- why IonQ
    s = Slide(5, "WHY IONQ", f"All-to-all links: {n['gates_ratio_12']:.1f}× fewer two-qubit gates", source=[
        f"Gate counts from transpiling only (Qiskit {n['qiskit']}, level 1); heavy-hex = median of {n['routing_seeds']} routing seeds",
        "on a generic heavy-hex map. No circuit was run on a heavy-hex device."], notes=(
        "Why IonQ? Every pair of assets is coupled, and trapped ions link every pair of qubits directly. "
        f"One layer needs {n['gates_all_12']} two-qubit gates. The same layer routed on a heavy-hex grid needs {n['gates_hh_12']}: "
        f"{n['gates_ratio_12']:.1f} times more, and {n['gates_ratio_29']:.1f} times at twenty-nine qubits."))
    s.image(figures["gates"], MX, 1.65, 7.0, 4.15, f"Column chart of two-qubit gates in one QAOA layer. At 12 qubits: {n['gates_all_12']} on all-to-all against {n['gates_hh_12']} on a heavy-hex grid. At 29 qubits: {n['gates_all_29']} against {n['gates_hh_29']:,}.")
    y = s.stat(8.25, 1.7, 4.45, f"{n['gates_ratio_12']:.1f}×", ["fewer two-qubit gates", f"at 12 qubits ({n['gates_all_12']} vs {n['gates_hh_12']})"])
    y = s.stat(8.25, y + 0.22, 4.45, f"{n['gates_ratio_29']:.1f}×", [f"fewer at 29 qubits ({n['gates_all_29']} vs {n['gates_hh_29']:,})"])
    s.text(8.25, y + 0.22, 4.45, [f"{n['pairs_12']} pairs = {n['gates_all_12']} gates, no SWAPs"], size=MIN_PT, color=MUTED)
    slides.append(s)

    # 6 ---------------------------------------------------------------- main result
    s = Slide(6, "MAIN RESULT", f"IonQ noise model: {ratio} random guessing", source=[
        f"{label_noisy}: 1 job, {shots:,} shots, 11 Oct 2026 · noiseless simulation",
        f"Classical: median of {n['classical_runs']} runs. Sampler medians come from the hit rate per shot. Training not counted."], notes=(
        f"The main result, on the IonQ Forte 1 noise model: {hits} of {shots:,} shots were the exact best portfolio, "
        f"{n['ratio_random']:.0f} times more than random guessing. "
        f"The noiseless simulation gives {per_thousand(n['p_noiseless'])}, so noise costs a factor of {n['noise_factor']:.0f}. "
        "Greedy search, a classical method, is in the same range as our noisy run."))
    s.rect(MX, 1.7, 4.45, 4.15)
    s.text(MX + 0.3, 1.92, 3.9, [label_noisy.replace(" (Qollab)", "")], size=MIN_PT, color=MUTED)
    y = s.stat(MX + 0.3, 2.28, 3.9, f"{hits} / {shots:,}", ["shots are the exact best", f"of {n['states']:,} portfolios"], size=54, color=AMBER)
    y = s.text(MX + 0.3, y + 0.22, 3.9, [[(ratio + "  ", {"size": 28, "bold": True}), ("random guessing", {"size": MIN_PT, "color": MUTED})],
                                         [(f"{per_thousand(n['p_noiseless'])} / 1,000  ", {"size": 28, "bold": True}), ("noiseless", {"size": MIN_PT, "color": MUTED})]], gap=0.04)
    s.text(MX + 0.3, y + 0.1, 3.9, [f"95 % interval: {lo}-{hi} per 1,000"], size=MIN_PT, color=MUTED)
    s.image(figures["main"], 5.43, 1.7, 7.3, 4.15, (
        f"Dot chart on a log scale: evaluations until the best portfolio is first found, median, fewer is better. Noiseless simulation of the same circuit {n['median_noiseless']:.0f}; "
        f"greedy descent {n['greedy_median']:.0f}; IonQ Forte 1 noise model {n['median_noisy']:.0f} with a 95 % interval from {n['median_noisy_lo']:.0f} to {n['median_noisy_hi']:.0f}; "
        f"simulated annealing {n['sa_median']:,.0f}; random guessing {n['median_random']:,.0f}."))
    slides.append(s)

    # 7 ---------------------------------------------------------------- scaling + honest comparison
    w = {int(k): v for k, v in n["depth_wider"].items()}
    gates16 = 16 * n["gates_layer_29"]
    s = Slide(7, "SCALING AND HONEST COMPARISON", "29 qubits: more layers help, only without noise", source=[
        f"Server simulation of all {n['states_29']:,} states, 11 Oct 2026; ramp tuned at 12 assets, no tuning at 29",
        f"Classical: greedy descent and annealing, {n['exp_repetitions']} runs, the same {n['budget']:,} evaluations"], notes=(
        f"At twenty-nine assets the three-layer circuit finds the best portfolio in {100 * w[3]:.0f} percent of runs, classical methods in {100 * n['exp_greedy_1000']:.0f}. "
        f"With more layers, the noiseless circuit reaches {100 * w[16]:.0f} percent at sixteen. "
        "But that is six and a half thousand two-qubit gates, far beyond today's noise. No advantage is claimed."))
    s.image(figures["depth"], MX, 1.65, 7.3, 4.2, (
        f"Line chart for 29 assets: share of 1,000-shot runs that contain the exact best portfolio, against the number of layers, in noiseless simulation. "
        f"With the ramp tuned on a wider grid: {100 * w[3]:.1f} % at 3 layers, {100 * w[12]:.0f} % at 12, {100 * w[16]:.0f} % at 16, {100 * w[24]:.1f} % at 24 and "
        f"{100 * w[32]:.0f} % at 32. With the ramp as in the paper: {100 * float(n['depth_paper']['16']):.0f} % at 16 layers. "
        f"Classical methods with 1,000 evaluations: {100 * n['exp_greedy_1000']:.0f} %."))
    y = s.text(8.3, 1.72, 4.43, ["Read with care"], size=22, bold=True)
    s.bullets(8.3, y + 0.18, 4.43, [["Noiseless simulation only"],
                                    [f"16 layers = {gates16:,} two-qubit", f"gates; our {n['gates_circuit']}-gate IonQ run", f"lost a factor {n['noise_factor']:.0f} to noise"],
                                    [f"Classical: {100 * n['exp_greedy_10000']:.0f} % with 10,000", "evaluations"],
                                    ["A shot is not the same work", "as a classical evaluation"]], size=MIN_PT, gap=0.17, color=WHITE)
    slides.append(s)

    # 8 ---------------------------------------------------------------- demo, limits, next steps
    s = Slide(8, "DEMO, LIMITS, NEXT STEPS", "Try it on Qollab: one click, one IonQ job", notes=(
        "[CUT TO LIVE QOLLAB DEMO] Here is the project on Qollab, with the code in main.py. "
        "We press Run, select the IonQ Forte 1 noise model, and one job returns counts and plots. "
        "[BACK TO SLIDE] Our limits: a noisy simulator, one job, and strong classical baselines. "
        "Next: other circuits, shorter circuits, and the depth noise allows."))
    s.rect(MX, 1.7, 5.6, 4.15)
    y = s.text(MX + 0.3, 1.95, 5.0, ["Live demo on Qollab"], size=24, bold=True) + 0.32
    steps = [["Open the project:", "the code is in main.py"], ["Run → Select QPU →", "IonQ Forte 1 (29q) noise model"], ["One job returns the counts,", "the table and the plots"]]
    for k, lines in enumerate(steps, 1):
        s.dot(MX + 0.3, y + 0.09, 0.5, TEAL)
        s.text(MX + 0.3, y + 0.09 + (0.5 - 0.3) / 2 - 0.01, 0.5, [str(k)], size=MIN_PT, bold=True, align="c")
        y = s.text(MX + 1.05, y, 4.4, lines, size=20) + 0.4
    y = s.text(6.75, 1.7, 5.98, ["Honest limits"], size=22, bold=True)
    y = s.bullets(6.75, y + 0.16, 5.98, [[f"A noisy simulator: one job of {shots:,} shots"],
                                         [f"Noise cuts the hits from {per_thousand(n['p_noiseless'])} to {hits} per 1,000"],
                                         ["Classical heuristics match or beat it"]], size=MIN_PT, gap=0.14)
    y = s.text(6.75, y + 0.42, 5.98, ["Next steps"], size=22, bold=True)
    s.bullets(6.75, y + 0.16, 5.98, [[f"Run the other {n['selected_other']} selected circuits"],
                                     [f"Shorter circuits: {n['gates_all_12']} instead of {n['gates_circuit']} gates"],
                                     ["Find how many layers the noise allows"]], size=MIN_PT, gap=0.14)
    slides.append(s)

    # 9 ---------------------------------------------------------------- closing
    s = Slide(9, "TEAM ZOLLERQU", ["Thank you for", "your attention"], title_size=52, hero=True, notes=(
        "Thank you for your attention. This effort is supported by Qollab and IonQ."))
    y = s.text(MX, 3.95, 6.9, ["Oussama Turki  ·  Tushar Pandey"], size=22, bold=True)
    closing = ["Portfolio Selection with QAOA"] + [f"{label}: {url}" for label, url in LINKS.items() if url]
    s.text(MX, y + 0.2, 6.9, closing, size=20, color=MUTED, gap=0.04)
    s.pill(MX, 5.75, 5.6, ATTRIBUTION)
    s.image(figures["hero"], 7.55, 1.0, 5.2, 5.2, "The network picture of the 12 assets from the title slide: 12 dots on a circle, every pair joined by a line.")
    slides.append(s)
    return slides, {"buy": buy, "pair_correlation": float(pair)}


# ---------------------------------------------------------------- speaker notes
def word_count(text):
    spoken = text.replace("[CUT TO LIVE QOLLAB DEMO]", "").replace("[BACK TO SLIDE]", "")
    return len(spoken.split())


def clock(minutes):
    return f"{int(minutes)}:{round((minutes - int(minutes)) * 60):02d}"


def write_notes(slides, figure_sources, pending, unverified):
    total = sum(word_count(s.notes) for s in slides)
    lines = ["# Speaker notes: Portfolio Selection with QAOA (ZollerQu, stocks project)", "",
             f"Spoken script for all {len(slides)} slides: **{total} words** (about {clock(total / 135)}-{clock(total / 130)} at 130-135 words per minute; target 330-380 words).", "",
             "Wording rules used: \"noisy simulator\" / \"noise model\" for the IonQ run, no advantage or speed-up claim, no wall-clock comparison.", ""]
    for s in slides:
        title = " ".join(s.title) if isinstance(s.title, list) else s.title
        lines += [f"## Slide {s.number}: {title}", ""]
        if s.number == 8:
            lines += ["**Cut to the live Qollab demo at the marker below, then come back to the slide.**", ""]
        lines += [f"**Script ({word_count(s.notes)} words):** {s.notes}", ""]
        for path, alt in s.alts:
            lines += [f"**Alt text** (`{path}`): {alt}", ""]
        if s.source:
            lines += ["**Source line on the slide:** " + " / ".join(s.source), ""]
    lines += ["## Figures and the result files behind them", "", "| Figure | Used on | Result files |", "|---|---|---|"]
    lines += [f"| `{path}` | slide {slide} | {source} |" for path, slide, source in figure_sources]
    lines += ["", "## Open points", ""] + [f"- {p}" for p in pending]
    lines += ["", "## Caveats and things not verified from the files", ""] + [f"- {u}" for u in unverified]
    lines += ["", "Rebuild everything with `./venv/bin/python slides/build_deck.py` (reads `results/` and `qollab/`; nothing is simulated)."]
    (OUT / "speaker_notes.md").write_text("\n".join(lines) + "\n")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, default=None, help="write a one-slide PPTX/PDF for render checks")
    parser.add_argument("--out", default=None, help="output folder for --only")
    args = parser.parse_args()

    n, sources = collect()
    tickers, corr = training_stats()
    buy = exact_optimum()
    style()
    figures = {"hero": fig_hero(tickers, corr), "graph": fig_graph(tickers, corr, buy), "score": fig_formula_score(), "ising": fig_formula_ising(),
               "circuit": fig_circuit(n), "gates": fig_gates(n), "main": fig_main(n), "depth": fig_depth29(n)}
    slides, extra = build_slides(n, figures, tickers, buy, corr)
    for slide in slides:                                # run the layout checks of the frame and the title too
        frame(slide)
        title_box(slide)
    if PROBLEMS:
        raise SystemExit("layout problems:\n  " + "\n  ".join(dict.fromkeys(PROBLEMS)))

    if args.only:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        chosen = [s for s in slides if s.number == args.only]
        render_pptx(chosen, out / f"slide{args.only}.pptx")
        render_pdf(chosen, out / f"slide{args.only}.pdf")
        return

    render_pptx(slides, OUT / f"{PROJECT}_ZollerQu.pptx")
    render_pdf(slides, OUT / f"{PROJECT}_ZollerQu.pdf")
    data_note = "data/kaggle_weekly_29.csv (training-window correlations, through stocks.task3_report)"
    figure_sources = [
        (figures["hero"], 1, data_note),
        (figures["score"], 3, "no data: the score definition in stocks/ising.py (mode B, buy or skip)"),
        (figures["ising"], 3, "no data: the Ising form in stocks/ising.py"),
        (figures["graph"], 3, f"{data_note}; exact best portfolio from the Ising model in qollab/selected_circuits.json (all 4,096 states)"),
        (figures["circuit"], 4, f"schematic; pair count from {TUNING_FILE} (meta.kept_links), shots from {IONQ_FILE}"),
        (figures["gates"], 5, f"{TUNING_FILE} (circuits.n12 and circuits.large, routing_p1)"),
        (figures["main"], 6, f"{IONQ_FILE} (IonQ run), {CANDIDATES_FILE} (noiseless rate, classical medians), {BUDGET_FILE} (random rate)"),
        (figures["depth"], 7, f"{EXP29_FILE} (29-asset depth experiment and its classical baselines)"),
    ]
    missing = [label for label, url in LINKS.items() if not url]
    pending = ["No placeholder is left on the slides. Slide 8 is shown together with a live Qollab session (no screenshot)."]
    if missing:
        pending.append(f"Links are not on the slides ({', '.join(missing)}): set LINKS in slides/build_deck.py to print them on the closing slide. "
                       "The GitHub repository is private and is not linked.")
    unverified = [
        f"Slide 4, best circuit per size: at 12 qubits CVaR-QAOA (3 layers) has the highest hit rate of all {n['circuits_compared']} circuits. At 16, 20 and 29 qubits "
        "only linear-ramp QAOA was simulated, so no other family was compared there. "
        f"Its best 29-qubit run has {n['best29_layers']} layers ({100 * n['best29_hit1000']:.1f} % of runs find the best portfolio, run '{n['best29_run']}'); "
        f"slide 7 shows the 3-layer run ({100 * n['scale_qaoa_29']:.1f} %) so that every size uses the same settings.",
        f"The stocks run file records the backend only as '{n['backend']}'. That the IonQ Forte 1 noise model was the simulator selected in the Qollab "
        "Run dialog is taken from the team, not from the file (the measured rate is far below the noiseless one, so a noise model was active).",
        "One IonQ job of 1,000 shots: no repeat run, so run-to-run variation on the noise model is not measured; the 95 % interval covers sampling error only.",
        "Heavy-hex gate counts come from Qiskit's router on a generic heavy-hex map (5 seeds); a better router or a real device layout could give other numbers.",
        "The medians for the three samplers on slide 6 are computed from the per-shot hit rate (geometric distribution), not measured by repeating runs.",
        "The PPTX was rendered with macOS Quick Look (slide by slide) and compared with the PDF; it was not opened in PowerPoint itself.",
    ]
    total = write_notes(slides, figure_sources, pending, unverified)
    record = {"note": "every number shown on the slides, with the file and key it was read from (or the formula)", "pair_of_best_portfolio": extra,
              "numbers": {key: {"value": n[key], "source": sources[key]} for key in n}}
    (OUT / "slide_numbers.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
    print(f"slides: {len(slides)} | spoken words: {total} | best portfolio: buy {', '.join(buy)} (correlation {extra['pair_correlation']:+.2f})")
    for s in slides:
        print(f"  {s.number}: {' '.join(s.title) if isinstance(s.title, list) else s.title}  [{word_count(s.notes)} words]")


if __name__ == "__main__":
    main()
