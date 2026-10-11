"""Collect the 29-asset experiment (results/exp29/out_*.jsonl + classical baselines) into one table.

    python -m stocks.exp29_report        # prints the table, writes results/stocks_exp29_summary.json

Every row is one circuit (a fixed set of angles) at 29 assets, graded the same way: P(optimum) per shot, the
probability that B evaluations contain the optimum (hit@B), and the expected quality of the best of B shots.
`tuning_evaluations_29` counts the 29-qubit circuit evaluations spent to choose the angles (0 for transferred ramps).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "exp29"
CLASSICAL = ROOT / "results" / "stocks_exp29_classical_buy_or_skip.json"
SUMMARY = ROOT / "results" / "stocks_exp29_summary.json"
KEYS = ("quality", "p_optimum", "hit@100", "hit@1000", "hit@10000", "hit@100000", "best_quality@1000", "cvar_0.1", "expectation", "seconds")


def load():
    records = {}
    for path in sorted(RUNS.glob("out_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            records[record["id"]] = record
    return records


def row(factor, label, metrics, tuning_evaluations, selected_by, source):
    out = {"factor": factor, "label": label, "mode": "long_short" if "[long_short]" in label else "buy_or_skip", "p": metrics["p"], "two_qubit_gates": metrics["two_qubit_gates"],
           "tuning_evaluations_29": tuning_evaluations, "selected_by": selected_by, "source": source}
    out.update({key: metrics.get(key) for key in KEYS})
    out["angles"] = {"gammas": metrics.get("gammas"), "betas": metrics.get("betas")}
    return out


def table(records):
    rows = []
    for rid, record in records.items():
        if record["problem"]["n"] != 29:
            continue
        job, kind, mode = record["job"], record["problem"]["kind"], record["problem"]["mode"]
        tag = ("" if kind == "dense" else f" [{kind} cost layer]") + ("" if mode == "buy_or_skip" else f" [{mode}]")
        if job["type"] == "eval":
            for metrics in record["results"]:
                label = metrics.get("label") or ("ramp tuned at 12 (as used in the paper), p=%d" % metrics["p"] if rid == "s0_baseline_ramp12" else job.get("label") or rid)
                rows.append(row(job.get("factor", ""), f"{label}{tag}", metrics, int(metrics.get("tuning_evaluations_29", 0)), metrics.get("selected_by", "transferred, not tuned at 29"), rid))
        elif job["type"] == "grid":
            base = records.get(rid[:-4]) if rid.endswith("_ext") else None
            if base:   # an extension of an earlier grid: its points include the earlier choice, so the pick is over both grids
                record = dict(record, evaluations=record["evaluations"] + base["evaluations"])
                tag += " (picked over both grids)"
            best = record["selected"]["by_expectation"]
            rows.append(row(job.get("factor", ""), f"ramp grid at 29, lowest expectation ({best['dgamma']}, {best['dbeta']}){tag}", best, record["evaluations"], "expected energy", rid))
            cvar = record["selected"]["by_cvar_0.1"]
            rows.append(row(job.get("factor", ""), f"ramp grid at 29, lowest CVaR 0.1 ({cvar['dgamma']}, {cvar['dbeta']}){tag}", cvar, record["evaluations"], "CVaR 0.1", rid))
            oracle = next(v for k, v in record["selected"].items() if k.startswith("oracle"))
            rows.append(row(job.get("factor", ""), f"ramp grid at 29, ORACLE best P(opt) ({oracle['dgamma']}, {oracle['dbeta']}){tag}", oracle, record["evaluations"],
                            "known optimum (not usable in practice)", rid))
        elif job["type"] == "opt" and record["selected"]:
            start = job["start"]
            name = "random start, seed %s" % start["random_seed"] if "random_seed" in start else start.get("name", "ramp (%s, %s)" % (start.get("dgamma"), start.get("dbeta")))
            label = f"{job['optimizer']}, {name}, objective {job['objective']}, seed {job.get('seed')}{tag}"
            rows.append(row(job.get("factor", ""), label, record["selected"], record["evaluations"], job["objective"], rid))
            rows[-1]["start"] = {key: record["trajectory"][0][key] for key in ("quality", "p_optimum", "hit@1000")}
    return rows


def ramp_table(records):
    """Ramp chosen (lowest expected energy) on the linear-ramp grid at the smaller sizes, per mode, size and depth."""
    out = {}
    for rid, record in records.items():
        if record["job"]["type"] != "grid" or record["problem"]["n"] == 29:
            continue
        best = record["selected"]["by_expectation"]
        key = f"{record['problem']['mode']}|n{record['problem']['n']}"
        out.setdefault(key, {})[f"p{record['job']['p']}|{rid}"] = {"dgamma": best["dgamma"], "dbeta": best["dbeta"], "quality": best["quality"],
                                                                     "p_optimum": best["p_optimum"], "grid_edge": record["best_on_grid_edge"],
                                                                     "dgamma_max": max(record["job"]["dgammas"])}
    return out


def figure(rows, classical, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = {"ramp tuned at 12, dgamma <= 1.5 (as in the paper)": {}, "ramp tuned at 12, dgamma <= 3.0": {}, "ramp tuned at 12, dgamma <= 6.0": {}}
    names = list(series)
    for r in rows:
        if r["mode"] != "buy_or_skip" or r["factor"] not in ("baseline", "depth"):
            continue
        if r["source"] == "s0_baseline_ramp12":
            series[names[0]][r["p"]] = r["hit@1000"]
            if r["p"] <= 4:                       # the wider grids pick the same ramp for p <= 4
                series[names[1]][r["p"]] = r["hit@1000"]
        elif "original grid" in r["label"]:
            series[names[0]][r["p"]] = r["hit@1000"]
        elif "wide grid" in r["label"]:
            series[names[1]][r["p"]] = r["hit@1000"]
        elif "deep grid" in r["label"]:
            series[names[2]][r["p"]] = r["hit@1000"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (name, points), color, marker in zip(series.items(), ("#66728A", "#17A2B8", "#0B5C6B"), ("s", "o", "D")):
        ps = sorted(points)
        ax.plot(ps, [100 * points[p] for p in ps], marker=marker, color=color, lw=2, label=f"QAOA, noiseless: {name}")
    if classical:
        block = classical["budgets"]["1000"]
        greedy, annealing = block["greedy descent (restarted)"]["hit_rate"], block["simulated annealing (one run)"]["hit_rate"]
        ax.axhline(100 * greedy, color="#D9791A", ls="--", lw=1.6,
                   label=f"classical, 1,000 evaluations: greedy {100 * greedy:.0f} %, annealing {100 * annealing:.0f} %")
        if annealing != greedy:
            ax.axhline(100 * annealing, color="#D9791A", ls=":", lw=1.6)
    ax.set_xlabel("QAOA layers p (two-qubit gates = 406 x p)")
    ax.set_ylabel("% of runs that find the exact best portfolio\nwithin 1,000 shots / evaluations")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("29 assets (buy or skip): exact noiseless simulation of all 536,870,912 states", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pct(x):
    if x is None:
        return "n/a"
    if x >= 1.0 - 1e-12:
        return "100 %"
    return f"{100 * x:.1f} %" if x < 0.995 else (f"{100 * x:.2f} %" if x < 0.9999 else "> 99.99 %")


def markdown_tables(rows, classical, classical_ls):
    """The tables of results/exp29/README.md, generated from the rows (no hand-copied numbers)."""
    head = "| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |\n|---|---|---|---|---|---|---|---|"
    def line(r):
        return (f"| {r['label']} | {r['p']} | {r['two_qubit_gates']:,} | {r['tuning_evaluations_29']} | {r['p_optimum']:.2e} | {pct(r['hit@1000'])} | "
                f"{pct(r['hit@10000'])} | {r['quality']:.4f} |")
    groups = [("Baseline: the ramp tuned at 12 assets, as in the paper", lambda r: r["factor"] == "baseline"),
              ("Factor 1: ramp re-tuned at 29 assets (3 layers)", lambda r: r["factor"] == "ramp re-tuned at 29"),
              ("Factor 2: optimiser over all 6 angles (3 layers, 40 circuits, start = ramp tuned at 12)", lambda r: r["factor"] == "optimiser"),
              ("Factor 3: tuning objective", lambda r: r["factor"] == "objective"),
              ("Factor 4: finite shots per tuning circuit", lambda r: r["factor"] == "shots"),
              ("Factor 5: initialisation (random start instead of the ramp)", lambda r: r["factor"] == "initialisation"),
              ("Factor 6: depth (ramp tuned at 12 assets, no tuning at 29)", lambda r: r["factor"] == "depth" and r["mode"] == "buy_or_skip"),
              ("Combination: 16 layers + ramp re-tuned at 29", lambda r: r["factor"].startswith("combination")),
              ("Robustness: the long/short problem (ramp tuned at 12, no tuning at 29)", lambda r: r["mode"] == "long_short")]
    out = []
    for title, keep in groups:
        chosen = sorted((r for r in rows if keep(r)), key=lambda r: (r["p"], r["label"]))
        if chosen:
            out += [f"### {title}", "", head] + [line(r) for r in chosen] + [""]
    for name, block in (("buy or skip", classical), ("long/short", classical_ls)):
        if block:
            out += [f"### Classical baselines, {name} (200 runs per budget)", "", "| Budget (evaluations) | Random sampling | Greedy descent (restarted) | Simulated annealing (one run) |", "|---|---|---|---|"]
            for budget, b in block["budgets"].items():
                out.append(f"| {int(budget):,} | {pct(b['random sampling']['hit_rate'])} | {pct(b['greedy descent (restarted)']['hit_rate'])} | {pct(b['simulated annealing (one run)']['hit_rate'])} |")
            out.append("")
    return "\n".join(out)


def main():
    records = load()
    rows = table(records)
    classical = json.loads(CLASSICAL.read_text()) if CLASSICAL.exists() else None
    SUMMARY.write_text(json.dumps({"note": "29 assets, buy_or_skip, eta 1, exact noiseless statevector; see stocks/exp29.py", "rows": rows, "classical": classical,
                                   "ramps_at_smaller_sizes": ramp_table(records)}, indent=1) + "\n")
    figure(rows, classical, ROOT / "results" / "figures" / "exp29_depth.png")
    ls_path = ROOT / "results" / "stocks_exp29_classical_long_short.json"
    (RUNS / "tables.md").write_text(markdown_tables(rows, classical, json.loads(ls_path.read_text()) if ls_path.exists() else None) + "\n")
    print(f"{'factor':<22} {'circuit':<78} {'p':>2} {'2q':>5} {'tune':>4} {'quality':>7} {'P(opt)':>9} {'hit@1k':>7} {'hit@10k':>7} {'best@1k':>8}")
    for r in sorted(rows, key=lambda r: (r["mode"], r["factor"], r["p"], -r["hit@1000"])):
        print(f"{r['factor'][:22]:<22} {r['label'][:78]:<78} {r['p']:>2} {r['two_qubit_gates']:>5} {r['tuning_evaluations_29']:>4} {r['quality']:7.4f} {r['p_optimum']:9.2e} "
              f"{r['hit@1000']:7.3f} {r['hit@10000'] if r['hit@10000'] is not None else float('nan'):7.3f} {r['best_quality@1000'] if r['best_quality@1000'] is not None else float('nan'):8.5f}")
    if classical:
        for budget, block in classical["budgets"].items():
            print(f"classical, budget {budget:>6}: " + "; ".join(f"{name}: hit {v['hit_rate']:.3f}, best quality {v['mean_best_quality']:.5f}" for name, v in block.items() if isinstance(v, dict)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
