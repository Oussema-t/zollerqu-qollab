"""Readable table of every IonQ candidate (from results/ionq_selection/candidates_table.csv) -> candidates_table.md.
usage: python scripts/ionq_selection_table.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ionq_selection"
rows = list(csv.DictReader(open(OUT / "candidates_table.csv")))
summary = json.loads((OUT / "selection_summary.json").read_text())
selected = {(s["scenario"], s["circuit"]) for s in summary["selected"]}
fallback = (summary["fallback"]["scenario"], summary["fallback"]["circuit"])


def num(value, digits=4):
    if value in ("", "None", None):
        return "–"
    value = float(value)
    if value == float("inf"):
        return "∞"
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}" if value >= 1e-3 else f"{value:.1e}"


lines = ["# IonQ circuit candidates (case 2, 12 qubits, seed 7)", "",
         "Existing 12-qubit results only (no new training). Columns:",
         "- **P(opt)** and **P(≤5 %)**: noiseless probability per shot of the optimum and of a split within 5 % of the optimum.",
         "- **2q**: two-qubit gates measured by transpiling to IonQ Forte 1 (QIS gate set as submitted; rzz and cx = 1 native gate each).",
         "- **native ZZ**: qiskit-ionq's local native compile, which splits each rzz into 2 ZZ (upper bound; IonQ's own compiler maps rzz to one ZZ).",
         "- **crude**: noiseless P(opt) × 0.995^(2q) (*crude estimate*).",
         "- **Aer**: density matrix with 0.5 % two-qubit and 0.05 % one-qubit depolarizing noise and 0.5 % readout flips (*approximate noise model, not IonQ*; shortlist only).",
         "- **shots to solution** = 1 / P(opt) (noiseless / crude).",
         "- ★ = in the batch, ◆ = fallback, † = starts from a classical answer (warm start). No circuit uses a generic state preparation; the starts are H on every qubit or one RY per qubit.", ""]
for scenario, info in summary["classical"].items():
    uniform = summary["uniform_p_opt"][scenario]
    share = summary["target_share_p_opt"][scenario]
    lines += [f"## {scenario}", "",
              f"Classical samples to solution (median over {info['simulated annealing']['runs']} runs, evaluations until the optimum first appears): "
              f"simulated annealing **{info['simulated annealing']['median']:,.0f}**, local search **{info['local search (restarts)']['median']:,.0f}**. "
              f"Random guessing needs 1 / {uniform:.2e} = {1 / uniform:,.0f} samples; the target-share sampler needs {1 / share:,.0f}.", "",
              "| | circuit | 2q | depth (2q depth) | native ZZ | P(opt) | P(≤5 %) | crude P(opt) | Aer P(opt) | shots to solution (noiseless / crude) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    scenario_rows = sorted([r for r in rows if r["scenario"] == scenario], key=lambda r: -float(r["crude_p_opt"]))
    for r in scenario_rows:
        mark = ("★" if (scenario, r["circuit"]) in selected or (r["circuit"] == "p = 0 control (uniform)" and ("both scenarios", r["circuit"]) in selected) else "") \
            + ("◆" if (scenario, r["circuit"]) == fallback else "") + ("†" if r["seeded_by"] else "")
        over = " (>200)" if int(r["two_qubit_gates"]) > 200 else ""
        lines.append(f"| {mark} | {r['circuit']} | {r['two_qubit_gates']}{over} | {r['depth']} ({r['two_qubit_depth']}) | {r['native_zz_upper_bound']} | "
                     f"{num(r['p_opt'])} | {num(r['p_near'])} | {num(r['crude_p_opt'])} | {num(r['aer_p_opt'])} | "
                     f"{num(r['shots_to_solution_noiseless'], 0)} / {num(r['shots_to_solution_crude'], 0)} |")
    lines.append("")
(OUT / "candidates_table.md").write_text("\n".join(lines) + "\n")
print(f"wrote {OUT / 'candidates_table.md'} ({len(rows)} candidates)")
