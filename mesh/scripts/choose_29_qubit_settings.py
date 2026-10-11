"""Pick the QAOA settings for the 29-qubit run from the large-mode n = 20 results only (never from the n = 29 instances).

Rule, per device scenario:
  1. best fixed schedule: among every LR-QAOA schedule family and depth p = 1..5 tested at n = 20, the one with the highest
     mean P(cost <= 1.05 x optimum) over the 6 instances (2 cases x 3 seeds); no training on the target instance;
  2. trained angles: QAOA + COBYLA at the best trained depth at n = 20, transferred as the mean of its 6 optimised angle sets
     (the angles concentrate across instances; the spread is saved below);
  3. LR-QAOA p = 2 (energy-calibrated), the row used in the n <= 24 scaling table, for continuity.
usage: python scripts/choose_29_qubit_settings.py   ->  scripts/q29_choice.json
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
runs = pd.read_csv(ROOT / "results_large_hpc" / "task04_runs_large.csv")
checkpoint = json.load(gzip.open(ROOT / "results_large_hpc" / "task04_checkpoint_large.json.gz"))
at20 = runs[runs.n == 20]
choice, tables = {}, {}
for scenario, rows in at20.groupby("scenario"):
    fixed = rows[rows.variant.str.startswith("LR-QAOA") & ~rows.variant.str.contains("ablation")]
    fixed_mean = fixed.groupby(["variant", "p"])["p_near"].mean()
    best_variant, best_p = fixed_mean.idxmax()
    trained = rows[(rows.variant == "QAOA") & (rows.optimiser == "COBYLA")].groupby("p")["p_near"].mean()
    trained_p = int(trained.idxmax())
    angles = {}
    for p in (1, 2):
        sets = [run["final_angles"] for key, record in checkpoint["instances"].items()
                if key.split("|")[1] == "20" and key.split("|")[3] == scenario and record.get("qaoa")
                for group in record["qaoa"].values() if group["variant"] == "QAOA" and group["optimiser"] == "COBYLA" and group["p"] == p
                for run in group["runs"]]
        angles[str(p)] = dict(mean=np.mean(sets, axis=0).tolist(), std=np.std(sets, axis=0).tolist(), n_sets=len(sets))
    choice[scenario] = dict(best_fixed=dict(variant=best_variant, p=int(best_p), p_near_n20=float(fixed_mean.max())),
                            trained=dict(p=trained_p, p_near_n20=float(trained.max()), angles=angles[str(trained_p)]["mean"]),
                            continuity=dict(variant="LR-QAOA", p=2, p_near_n20=float(fixed_mean[("LR-QAOA", 2)])),
                            cobyla_angles_n20=angles)
    tables[scenario] = {f"{variant} p={p}": float(v) for (variant, p), v in fixed_mean.items()}
    tables[scenario].update({f"QAOA + COBYLA p={p}": float(v) for p, v in trained.items()})
out = dict(rule=__doc__.split("usage:")[0].strip(), source="results_large_hpc/task04_runs_large.csv (n = 20, 2 cases x 3 seeds)",
           choice=choice, n20_mean_p_near=tables)
(ROOT / "scripts" / "q29_choice.json").write_text(json.dumps(out, indent=1))
for scenario, c in choice.items():
    print(f"{scenario}: best fixed {c['best_fixed']}, trained COBYLA p = {c['trained']['p']} ({c['trained']['p_near_n20']:.2e})")
